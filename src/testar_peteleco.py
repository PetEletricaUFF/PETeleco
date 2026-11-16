#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bancada Interativa de Teste — PETeleco (Webcam + Voz + STT + Wake/Roteador + Persona + TTS).

Permite testar ponta a ponta no PC (Linux ou Windows):
- Captura continua de webcam (`WebcamLive`) com deteccao/reconhecimento `YuNet + SFace` e HUD opcional (`--preview`)
- Captura de microfone (VAD com pre-roll) e transcricao STT (`Gemini Audio` / `faster-whisper small`)
- Biometria de voz (`IdentificadorVoz`) mesmo sem camera/rosto
- Modo Escuta Ativa por wake-word (`"PETeleco"` abre janela rolante de 60s sem descartar por ruido)
- Roteador Jev `/v1/systemone` (`tev1:0.8b`) e modo rapido `--roteador auto|gemini|jev`
- Persona v3.2 adaptativa por pessoa, ritmo conversacional e consciencia corporal (`chao` / `bancada` / `no_colo`)
- Memoria evolutiva em 3 camadas (`data/perfis_pessoas.json` + `SQLite`)
- Leitura em voz alta natural (`edge-tts pt-BR-AntonioNeural`, `gemini-tts` ou `Piper TTS` offline)
"""

import argparse
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.escuta_stt import EscutaSTT
from src.memoria_pessoas import GerenciadorMemoriaPessoas, normalizar_chave
from src.persona_peteleco import EstadoCorporal, MotorPersonaPETeleco, parece_pedido_visual
from src.router_systemone import ContextoRover, DecisaoRoteador, RoteadorSystemOne
from src.tts_local import DEFAULT_EDGE_VOICE, DEFAULT_PIPER_VOICE, SintetizadorTTS
from src.visao_webcam import WebcamLive
from src.voz_biometria import IdentificadorVoz


def _estado_para_modo_jev(estado: EstadoCorporal) -> str:
    mapa = {
        "parado": "STANDBY",
        "andando": "APROXIMA",
        "seguindo_rosto": "CENTRALIZA",
        "radar": "RADAR",
    }
    return mapa.get(estado.movimento, "STANDBY")


def _comando_ollama_hint() -> str:
    if sys.platform.startswith("win"):
        return "$env:OLLAMA_VULKAN='0'; ollama serve"
    return "ollama serve"


def _imprimir_ajuda() -> None:
    print(
        "\nComandos da bancada:\n"
        "  [ENTER vazio]         -> Grava pelo microfone, identifica voz/rosto, STT, roda Jev e responde\n"
        "  <qualquer frase>      -> Envia texto direto para Jev + Persona + Memoria (+ webcam se pedir visao)\n"
        "  /continuo             -> Entra em escuta continua maos-livres (diga 'PETeleco' para Escuta Ativa 60s; 'Q'/Ctrl+C sai)\n"
        "  /preview              -> Abre/fecha a janela HUD da webcam com caixas faciais YuNet+SFace\n"
        "  /cam                  -> Tira 1 foto da webcam agora e testa reconhecimento facial YuNet+SFace\n"
        "  /rosto <id_ou_nome>   -> Cadastra/atualiza o rosto atual da webcam em data/known_faces.json\n"
        "  /voz <id_ou_apelido>  -> Grava 4s no mic para cadastrar voz (ex: /voz caio, /voz to2)\n"
        "  /testar_voz           -> Grava 4s no mic e mostra ranking de similaridade cosseno + STT\n"
        "  /foto <arq> [frase]   -> Envia imagem JPEG/PNG com reconhecimento facial SFace + Gemini\n"
        "  /tts                  -> Liga/desliga leitura da resposta em voz alta (Edge Neural / Gemini / Piper)\n"
        "  /pessoa <id|auto>     -> Simula rosto SFace na camera (ou 'auto' para webcam/voz real)\n"
        "  /corpo <chao|bancada|no_colo> -> Muda superficie fisica simulada do robo\n"
        "  /modo <normal|serio>  -> Alterna MODO_PERSONA manualmente\n"
        "  /perfil [id]          -> Mostra ficha quente JSON + ultimos episodios no SQLite\n"
        "  /jev <frase>          -> Inspeciona todas as probabilidades do /v1/systemone para a frase\n"
        "  /mics                 -> Lista dispositivos de microfone disponiveis\n"
        "  /ajuda | /sair        -> Mostra comandos / encerra\n"
    )


def _verificar_jev(roteador: RoteadorSystemOne) -> tuple[bool, str]:
    try:
        versao = roteador.verificar_servidor()
        return True, versao
    except Exception:
        return False, "offline"


def garantir_ollama_local(
    roteador: RoteadorSystemOne,
    timeout_s: float = 12.0,
) -> tuple[bool, str, Optional[subprocess.Popen[Any]]]:
    """Verifica Ollama e, se for local e estiver offline, inicia `ollama serve` automaticamente."""
    online, versao = _verificar_jev(roteador)
    if online:
        return True, versao, None

    host = (urlparse(roteador.base_url).hostname or "").lower()
    if host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return False, "offline", None

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return False, "offline", None

    env = os.environ.copy()
    if sys.platform.startswith("win"):
        env.setdefault("OLLAMA_VULKAN", "0")

    try:
        proc = subprocess.Popen(
            [ollama_bin, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return False, "offline", None

    limite = time.monotonic() + timeout_s
    while time.monotonic() < limite:
        if proc.poll() is not None:
            break
        online, versao = _verificar_jev(roteador)
        if online:
            return True, versao, proc
        time.sleep(0.35)

    try:
        proc.terminate()
    except Exception:
        pass
    return False, "offline", None


def _sincronizar_hud(
    webcam: Optional[WebcamLive],
    status: str,
    pessoa: str,
    estado: EstadoCorporal,
    motor: MotorPersonaPETeleco,
    jev_online: bool,
    tts_ativo: bool,
    fala_user: Optional[str] = None,
    fala_robo: Optional[str] = None,
) -> None:
    if webcam is None:
        return
    campos = {
        "status": status,
        "pessoa": pessoa,
        "corpo": f"{estado.superficie}/{estado.movimento}",
        "modo": motor.modo_persona,
        "jev": "ON" if jev_online else "OFF",
        "tts": "ON" if tts_ativo else "OFF",
    }
    if fala_user is not None:
        campos["fala_user"] = fala_user
    if fala_robo is not None:
        campos["fala_robo"] = fala_robo
    webcam.atualizar_hud(**campos)


def _capturar_b64_seguro(
    webcam: Optional[WebcamLive],
    motor: MotorPersonaPETeleco,
    camera_idx: int,
) -> Optional[str]:
    if webcam is not None and webcam.disponivel:
        return webcam.capturar_b64()
    cam_temp = WebcamLive(camera_idx=camera_idx, detector_faces=motor.detectar_e_reconhecer_faces, preview=False)
    try:
        if cam_temp.iniciar(timeout_s=3.0):
            return cam_temp.capturar_b64()
    finally:
        cam_temp.fechar()
    return None


_PADROES_WAKE = (
    r"\bpeteleco\b",
    r"\bpeteleko\b",
    r"\bpe\s+teleco\b",
    r"\bpe\s+teleko\b",
    r"\bpet\s+teleco\b",
    r"\bpete\s+leco\b",
    r"\bteleco\b",
)

_FRASES_ENCERRAR_ESCUTA = (
    "pode dormir",
    "pode descansar",
    "encerrar escuta",
    "desliga a escuta",
    "entra em standby",
    "era so isso",
    "valeu era so isso",
)


@dataclass
class EstadoEscutaAtiva:
    """Controla a janela de conversa aberta apos o usuario chamar 'PETeleco'."""

    janela_s: float = 60.0
    exigir_wake_continuo: bool = True
    ativo_ate: float = 0.0

    def esta_ativa(self) -> bool:
        return time.monotonic() < self.ativo_ate

    def restante_s(self) -> int:
        return max(0, int(round(self.ativo_ate - time.monotonic())))

    def ativar(self) -> None:
        self.ativo_ate = time.monotonic() + max(10.0, float(self.janela_s))

    def encerrar(self) -> None:
        self.ativo_ate = 0.0


def detectar_wake_word(transcricao: str) -> Tuple[bool, str]:
    """
    Detecta 'PETeleco' (e variacoes foneticas do STT) e separa eventual comando apos o chamado.
    """
    bruto = (transcricao or "").strip()
    norm = normalizar_chave(bruto)
    if not norm:
        return False, ""

    acordou = any(re.search(p, norm) for p in _PADROES_WAKE)
    if not acordou:
        return False, bruto

    # Remove prefixo opcional ("ei peteleco", "fala peteleco", "o peteleco") se houver comando na sequencia
    limpo = re.sub(
        r"^(?:(?:ei|oi|ola|fala|e\s+ai|eai|o|ô)\s+)?(?:p[eé]\s*tele[ck]o|pet\s*tele[ck]o|pete\s*leco|tele[ck]o)\b[\s,.:;!?-]*",
        "",
        bruto,
        count=1,
        flags=re.IGNORECASE,
    ).strip()

    # Se havia apenas saudacao curta + wake word (ex: "Fala PETeleco!"), retorna vazio para ack rapido de ativacao
    norm_resto = normalizar_chave(limpo)
    if not norm_resto or norm_resto in ("ai", "aqui", "mano", "meu chapa", "meu amigo"):
        return True, ""
    return True, limpo


def _eh_pedido_encerrar_escuta(transcricao: str) -> bool:
    norm = normalizar_chave(transcricao)
    return any(f in norm for f in _FRASES_ENCERRAR_ESCUTA)


def _decisao_para_dict(dec: Optional[DecisaoRoteador]) -> Optional[Dict[str, Any]]:
    if dec is None:
        return None
    return {
        "rota_final": dec.rota_final,
        "subtipo": dec.subtipo,
        "confianca": round(float(dec.confianca), 4),
        "dirigido_ao_robo": round(float(dec.dirigido_ao_robo), 4),
        "requer_confirmacao": bool(dec.requer_confirmacao),
        "precisa_imagem_camera": bool(dec.precisa_imagem_camera),
        "latencia_ms": round(float(dec.latencia_ms), 1),
        "respostas_brutas": dec.respostas_brutas,
    }


def executar_turno_detalhado(
    *,
    entrada_texto: str,
    imagem_b64: Optional[str],
    usou_comando_foto: bool,
    face_simulada: str,
    memoria: GerenciadorMemoriaPessoas,
    voz_bio: IdentificadorVoz,
    stt: EscutaSTT,
    estado: EstadoCorporal,
    motor: MotorPersonaPETeleco,
    roteador: RoteadorSystemOne,
    tts: SintetizadorTTS,
    webcam: Optional[WebcamLive],
    camera_idx: int,
    jev_online: bool,
    sem_jev: bool,
    tts_ativo: bool,
    limiar_mic: float,
    roteador_modo: str = "auto",
    escuta_ativa_ctrl: Optional[EstadoEscutaAtiva] = None,
    origem_manual_ou_ui: bool = True,
    silencioso_timeout: bool = False,
    stop_event: Optional[threading.Event] = None,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Executa um turno completo com telemetria por estagio (compartilhada entre CLI e UI)."""

    def _emit(tipo: str, **payload: Any) -> None:
        if on_event is not None:
            try:
                on_event(tipo, payload)
            except Exception:
                pass

    t_turno = time.perf_counter()
    tempos_ms: Dict[str, float] = {
        "stt_ms": 0.0,
        "face_ms": 0.0,
        "jev_ms": 0.0,
        "persona_ms": 0.0,
        "tts_ms": 0.0,
        "total_ms": 0.0,
    }

    voice_id = "desconhecido"
    voice_conf = 0.0
    voice_ranking: Dict[str, float] = {}
    stt_backend = "texto"
    audio_mic: Any = None
    transcricao = entrada_texto.strip()
    label_hud = face_simulada if face_simulada != "nenhuma" else ("cam+voz" if (webcam and webcam.disponivel) else "auto")

    if not transcricao:
        if not silencioso_timeout:
            print("[MIC] Ouvindo... fale agora (para sozinho apos 1,2s de silencio).")
        _sincronizar_hud(webcam, "OUVINDO...", label_hud, estado, motor, jev_online, tts_ativo)
        _emit("stage", stage="vad", mensagem="Aguardando voz no microfone (VAD RMS)...")
        audio_mic = stt.gravar_microfone(
            limiar_rms=limiar_mic,
            stop_event=stop_event,
            on_speech_start=lambda: (
                _sincronizar_hud(webcam, "CAPTANDO FALA", label_hud, estado, motor, jev_online, tts_ativo),
                _emit("stage", stage="gravando", mensagem="Fala detectada no microfone, gravando..."),
            ),
        )
        if audio_mic.size == 0:
            if not silencioso_timeout:
                print("[MIC] Nenhuma fala detectada.")
            _sincronizar_hud(webcam, "PRONTO", label_hud, estado, motor, jev_online, tts_ativo)
            _emit("stage", stage="pronto", mensagem="Silencio no VAD / captura interrompida.")
            return {"respondeu": False, "motivo": "sem_fala", "tempos_ms": tempos_ms}

        _sincronizar_hud(webcam, "PENSANDO (STT)...", label_hud, estado, motor, jev_online, tts_ativo)
        _emit("stage", stage="stt", mensagem="Transcrevendo fala e extraindo biometria vocal...")
        voice_id, voice_conf, voice_ranking = voz_bio.identificar(audio_mic, sr=stt.sample_rate)
        transcricao, stt_backend, lat_stt = stt.transcrever(audio_mic, sr=stt.sample_rate)
        tempos_ms["stt_ms"] = round(float(lat_stt), 1)
        print(f"[STT {stt_backend} | {lat_stt:.0f} ms]: \"{transcricao}\"")
        if voice_id != "desconhecido":
            print(f"[VOZ] Locutor reconhecido pela voz: {voice_id} ({voice_conf:.2f})")
        _emit(
            "stt",
            transcricao=transcricao,
            backend=stt_backend,
            latencia_ms=tempos_ms["stt_ms"],
            voice_id=voice_id,
            voice_conf=round(float(voice_conf), 4),
            voice_ranking={k: round(float(v), 4) for k, v in voice_ranking.items()},
        )
        if not transcricao:
            if not silencioso_timeout:
                print("[AVISO] Nao consegui transcrever a fala.")
            _sincronizar_hud(webcam, "PRONTO", label_hud, estado, motor, jev_online, tts_ativo)
            _emit("stage", stage="pronto", mensagem="Audio sem transcricao valida.")
            return {"respondeu": False, "motivo": "sem_transcricao", "tempos_ms": tempos_ms}
    else:
        _emit("stt", transcricao=transcricao, backend="texto", latencia_ms=0.0, voice_id="desconhecido", voice_conf=0.0)

    acordou_wake, comando_pos_wake = detectar_wake_word(transcricao)
    escuta_ativa_agora = bool(escuta_ativa_ctrl is not None and escuta_ativa_ctrl.esta_ativa())

    # Em modo continuo Standby (antes de ouvir 'PETeleco'), descarta fala sem wake-word em 0 ms
    if (
        not origem_manual_ou_ui
        and escuta_ativa_ctrl is not None
        and escuta_ativa_ctrl.exigir_wake_continuo
        and not escuta_ativa_agora
        and not acordou_wake
    ):
        tempos_ms["total_ms"] = round((time.perf_counter() - t_turno) * 1000.0, 1)
        print(f"[STANDBY] Aguardando 'PETeleco' (ignorado: \"{transcricao}\").")
        _sincronizar_hud(webcam, "STANDBY", label_hud, estado, motor, jev_online, tts_ativo)
        _emit("log", nivel="info", canal="WAKE", texto=f"Standby: ignorado sem wake-word ('{transcricao[:70]}').")
        _emit("stage", stage="pronto", mensagem="Standby ativo: diga 'PETeleco' para iniciar escuta ativa.")
        return {
            "respondeu": False,
            "motivo": "sem_wake_word",
            "transcricao": transcricao,
            "stt_backend": stt_backend,
            "tempos_ms": tempos_ms,
        }

    apenas_chamado_wake = False
    if acordou_wake:
        escuta_ativa_agora = True
        if escuta_ativa_ctrl is not None:
            escuta_ativa_ctrl.ativar()
            _emit(
                "log",
                nivel="ok",
                canal="WAKE",
                texto=f"Wake-word 'PETeleco' detectada! Escuta Ativa por {int(escuta_ativa_ctrl.janela_s)}s.",
            )
        if not comando_pos_wake:
            apenas_chamado_wake = True
        else:
            transcricao = comando_pos_wake
    elif escuta_ativa_agora and escuta_ativa_ctrl is not None:
        escuta_ativa_ctrl.ativar()

    modo_escuta_turno = bool(escuta_ativa_agora or origem_manual_ou_ui)

    _sincronizar_hud(
        webcam, "PENSANDO...", label_hud, estado, motor, jev_online, tts_ativo, fala_user=transcricao
    )

    frame_cam_turno: Optional[str] = None
    face_turno = face_simulada
    webcam_ativa = bool(webcam is not None and webcam.disponivel)
    _emit("stage", stage="visao", mensagem="Analisando biometria facial (YuNet + SFace)...")
    t0_face = time.perf_counter()
    if face_turno == "nenhuma" and imagem_b64:
        face_auto = motor.reconhecer_rosto_imagem_b64(imagem_b64)
        if face_auto != "nenhuma":
            face_turno = face_auto
            print(f"[SFACE] Rosto detectado automaticamente na foto: {face_auto}")
    elif face_turno == "nenhuma" and webcam_ativa:
        frame_cam_turno = _capturar_b64_seguro(webcam, motor, camera_idx)
        if frame_cam_turno:
            face_auto = motor.reconhecer_rosto_imagem_b64(frame_cam_turno)
            if face_auto != "nenhuma":
                face_turno = face_auto
                print(f"[SFACE CAM] Rosto detectado ao vivo: {face_auto}")
    tempos_ms["face_ms"] = round((time.perf_counter() - t0_face) * 1000.0, 1)

    auto_voz_salva = 0
    if audio_mic is not None and face_turno != "nenhuma" and voice_id == "desconhecido":
        pid_face = memoria.mapear_identificador(face_turno.split("(")[0].strip())
        if pid_face and pid_face != "desconhecido":
            n = voz_bio.cadastrar_amostra(pid_face, audio_mic, sr=stt.sample_rate)
            if n > 0:
                auto_voz_salva = n
                print(f"[AUTO-VOZ] Amostra vocal salva automaticamente para {pid_face} ({n} total).")

    _emit(
        "biometria",
        face_id=face_turno,
        voice_id=voice_id,
        voice_conf=round(float(voice_conf), 4),
        auto_voz_salva=auto_voz_salva,
        face_ms=tempos_ms["face_ms"],
    )

    decisao_jev: Optional[DecisaoRoteador] = None
    api_key_gem = motor._env.get("GEMINI_API_KEY", "").strip()
    if apenas_chamado_wake:
        pessoa_id_w, perfil_w, modo_w, display_w = memoria.resolver_pessoa(
            transcricao=transcricao,
            face_id=face_turno,
            voice_id=voice_id,
            voice_conf=voice_conf,
        )
        nome_w = perfil_w.get("nome", "")
        voc_w = f", {nome_w}" if nome_w and nome_w != "desconhecido" else ""
        if motor.eh_modo_formal(perfil_w):
            falas_wake = [
                f"Sim{', professor Felipe' if perfil_w.get('papel') == 'professor_orientador' else voc_w}? Estou na escuta.",
                "Pois não? Estou ouvindo perfeitamente.",
            ]
        else:
            falas_wake = [
                f"Opa{voc_w}! Haha, tô na escuta! Pode falar.",
                f"Fala{voc_w}! Tô ligado aqui, manda a boa.",
                f"Ôpa{voc_w}, acordei! Qual é a parada agora?",
            ]
        resp_wake = random.choice(falas_wake)
        resultado = {
            "resposta": resp_wake,
            "resposta_tts": resp_wake,
            "origem": "local:wake_word",
            "pessoa_id": pessoa_id_w,
            "pessoa_display": display_w,
            "modo_identificacao": modo_w,
            "modo_persona": "SERIO" if motor.eh_modo_formal(perfil_w) else motor.modo_persona,
            "estado_corporal": estado.formatar_prompt(),
        }
        precisa_imagem = False
        imagem_para_llm = None
        tts_backend = ""
        print(f"PETeleco: {resp_wake}")
        if tts_ativo:
            _sincronizar_hud(webcam, "FALANDO...", display_w, estado, motor, jev_online, tts_ativo, fala_user=transcricao, fala_robo=resp_wake)
            _emit("stage", stage="tts", mensagem=f"Sintetizando voz ({tts.descrever_backend()})...")
            t0_tts = time.perf_counter()
            tts_backend = tts.falar(resp_wake)
            tempos_ms["tts_ms"] = round((time.perf_counter() - t0_tts) * 1000.0, 1)
        if escuta_ativa_ctrl is not None:
            escuta_ativa_ctrl.ativar()
        _sincronizar_hud(webcam, "PRONTO", display_w, estado, motor, jev_online, tts_ativo, fala_user=transcricao, fala_robo=resp_wake)
        tempos_ms["total_ms"] = round((time.perf_counter() - t_turno) * 1000.0, 1)
        detalhes = {
            "respondeu": True,
            "transcricao": transcricao,
            "stt_backend": stt_backend,
            "face_id": face_turno,
            "voice_id": voice_id,
            "voice_conf": round(float(voice_conf), 4),
            "precisa_imagem": False,
            "usou_imagem": False,
            "decisao_jev": None,
            "resultado": resultado,
            "tts_backend": tts_backend,
            "escuta_ativa": True,
            "tempos_ms": tempos_ms,
        }
        _emit("stage", stage="pronto", mensagem="Escuta Ativa ligada: fale naturalmente!")
        return detalhes

    usar_gemini_router = False
    if roteador_modo == "gemini":
        usar_gemini_router = bool(api_key_gem)
    elif roteador_modo == "auto" and not sem_jev:
        usar_gemini_router = ((not torch.cuda.is_available()) or (not jev_online)) and bool(api_key_gem)

    pessoa_ctx = face_turno
    if pessoa_ctx == "nenhuma" and voice_id != "desconhecido":
        pessoa_ctx = f"{memoria.obter_perfil(voice_id)['nome']} ({int(round(voice_conf * 100))}% voz)"
    ctx = ContextoRover(
        transcricao=transcricao,
        estado_rover=_estado_para_modo_jev(estado),
        pessoa_sface=pessoa_ctx,
        objetos_yolo=estado.objetos_yolo,
        modo_escuta_ativa=modo_escuta_turno,
    )

    if usar_gemini_router or not sem_jev:
        if usar_gemini_router:
            _emit("stage", stage="jev", mensagem="Classificando intencao via Gemini Flash-Lite...")
            gem_model_router = (motor._env.get("GEMINI_ROUTER_MODEL") or "gemini-3.5-flash-lite").strip()
            decisao_jev = roteador.decidir_gemini(ctx, api_key=api_key_gem, model=gem_model_router)
            if decisao_jev is not None:
                tempos_ms["jev_ms"] = round(float(decisao_jev.latencia_ms), 1)
                print(
                    f"[ROTEADOR GEMINI {decisao_jev.latencia_ms:.1f} ms] rota={decisao_jev.rota_final}:{decisao_jev.subtipo} "
                    f"(conf={decisao_jev.confianca:.2f}, dir={decisao_jev.dirigido_ao_robo:.2f}, "
                    f"escuta_ativa={modo_escuta_turno}, img={decisao_jev.precisa_imagem_camera})"
                )
                _emit("jev", decisao=_decisao_para_dict(decisao_jev))

        if decisao_jev is None and jev_online and not sem_jev:
            _emit("stage", stage="jev", mensagem="Consultando roteador Jev (/v1/systemone)...")
            try:
                decisao_jev = roteador.decidir(ctx)
                tempos_ms["jev_ms"] = round(float(decisao_jev.latencia_ms), 1)
                print(
                    f"[JEV {decisao_jev.latencia_ms:.1f} ms] rota={decisao_jev.rota_final}:{decisao_jev.subtipo} "
                    f"(conf={decisao_jev.confianca:.2f}, dir={decisao_jev.dirigido_ao_robo:.2f}, "
                    f"confirmar={decisao_jev.requer_confirmacao}, img={decisao_jev.precisa_imagem_camera})"
                )
                _emit("jev", decisao=_decisao_para_dict(decisao_jev))
            except Exception as exc:
                print(f"[AVISO JEV] Falha ao consultar SystemOne ({exc}); seguindo direto para Persona.")
                _emit("log", nivel="aviso", canal="JEV", texto=f"Falha no /v1/systemone ({exc}); indo direto para Persona.")
                decisao_jev = None

    precisa_imagem = (
        usou_comando_foto
        or bool(decisao_jev and decisao_jev.precisa_imagem_camera)
        or parece_pedido_visual(transcricao)
    )
    imagem_para_llm = imagem_b64
    if precisa_imagem and not imagem_para_llm:
        if frame_cam_turno is None and webcam_ativa:
            frame_cam_turno = _capturar_b64_seguro(webcam, motor, camera_idx)
        imagem_para_llm = frame_cam_turno
        if imagem_para_llm:
            print("[CAM] Frame da webcam anexado ao turno visual.")
            _emit("log", nivel="info", canal="CAM", texto="Frame da webcam anexado para raciocinio visual multimodal.")

    decisao_para_motor = decisao_jev
    if (
        precisa_imagem
        and imagem_para_llm
        and decisao_jev is not None
        and decisao_jev.rota_final in ("ignorar", "pool_fala_pronta")
    ):
        decisao_para_motor = None

    _emit(
        "stage",
        stage="persona",
        mensagem="Aplicando travas fisicas, memoria e gerando resposta (Local / Gemini)...",
    )
    t0_persona = time.perf_counter()
    resultado = motor.responder(
        transcricao=transcricao,
        decisao_jev=decisao_para_motor,
        face_id=face_turno,
        voice_id=voice_id,
        voice_conf=voice_conf,
        imagem_jpeg_b64=imagem_para_llm,
    )
    tempos_ms["persona_ms"] = round((time.perf_counter() - t0_persona) * 1000.0, 1)
    print(
        f"[CONTEXTO] pessoa={resultado['pessoa_display']} | id={resultado['modo_identificacao']} "
        f"| modo={resultado['modo_persona']} | origem={resultado['origem']}"
    )

    tts_backend = ""
    if resultado["resposta"]:
        print(f"PETeleco: {resultado['resposta']}")
        if resultado["resposta_tts"] != resultado["resposta"]:
            print(f"   [TTS]: {resultado['resposta_tts']}")
        if tts_ativo:
            _sincronizar_hud(
                webcam,
                "FALANDO...",
                resultado["pessoa_display"],
                estado,
                motor,
                jev_online,
                tts_ativo,
                fala_user=transcricao,
                fala_robo=resultado["resposta"],
            )
            _emit("stage", stage="tts", mensagem=f"Sintetizando voz ({tts.descrever_backend()})...")
            t0_tts = time.perf_counter()
            tts_backend = tts.falar(resultado["resposta_tts"])
            tempos_ms["tts_ms"] = round((time.perf_counter() - t0_tts) * 1000.0, 1)

        eh_despedida = _eh_pedido_encerrar_escuta(transcricao) or bool(
            decisao_jev and decisao_jev.rota_final == "pool_fala_pronta" and decisao_jev.subtipo == "despedida"
        )
        if escuta_ativa_ctrl is not None:
            if eh_despedida:
                escuta_ativa_ctrl.encerrar()
                _emit("log", nivel="info", canal="WAKE", texto="Escuta Ativa encerrada (despedida); voltando para Standby.")
            elif escuta_ativa_agora:
                # Renova janela pos-fala do robo para o tempo de audio nao consumir a escuta
                escuta_ativa_ctrl.ativar()

        _sincronizar_hud(
            webcam,
            "PRONTO",
            resultado["pessoa_display"],
            estado,
            motor,
            jev_online,
            tts_ativo,
            fala_user=transcricao,
            fala_robo=resultado["resposta"],
        )
        tempos_ms["total_ms"] = round((time.perf_counter() - t_turno) * 1000.0, 1)
        detalhes = {
            "respondeu": True,
            "transcricao": transcricao,
            "stt_backend": stt_backend,
            "face_id": face_turno,
            "voice_id": voice_id,
            "voice_conf": round(float(voice_conf), 4),
            "precisa_imagem": bool(precisa_imagem),
            "usou_imagem": bool(imagem_para_llm),
            "decisao_jev": _decisao_para_dict(decisao_jev),
            "resultado": resultado,
            "tts_backend": tts_backend,
            "escuta_ativa": bool(escuta_ativa_ctrl is not None and escuta_ativa_ctrl.esta_ativa()),
            "tempos_ms": tempos_ms,
        }
        _emit("stage", stage="pronto", mensagem="Turno concluido.")
        return detalhes

    print("[IGNORADO] Fala classificada como conversa paralela / ruido.")
    _sincronizar_hud(
        webcam, "PRONTO", resultado["pessoa_display"], estado, motor, jev_online, tts_ativo, fala_user=transcricao
    )
    tempos_ms["total_ms"] = round((time.perf_counter() - t_turno) * 1000.0, 1)
    detalhes = {
        "respondeu": False,
        "motivo": "ignorar",
        "transcricao": transcricao,
        "stt_backend": stt_backend,
        "face_id": face_turno,
        "voice_id": voice_id,
        "voice_conf": round(float(voice_conf), 4),
        "precisa_imagem": bool(precisa_imagem),
        "usou_imagem": False,
        "decisao_jev": _decisao_para_dict(decisao_jev),
        "resultado": resultado,
        "tts_backend": "",
        "tempos_ms": tempos_ms,
    }
    _emit("stage", stage="pronto", mensagem="Fala ignorada (conversa paralela / ruido).")
    return detalhes


def processar_turno(**kwargs: Any) -> bool:
    """Executa um turno completo e retorna True se o robo respondeu."""
    return bool(executar_turno_detalhado(**kwargs).get("respondeu"))


def _executar_loop_continuo(**kwargs: Any) -> None:
    webcam: Optional[WebcamLive] = kwargs["webcam"]
    ctrl: Optional[EstadoEscutaAtiva] = kwargs.get("escuta_ativa_ctrl")
    if ctrl is None:
        ctrl = EstadoEscutaAtiva()
        kwargs["escuta_ativa_ctrl"] = ctrl
    exige_wake = ctrl.exigir_wake_continuo
    print("\n[CONTINUO] Modo maos-livres ativo! Fale naturalmente olhando para a webcam.")
    if exige_wake:
        print(f"[CONTINUO] Diga 'PETeleco' para entrar em Escuta Ativa ({int(ctrl.janela_s)}s sem descartar ruido).")
    print("[CONTINUO] Pressione Ctrl+C no terminal (ou 'Q' na janela de preview) para sair.\n")
    try:
        while True:
            if webcam is not None and webcam.fechado_pelo_usuario:
                print("\n[CONTINUO] Janela de preview fechada; saindo do modo continuo.")
                break
            falou = processar_turno(
                entrada_texto="",
                imagem_b64=None,
                usou_comando_foto=False,
                silencioso_timeout=True,
                origem_manual_ou_ui=False,
                **kwargs,
            )
            if falou and kwargs.get("tts_ativo"):
                # Evita que o eco da caixa de som re-dispare o VAD imediatamente
                time.sleep(0.30)
    except KeyboardInterrupt:
        print("\n[CONTINUO] Encerrando modo continuo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bancada de teste PETeleco (Webcam + Voz + STT + Jev + Persona)")
    parser.add_argument("--ui", action="store_true", help="Abre a UI web local com webcam + visao debaixo dos panos")
    parser.add_argument("--porta", type=int, default=8765, help="Porta da UI web local (padrao: 8765)")
    parser.add_argument("--sem-browser", action="store_true", help="Nao abre o navegador automaticamente com --ui")
    parser.add_argument("--auto-ollama", action="store_true", help="Sobe `ollama serve` automaticamente se estiver offline")
    parser.add_argument("--sem-jev", action="store_true", help="Pula roteador /v1/systemone e vai direto para a Persona")
    parser.add_argument("--sem-webcam", action="store_true", help="Desativa captura automatica de webcam na bancada")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam local (padrao: 0)")
    parser.add_argument("--preview", action="store_true", help="Abre janela ao vivo da webcam com HUD YuNet+SFace")
    parser.add_argument("--continuo", action="store_true", help="Modo maos-livres continuo (escuta VAD + webcam)")
    parser.add_argument("--sem-wake-word", action="store_true", help="No modo continuo, nao exige dizer 'PETeleco' antes")
    parser.add_argument("--tts", action="store_true", help="Ativa leitura das respostas em voz alta (Edge Neural / Piper)")
    parser.add_argument("--tts-modo", choices=["auto", "edge", "gemini", "piper"], default="auto", help="Engine de voz TTS (padrao: auto -> Edge Neural)")
    parser.add_argument("--tts-voz", default=DEFAULT_EDGE_VOICE, help=f"Voz neural Edge PT-BR (padrao: {DEFAULT_EDGE_VOICE})")
    parser.add_argument("--piper-voz", default=DEFAULT_PIPER_VOICE, help="Modelo Piper PT-BR offline (padrao: pt_BR-faber-medium)")
    parser.add_argument("--stt-modo", choices=["auto", "gemini", "whisper"], default="auto", help="Backend de transcricao STT (padrao: auto)")
    parser.add_argument(
        "--whisper-modelo",
        default="auto",
        help="Tamanho do faster-whisper local: tiny, base, small, medium (auto: small)",
    )
    parser.add_argument("--roteador", choices=["auto", "gemini", "jev"], default="auto", help="Roteador de intencoes (auto: Jev na CUDA / Gemini na CPU)")
    parser.add_argument("--mic", type=int, default=None, help="Indice do microfone (veja com /mics)")
    parser.add_argument("--limiar-mic", type=float, default=0.012, help="Limiar RMS do VAD do microfone (padrao: 0.012)")
    parser.add_argument("--superficie", choices=["chao", "bancada", "no_colo"], default="chao")
    parser.add_argument("--pessoa", default="auto", help="Pessoa simulada no SFace inicial (padrao: auto)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    if args.ui:
        from src.ui_peteleco import executar_ui

        executar_ui(args)
        return

    whisper_modelo = args.whisper_modelo
    if whisper_modelo == "auto":
        whisper_modelo = "small"

    memoria = GerenciadorMemoriaPessoas()
    voz_bio = IdentificadorVoz()
    stt = EscutaSTT(
        whisper_model_size=whisper_modelo,
        aliases_stt=memoria.listar_aliases_stt(),
        device_index=args.mic,
        stt_modo=args.stt_modo,
    )
    estado = EstadoCorporal(superficie=args.superficie)
    motor = MotorPersonaPETeleco(memoria=memoria, estado=estado, ollama_url=args.ollama_url)
    roteador = RoteadorSystemOne(base_url=args.ollama_url)
    tts = SintetizadorTTS(voice_name=args.piper_voz, tts_modo=args.tts_modo, edge_voice=args.tts_voz)
    escuta_ativa_ctrl = EstadoEscutaAtiva(exigir_wake_continuo=not args.sem_wake_word)

    jev_online = False
    jev_versao = "desativado"
    ollama_proc: Optional[subprocess.Popen[Any]] = None
    if not args.sem_jev:
        if args.auto_ollama:
            jev_online, jev_versao, ollama_proc = garantir_ollama_local(roteador)
        else:
            jev_online, jev_versao = _verificar_jev(roteador)

    webcam: Optional[WebcamLive] = None
    webcam_ativa = False
    if not args.sem_webcam:
        webcam = WebcamLive(
            camera_idx=args.camera,
            detector_faces=motor.detectar_e_reconhecer_faces,
            preview=args.preview,
        )
        webcam_ativa = webcam.iniciar()

    tts_ativo = bool(args.tts)
    face_simulada = "nenhuma"
    if args.pessoa != "auto":
        pid_arg = memoria.mapear_identificador(args.pessoa)
        if pid_arg:
            face_simulada = f"{memoria.obter_perfil(pid_arg)['nome']} (95%)"

    tem_gemini = bool(motor._env.get("GEMINI_API_KEY", "").strip())

    try:
        print("=" * 72)
        print("BANCADA PETELECO — Webcam + Voz + STT + Jev SystemOne + Persona v3.1")
        print("=" * 72)
        print(f"Biometria de Voz : {voz_bio.backend} | vozes cadastradas: {list(voz_bio.vozes.keys()) or 'nenhuma'}")
        print(f"STT Ativo        : {stt.descrever_backend()}")
        print(f"Perfis carregados: {len(memoria.perfis)} entradas ({memoria.perfis_path.name})")
        print(
            f"Webcam local     : {'ATIVA (YuNet+SFace)' if webcam_ativa else 'OFF (use /cam ou --camera N)'}"
            f"{' [HUD PREVIEW]' if (webcam_ativa and args.preview) else ''}"
        )
        print(f"Saida TTS        : {'ON' if tts_ativo else 'OFF (use /tts para ativar)'} ({tts.descrever_backend()})")
        print(f"Gemini API Key   : {'OK' if tem_gemini else 'NAO ENCONTRADA (.env — usando rotas locais / fallback Ollama)'}")
        roteador_label = args.roteador
        if args.roteador == "auto":
            roteador_label = "gemini (CPU fast)" if (not torch.cuda.is_available() and tem_gemini and not args.sem_jev) else "jev (CUDA)"
        print(f"Roteamento       : {roteador_label} | wake-word continuo={'OFF' if args.sem_wake_word else 'ON (PETeleco)'}")
        if jev_online:
            print(f"Jev /v1/systemone: ONLINE (Ollama {jev_versao} | modelo {roteador.model})")
        else:
            print(f"Jev /v1/systemone: OFFLINE (inicie com: {_comando_ollama_hint()})")

        label_init = face_simulada if face_simulada != "nenhuma" else ("cam+voz" if webcam_ativa else "auto")
        _sincronizar_hud(webcam, "PRONTO", label_init, estado, motor, jev_online, tts_ativo)

        if args.continuo:
            _executar_loop_continuo(
                face_simulada=face_simulada,
                memoria=memoria,
                voz_bio=voz_bio,
                stt=stt,
                estado=estado,
                motor=motor,
                roteador=roteador,
                tts=tts,
                webcam=webcam,
                camera_idx=args.camera,
                jev_online=jev_online,
                sem_jev=args.sem_jev,
                tts_ativo=tts_ativo,
                limiar_mic=args.limiar_mic,
                roteador_modo=args.roteador,
                escuta_ativa_ctrl=escuta_ativa_ctrl,
            )
            return

        _imprimir_ajuda()

        while True:
            try:
                label_face = face_simulada if face_simulada != "nenhuma" else ("cam+voz" if webcam_ativa else "auto")
                _sincronizar_hud(webcam, "PRONTO", label_face, estado, motor, jev_online, tts_ativo)
                status_prompt = (
                    f"\n[rosto={label_face} | corpo={estado.superficie}/{estado.movimento} "
                    f"| modo={motor.modo_persona} | jev={'ON' if jev_online else 'OFF'} | tts={'ON' if tts_ativo else 'OFF'}]\n> "
                )
                entrada = input(status_prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEncerrando bancada PETeleco.")
                break

            if entrada.lower() in ("/sair", "sair", "exit", "quit"):
                print("Encerrando bancada PETeleco.")
                break
            if entrada.lower() in ("/ajuda", "/help", "ajuda"):
                _imprimir_ajuda()
                continue

            if entrada.lower() == "/mics":
                for idx, nome in stt.listar_microfones():
                    print(f"  [{idx}] {nome}")
                continue

            if entrada.lower() == "/tts":
                tts_ativo = not tts_ativo
                print(f"[INFO] TTS em voz alta: {'ON' if tts_ativo else 'OFF'} ({tts.descrever_backend()})")
                continue

            if entrada.lower() == "/preview":
                if not webcam_ativa or webcam is None:
                    print("[AVISO] Webcam indisponivel.")
                else:
                    novo_prev = not webcam.preview_ativo
                    webcam.ativar_preview(novo_prev)
                    print(f"[INFO] Preview HUD da webcam: {'ON' if novo_prev else 'OFF'}")
                continue

            if entrada.lower() == "/continuo":
                _executar_loop_continuo(
                    face_simulada=face_simulada,
                    memoria=memoria,
                    voz_bio=voz_bio,
                    stt=stt,
                    estado=estado,
                    motor=motor,
                    roteador=roteador,
                    tts=tts,
                    webcam=webcam,
                    camera_idx=args.camera,
                    jev_online=jev_online,
                    sem_jev=args.sem_jev,
                    tts_ativo=tts_ativo,
                    limiar_mic=args.limiar_mic,
                    roteador_modo=args.roteador,
                    escuta_ativa_ctrl=escuta_ativa_ctrl,
                )
                continue

            if entrada.lower() == "/cam":
                frame_cam = _capturar_b64_seguro(webcam, motor, args.camera)
                if not frame_cam:
                    print(f"[ERRO] Nao consegui capturar frame da webcam {args.camera}.")
                else:
                    webcam_ativa = True
                    face_cam = motor.reconhecer_rosto_imagem_b64(frame_cam)
                    print(f"[CAM] Frame capturado! Reconhecimento YuNet+SFace: {face_cam}")
                continue

            if entrada.lower().startswith("/rosto"):
                arg = entrada[6:].strip()
                pid = memoria.mapear_identificador(arg) if arg else None
                if not pid:
                    print("[AVISO] Informe um membro valido. Ex.: /rosto caio | /rosto to2 | /rosto felipe_sass")
                    continue
                frame_cam = _capturar_b64_seguro(webcam, motor, args.camera)
                if not frame_cam:
                    print(f"[ERRO] Nao consegui capturar frame da webcam {args.camera}.")
                    continue
                nome_oficial = memoria.obter_perfil(pid)["nome"]
                ok, msg = motor.cadastrar_rosto_imagem_b64(nome_oficial, frame_cam)
                print(f"[{'OK' if ok else 'AVISO'}] {msg}")
                continue

            if entrada.lower().startswith("/pessoa"):
                arg = entrada[7:].strip()
                if not arg or arg.lower() == "auto":
                    face_simulada = "nenhuma"
                    print("[INFO] Rosto simulado resetado para 'auto' (identifica por webcam/voz).")
                else:
                    pid = memoria.mapear_identificador(arg)
                    if not pid:
                        print(f"[AVISO] Pessoa '{arg}' nao encontrada no roster. Use nome oficial ou apelido.")
                    else:
                        face_simulada = f"{memoria.obter_perfil(pid)['nome']} (95%)"
                        print(f"[INFO] Rosto SFace simulado agora: {face_simulada}")
                continue

            if entrada.lower().startswith("/corpo"):
                sup = entrada[6:].strip().lower()
                if sup in ("chao", "bancada", "no_colo"):
                    estado.superficie = sup
                    estado.movimento = "parado"
                    print(f"[INFO] Superficie atualizada: {estado.formatar_prompt()}")
                else:
                    print("[AVISO] Use: /corpo chao | /corpo bancada | /corpo no_colo")
                continue

            if entrada.lower().startswith("/modo"):
                m = entrada[5:].strip().upper()
                if m in ("NORMAL", "SERIO", "DESGRACADO"):
                    motor.modo_persona = m
                    print(f"[INFO] MODO_PERSONA = {motor.modo_persona}")
                else:
                    print("[AVISO] Use: /modo normal | /modo serio | /modo desgracado")
                continue

            if entrada.lower().startswith("/perfil"):
                arg = entrada[7:].strip()
                pid = (
                    memoria.mapear_identificador(arg)
                    if arg
                    else (
                        memoria.mapear_identificador(face_simulada.split("(")[0].strip())
                        if face_simulada != "nenhuma"
                        else "caio"
                    )
                )
                pid = pid or "desconhecido"
                print(f"\n--- Ficha Quente ({pid}) ---")
                print(json.dumps(memoria.obter_perfil(pid), ensure_ascii=False, indent=2))
                frios = memoria.buscar_memoria_fria(pid, limite=3)
                if frios:
                    print(f"--- Arquivo Frio SQLite (ultimos {len(frios)}) ---")
                    for ep in frios:
                        print(f"  [{ep['timestamp']}] {ep['fala_usuario']} -> {ep['resposta_robo']}")
                continue

            if entrada.lower().startswith("/voz"):
                arg = entrada[4:].strip()
                pid = memoria.mapear_identificador(arg) if arg else None
                if not pid:
                    print("[AVISO] Informe um membro valido. Ex.: /voz caio | /voz to2 | /voz felipe_sass")
                    continue
                nome = memoria.obter_perfil(pid)["nome"]
                print(f"[MIC] Fale naturalmente por 4 segundos para cadastrar a voz de {nome}...")
                audio = stt.gravar_duracao_fixa(4.0)
                n_amostras = voz_bio.cadastrar_amostra(pid, audio, sr=stt.sample_rate)
                texto_stt, backend_stt, lat_stt = stt.transcrever(audio, sr=stt.sample_rate)
                if n_amostras > 0:
                    print(f"[OK] Voz cadastrada para '{nome}' ({n_amostras} amostra(s) | backend {voz_bio.backend}).")
                else:
                    print("[AVISO] Audio curto ou baixo demais para extrair embedding vocal. Tente falar mais perto/alto.")
                if texto_stt:
                    print(f"[STT {backend_stt} | {lat_stt:.0f} ms]: \"{texto_stt}\"")
                continue

            if entrada.lower() == "/testar_voz":
                print("[MIC] Fale por 4 segundos para testar reconhecimento de voz...")
                audio = stt.gravar_duracao_fixa(4.0)
                vid, vconf, ranking = voz_bio.identificar(audio, sr=stt.sample_rate)
                texto_stt, backend_stt, lat_stt = stt.transcrever(audio, sr=stt.sample_rate)
                print(f"[VOZ] Resultado: {vid} (score={vconf:.3f})")
                if ranking:
                    top = sorted(ranking.items(), key=lambda kv: kv[1], reverse=True)
                    print("[VOZ] Ranking:", ", ".join(f"{k}={v:.3f}" for k, v in top))
                print(f"[STT {backend_stt} | {lat_stt:.0f} ms]: \"{texto_stt}\"")
                continue

            if entrada.lower().startswith("/jev "):
                frase = entrada[5:].strip()
                if not jev_online:
                    jev_online, jev_versao = _verificar_jev(roteador)
                if not jev_online:
                    print(f"[ERRO] Ollama/Jev esta offline. Inicie com: {_comando_ollama_hint()}")
                    continue
                ctx = ContextoRover(
                    transcricao=frase,
                    estado_rover=_estado_para_modo_jev(estado),
                    pessoa_sface=face_simulada,
                    objetos_yolo=estado.objetos_yolo,
                )
                dec = roteador.decidir(ctx)
                print(
                    f"[JEV {dec.latencia_ms:.1f} ms] rota={dec.rota_final}:{dec.subtipo} "
                    f"(conf={dec.confianca:.2f}, dir={dec.dirigido_ao_robo:.2f}, "
                    f"confirmar={dec.requer_confirmacao}, img={dec.precisa_imagem_camera})"
                )
                print(json.dumps(dec.respostas_brutas, ensure_ascii=False, indent=2))
                continue

            imagem_b64: Optional[str] = None
            usou_comando_foto = False
            transcricao = entrada

            if entrada.lower().startswith("/foto"):
                resto = entrada[5:].strip()
                try:
                    partes_cmd = shlex.split(resto, posix=False)
                except ValueError:
                    partes_cmd = resto.split(None, 1)
                if not partes_cmd:
                    print("[AVISO] Uso: /foto caminho_da_foto.jpg [pergunta opcional]")
                    continue
                caminho_foto = Path(partes_cmd[0].strip('"').strip("'")).expanduser()
                if not caminho_foto.exists():
                    print(f"[ERRO] Arquivo nao encontrado: {caminho_foto}")
                    continue
                imagem_b64 = base64.b64encode(caminho_foto.read_bytes()).decode("ascii")
                usou_comando_foto = True
                transcricao = " ".join(partes_cmd[1:]).strip() if len(partes_cmd) > 1 else "Olha aqui o que eu tô te mostrando!"
                print(f"[FOTO] Carregada: {caminho_foto.name} | pergunta: \"{transcricao}\"")

            processar_turno(
                entrada_texto=transcricao,
                imagem_b64=imagem_b64,
                usou_comando_foto=usou_comando_foto,
                face_simulada=face_simulada,
                memoria=memoria,
                voz_bio=voz_bio,
                stt=stt,
                estado=estado,
                motor=motor,
                roteador=roteador,
                tts=tts,
                webcam=webcam,
                camera_idx=args.camera,
                jev_online=jev_online,
                sem_jev=args.sem_jev,
                tts_ativo=tts_ativo,
                limiar_mic=args.limiar_mic,
                roteador_modo=args.roteador,
                escuta_ativa_ctrl=escuta_ativa_ctrl,
                origem_manual_ou_ui=True,
            )
    finally:
        if webcam is not None:
            webcam.fechar()
        if ollama_proc is not None:
            try:
                ollama_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
