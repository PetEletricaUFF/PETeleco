#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI Web Local — PETeleco (Webcam Ao Vivo + YuNet/SFace + Telemetria Debaixo dos Panos).

Servidor HTTP + MJPEG + Server-Sent Events (SSE) usando apenas stdlib + modulos locais:
- Stream MJPEG `/video.mjpg` com caixas faciais YuNet+SFace, 5 pontos faciais e score cosseno
- Painel "Debaixo dos Panos" ao vivo: estagio do pipeline, probabilidades do roteador Jev (`/v1/systemone`),
  biometria facial/vocal, trava fisica de bancada, memoria em 3 camadas (JSON + SQLite) e trace em tempo real
- Controles interativos no computador local: Falar 1x (mic), Escuta Continua, TTS Piper, cadastro rapido `/rosto` e `/voz`
"""

import argparse
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional
import webbrowser

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.escuta_stt import EscutaSTT
from src.memoria_pessoas import GerenciadorMemoriaPessoas
from src.persona_peteleco import EstadoCorporal, MotorPersonaPETeleco
from src.router_systemone import ContextoRover, RoteadorSystemOne
from src.testar_peteleco import (
    EstadoEscutaAtiva,
    _estado_para_modo_jev,
    _verificar_jev,
    executar_turno_detalhado,
    garantir_ollama_local,
)
from src.tts_local import DEFAULT_EDGE_VOICE, DEFAULT_PIPER_VOICE, SintetizadorTTS
from src.visao_webcam import WebcamLive
from src.voz_biometria import IdentificadorVoz

HTML_UI_PATH = Path(__file__).resolve().parent / "static" / "ui_peteleco.html"


def _hora_curta() -> str:
    return time.strftime("%H:%M:%S")


class EstadoSessaoUI:
    """Mantem o estado compartilhado da bancada UI, stream de eventos SSE e threads de captura."""

    def __init__(
        self,
        *,
        camera_idx: int = 0,
        sem_webcam: bool = False,
        sem_jev: bool = False,
        tts_ativo: bool = True,
        tts_modo: str = "auto",
        tts_voz: str = DEFAULT_EDGE_VOICE,
        stt_modo: str = "auto",
        roteador_modo: str = "auto",
        exigir_wake_continuo: bool = True,
        whisper_modelo: str = "auto",
        piper_voz: str = DEFAULT_PIPER_VOICE,
        mic_idx: Optional[int] = None,
        limiar_mic: float = 0.012,
        superficie: str = "chao",
        pessoa: str = "auto",
        ollama_url: str = "http://localhost:11434",
        auto_ollama: bool = True,
    ) -> None:
        if whisper_modelo == "auto":
            whisper_modelo = "small"

        self.whisper_modelo = whisper_modelo
        self.whisper_device = "CUDA" if torch.cuda.is_available() else "CPU"
        self.camera_idx = int(camera_idx)
        self.sem_jev = bool(sem_jev)
        self.tts_ativo = bool(tts_ativo)
        self.roteador_modo = roteador_modo if roteador_modo in ("auto", "gemini", "jev") else "auto"
        self.limiar_mic = float(limiar_mic)

        self.memoria = GerenciadorMemoriaPessoas()
        self.voz_bio = IdentificadorVoz()
        self.stt = EscutaSTT(
            whisper_model_size=whisper_modelo,
            aliases_stt=self.memoria.listar_aliases_stt(),
            device_index=mic_idx,
            stt_modo=stt_modo,
        )
        self.estado = EstadoCorporal(superficie=superficie)
        self.motor = MotorPersonaPETeleco(memoria=self.memoria, estado=self.estado, ollama_url=ollama_url)
        self.roteador = RoteadorSystemOne(base_url=ollama_url)
        self.tts = SintetizadorTTS(voice_name=piper_voz, tts_modo=tts_modo, edge_voice=tts_voz)
        self.escuta_ativa_ctrl = EstadoEscutaAtiva(
            janela_s=60.0,
            exigir_wake_continuo=bool(exigir_wake_continuo),
        )

        self.jev_online = False
        self.jev_versao = "desativado"
        self.ollama_proc: Any = None
        if not self.sem_jev:
            if auto_ollama:
                self.jev_online, self.jev_versao, self.ollama_proc = garantir_ollama_local(self.roteador)
            else:
                self.jev_online, self.jev_versao = _verificar_jev(self.roteador)

        self.webcam: Optional[WebcamLive] = None
        if not sem_webcam:
            self.webcam = WebcamLive(
                camera_idx=self.camera_idx,
                detector_faces=self.motor.detectar_e_reconhecer_faces,
                preview=False,
                deteccao_continua=True,
                intervalo_face_s=0.30,
            )
            self.webcam.iniciar()

        self.face_simulada = "nenhuma"
        if pessoa != "auto":
            pid_arg = self.memoria.mapear_identificador(pessoa)
            if pid_arg:
                self.face_simulada = f"{self.memoria.obter_perfil(pid_arg)['nome']} (95%)"

        self.stage = "pronto"
        self.stage_msg = "Aguardando fala, texto ou deteccao na webcam."
        self.ocupado = False
        self.continuo_ativo = False
        self.ultima_pessoa_id = "desconhecido"
        self.last_turn: Optional[Dict[str, Any]] = None
        self.timeline: Deque[Dict[str, Any]] = deque(maxlen=80)
        self.chat_history: Deque[Dict[str, Any]] = deque(maxlen=24)

        self._turn_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._sub_lock = threading.Lock()
        self._subs: set[queue.Queue[Optional[str]]] = set()

        self._stop_all = threading.Event()
        self._stop_continuo = threading.Event()
        self._continuo_thread: Optional[threading.Thread] = None

        self._adicionar_log("INFO", "BOOT", self._resumo_boot())
        self._telemetry_thread = threading.Thread(
            target=self._loop_telemetria, name="peteleco-ui-telemetry", daemon=True
        )
        self._telemetry_thread.start()

    def _resumo_boot(self) -> str:
        cam_str = "webcam ON" if (self.webcam and self.webcam.disponivel) else "webcam OFF"
        jev_str = f"Jev ONLINE ({self.jev_versao})" if self.jev_online else "Jev OFFLINE"
        gem_str = "Gemini OK" if bool(self.motor._env.get("GEMINI_API_KEY", "").strip()) else "Gemini OFF"
        return (
            f"UI iniciada: {cam_str} | {jev_str} | {gem_str} | "
            f"Roteador={self.roteador_modo} | STT={self.stt.descrever_backend()} | TTS={self.tts.descrever_backend()}"
        )

    # ------------------------------------------------------------------
    # SSE Pub/Sub & Timeline
    # ------------------------------------------------------------------
    def subscribe(self) -> queue.Queue[Optional[str]]:
        q: queue.Queue[Optional[str]] = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[Optional[str]]) -> None:
        with self._sub_lock:
            self._subs.discard(q)

    def emit(self, evento: str, payload: Dict[str, Any]) -> None:
        msg = f"event: {evento}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        with self._sub_lock:
            mortos = []
            for q in self._subs:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(msg)
                    except Exception:
                        mortos.append(q)
            for q in mortos:
                self._subs.discard(q)

    def _adicionar_log(self, nivel: str, canal: str, texto: str) -> None:
        entry = {
            "hora": _hora_curta(),
            "nivel": nivel.upper(),
            "canal": canal.upper(),
            "texto": texto,
        }
        with self._state_lock:
            self.timeline.append(entry)
        self.emit("log", entry)

    def _set_stage(self, stage: str, msg: str) -> None:
        with self._state_lock:
            self.stage = stage
            self.stage_msg = msg
        self.emit("stage", {"stage": stage, "mensagem": msg, "ocupado": self.ocupado})

    def _loop_telemetria(self) -> None:
        ultimo_assinatura = ""
        era_escuta_ativa = False
        while not self._stop_all.is_set():
            time.sleep(0.28)
            snap = self._snapshot_visao()
            assinatura = json.dumps(
                {
                    "disp": snap.get("disponivel"),
                    "faces": [
                        (f.get("nome"), round(float(f.get("score", 0.0)), 3), f.get("bbox"))
                        for f in snap.get("faces", [])
                    ],
                    "fps": round(float(snap.get("fps", 0.0)), 0),
                },
                ensure_ascii=False,
            )
            if assinatura != ultimo_assinatura:
                ultimo_assinatura = assinatura
                self.emit("vision", snap)

            agora_ativa = self.escuta_ativa_ctrl.esta_ativa()
            if era_escuta_ativa and not agora_ativa:
                era_escuta_ativa = False
                self._adicionar_log("INFO", "WAKE", "Janela de Escuta Ativa expirou; voltando para Standby.")
                self.emit("state", self.obter_estado_completo())
            elif agora_ativa:
                era_escuta_ativa = True

    def _snapshot_visao(self) -> Dict[str, Any]:
        if self.webcam is None:
            return {
                "disponivel": False,
                "erro": "Webcam desativada (--sem-webcam).",
                "resolucao": [0, 0],
                "fps": 0.0,
                "face_ms": 0.0,
                "frame_seq": 0,
                "deteccao_continua": False,
                "faces": [],
            }
        return self.webcam.obter_telemetria()

    def obter_estado_completo(self) -> Dict[str, Any]:
        """Monta snapshot completo para `/api/state`."""
        with self._state_lock:
            pessoa_id = self.ultima_pessoa_id
            last_turn = self.last_turn
            timeline = list(self.timeline)
            chat = list(self.chat_history)
            stage = self.stage
            stage_msg = self.stage_msg

        perfil = self.memoria.obter_perfil(pessoa_id)
        episodios_frios = (
            self.memoria.buscar_memoria_fria(pessoa_id, limite=5)
            if pessoa_id and pessoa_id != "desconhecido"
            else []
        )

        membros = [
            {"id": pid, "nome": info.get("nome", pid), "papel": info.get("papel", "")}
            for pid, info in sorted(self.memoria.perfis.items())
            if pid != "desconhecido"
        ]

        return {
            "stage": stage,
            "stage_msg": stage_msg,
            "ocupado": self.ocupado,
            "continuo_ativo": self.continuo_ativo,
            "tts_ativo": self.tts_ativo,
            "face_simulada": self.face_simulada,
            "estado_corporal": {
                "superficie": self.estado.superficie,
                "movimento": self.estado.movimento,
                "trava_bancada": self.estado.superficie in ("bancada", "no_colo"),
            },
            "modo_persona": self.motor.modo_persona,
            "escuta_ativa": {
                "ativo": self.escuta_ativa_ctrl.esta_ativa(),
                "restante_s": self.escuta_ativa_ctrl.restante_s(),
                "janela_s": int(self.escuta_ativa_ctrl.janela_s),
                "exigir_wake_continuo": self.escuta_ativa_ctrl.exigir_wake_continuo,
            },
            "servicos": {
                "jev_online": self.jev_online,
                "jev_versao": self.jev_versao,
                "jev_modelo": self.roteador.model,
                "roteador_modo": self.roteador_modo,
                "gemini_ok": bool(self.motor._env.get("GEMINI_API_KEY", "").strip()),
                "stt": self.stt.descrever_backend(),
                "stt_modo": self.stt.stt_modo,
                "whisper_modelo": self.whisper_modelo,
                "voz_bio": self.voz_bio.backend,
                "tts_backend": self.tts.descrever_backend(),
                "tts_modo": self.tts.tts_modo,
                "tts_voz": self.tts.edge_voice,
            },
            "biometria": {
                "rostos_conhecidos": self.motor.listar_rostos_conhecidos(),
                "vozes_cadastradas": sorted(self.voz_bio.vozes.keys()),
                "membros": membros,
            },
            "pessoa_ativa": {
                "id": pessoa_id,
                "perfil": perfil,
                "episodios_sqlite": episodios_frios,
            },
            "webcam": self._snapshot_visao(),
            "last_turn": last_turn,
            "chat_history": chat,
            "timeline": timeline,
        }

    # ------------------------------------------------------------------
    # Execucao de Turnos & Controles
    # ------------------------------------------------------------------
    def _on_turno_event(self, tipo: str, payload: Dict[str, Any]) -> None:
        if tipo == "stage":
            self._set_stage(str(payload.get("stage", "pronto")), str(payload.get("mensagem", "")))
        elif tipo == "stt":
            txt = payload.get("transcricao", "")
            bk = payload.get("backend", "")
            ms = payload.get("latencia_ms", 0)
            vid = payload.get("voice_id", "desconhecido")
            vconf = payload.get("voice_conf", 0.0)
            if bk != "texto":
                self._adicionar_log("INFO", "STT", f"[{bk} {ms:.0f} ms] \"{txt}\" | voz={vid} ({vconf:.2f})")
            else:
                self._adicionar_log("INFO", "INPUT", f"Texto enviado: \"{txt}\"")
        elif tipo == "biometria":
            fid = payload.get("face_id", "nenhuma")
            vid = payload.get("voice_id", "desconhecido")
            fms = payload.get("face_ms", 0.0)
            self._adicionar_log("INFO", "BIOMETRIA", f"Rosto={fid} ({fms:.1f} ms) | Voz={vid}")
            if payload.get("auto_voz_salva"):
                self._adicionar_log("OK", "AUTO-VOZ", f"Amostra vocal salva automaticamente ({payload['auto_voz_salva']}).")
        elif tipo == "jev":
            dec = payload.get("decisao") or {}
            self._adicionar_log(
                "JEV",
                "JEV",
                f"{dec.get('latencia_ms', 0):.0f} ms -> {dec.get('rota_final')}:{dec.get('subtipo')} "
                f"(conf={dec.get('confianca', 0):.2f}, dir={dec.get('dirigido_ao_robo', 0):.2f}, img={dec.get('precisa_imagem_camera')})",
            )
            self.emit("jev", dec)
        elif tipo == "log":
            self._adicionar_log(str(payload.get("nivel", "INFO")), str(payload.get("canal", "SYS")), str(payload.get("texto", "")))

    def executar_turno_ui(
        self,
        *,
        entrada_texto: str = "",
        incluir_webcam_forcada: bool = False,
        silencioso_timeout: bool = False,
        origem_manual_ou_ui: bool = True,
        stop_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        with self._turn_lock:
            self.ocupado = True
            self.emit("busy", {"ocupado": True})
            try:
                imagem_b64 = None
                if incluir_webcam_forcada and self.webcam and self.webcam.disponivel:
                    imagem_b64 = self.webcam.capturar_b64()
                    if imagem_b64:
                        self._adicionar_log("INFO", "CAM", "Frame atual da webcam capturado manualmente para visao.")

                detalhes = executar_turno_detalhado(
                    entrada_texto=entrada_texto,
                    imagem_b64=imagem_b64,
                    usou_comando_foto=bool(incluir_webcam_forcada and imagem_b64),
                    face_simulada=self.face_simulada,
                    memoria=self.memoria,
                    voz_bio=self.voz_bio,
                    stt=self.stt,
                    estado=self.estado,
                    motor=self.motor,
                    roteador=self.roteador,
                    tts=self.tts,
                    webcam=self.webcam,
                    camera_idx=self.camera_idx,
                    jev_online=self.jev_online,
                    sem_jev=self.sem_jev,
                    tts_ativo=self.tts_ativo,
                    limiar_mic=self.limiar_mic,
                    roteador_modo=self.roteador_modo,
                    escuta_ativa_ctrl=self.escuta_ativa_ctrl,
                    origem_manual_ou_ui=origem_manual_ou_ui,
                    silencioso_timeout=silencioso_timeout,
                    stop_event=stop_event,
                    on_event=self._on_turno_event,
                )

                res = detalhes.get("resultado")
                if res:
                    origem = res.get("origem", "")
                    resp_txt = res.get("resposta", "")
                    with self._state_lock:
                        self.ultima_pessoa_id = res.get("pessoa_id", "desconhecido")
                        self.last_turn = detalhes
                        if detalhes.get("transcricao"):
                            self.chat_history.append(
                                {
                                    "hora": _hora_curta(),
                                    "pessoa": res.get("pessoa_display", "desconhecido"),
                                    "usuario": detalhes.get("transcricao", ""),
                                    "robo": resp_txt if resp_txt else "(ignorado — conversa paralela)",
                                    "origem": origem,
                                    "tempos_ms": detalhes.get("tempos_ms", {}),
                                }
                            )
                    if resp_txt:
                        canal = (
                            "WAKE"
                            if origem == "local:wake_word"
                            else ("TRAVA" if "trava" in origem or "superficie" in origem else ("GEMINI" if origem.startswith("gemini:") else "PERSONA"))
                        )
                        self._adicionar_log(
                            "OK",
                            canal,
                            f"[{origem} | {detalhes['tempos_ms']['total_ms']:.0f} ms] {res.get('pessoa_display')}: \"{resp_txt[:110]}\"",
                        )
                    else:
                        self._adicionar_log("AVISO", "ROTA", "Fala descartada como conversa paralela / ruido.")

                return detalhes
            finally:
                self.ocupado = False
                self._set_stage("pronto", "Aguardando fala, texto ou deteccao na webcam.")
                self.emit("busy", {"ocupado": False})
                self.emit("state", self.obter_estado_completo())

    def cadastrar_rosto_atual(self, nome_ou_id: str) -> Dict[str, Any]:
        if not self.webcam or not self.webcam.disponivel:
            return {"ok": False, "mensagem": "Webcam indisponivel para capturar rosto."}
        pid = self.memoria.mapear_identificador(nome_ou_id)
        if not pid:
            pid = self.memoria.garantir_pessoa(nome_ou_id)
        frame_b64 = self.webcam.capturar_b64(max_dim=960, qualidade=92)
        if not frame_b64:
            return {"ok": False, "mensagem": "Nao foi possivel capturar frame da webcam."}
        ok, msg = self.motor.cadastrar_rosto_imagem_b64(pid, frame_b64)
        self.webcam.obter_faces_atuais()
        self._adicionar_log("OK" if ok else "ERRO", "ROSTO", msg)
        self.emit("state", self.obter_estado_completo())
        return {"ok": ok, "mensagem": msg, "pessoa_id": pid}

    def cadastrar_voz_atual(self, nome_ou_id: str) -> Dict[str, Any]:
        with self._turn_lock:
            self.ocupado = True
            self._set_stage("gravando", f"Gravando 4 segundos no microfone para cadastrar '{nome_ou_id}'...")
            self.emit("busy", {"ocupado": True})
            try:
                pid = self.memoria.mapear_identificador(nome_ou_id)
                if not pid:
                    pid = self.memoria.garantir_pessoa(nome_ou_id)
                audio = self.stt.gravar_duracao_fixa(4.0)
                n = self.voz_bio.cadastrar_amostra(pid, audio, sr=self.stt.sample_rate)
                texto_stt, bk_stt, _ = self.stt.transcrever(audio, sr=self.stt.sample_rate)
                if n > 0:
                    msg = f"Voz cadastrada para '{pid}' ({n} amostra(s)). STT ({bk_stt}): \"{texto_stt}\""
                    self._adicionar_log("OK", "VOZ", msg)
                    ok = True
                else:
                    msg = "Audio curto ou baixo demais para extrair embedding vocal."
                    self._adicionar_log("AVISO", "VOZ", msg)
                    ok = False
                return {"ok": ok, "mensagem": msg, "pessoa_id": pid}
            finally:
                self.ocupado = False
                self._set_stage("pronto", "Aguardando fala, texto ou deteccao na webcam.")
                self.emit("busy", {"ocupado": False})
                self.emit("state", self.obter_estado_completo())

    def atualizar_config(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        if "superficie" in dados and dados["superficie"] in ("chao", "bancada", "no_colo"):
            self.estado.superficie = str(dados["superficie"])
            if self.estado.superficie != "chao":
                self.estado.movimento = "parado"
            self._adicionar_log("INFO", "CORPO", f"Superficie ajustada para '{self.estado.superficie}' (trava={self.estado.superficie != 'chao'}).")

        if "modo_persona" in dados and dados["modo_persona"] in ("NORMAL", "SERIO", "DESGRACADO"):
            self.motor.modo_persona = str(dados["modo_persona"])
            self._adicionar_log("INFO", "MODO", f"Modo da persona ajustado para {self.motor.modo_persona}.")

        if "tts_ativo" in dados:
            self.tts_ativo = bool(dados["tts_ativo"])
            self._adicionar_log("INFO", "TTS", f"Saida de voz {'ATIVADA' if self.tts_ativo else 'DESATIVADA'}.")

        if "tts_modo" in dados and str(dados["tts_modo"]) in ("auto", "edge", "gemini", "piper"):
            self.tts.configurar(tts_modo=str(dados["tts_modo"]))
            self._adicionar_log("INFO", "TTS", f"Modo de voz ajustado para {self.tts.descrever_backend()}.")

        if "tts_voz" in dados and str(dados["tts_voz"]).strip():
            self.tts.configurar(edge_voice=str(dados["tts_voz"]).strip())
            self._adicionar_log("INFO", "TTS", f"Voz selecionada: {self.tts.edge_voice}.")

        if "stt_modo" in dados and str(dados["stt_modo"]) in ("auto", "gemini", "whisper"):
            self.stt.configurar(stt_modo=str(dados["stt_modo"]))
            self._adicionar_log("INFO", "STT", f"STT ajustado para {self.stt.descrever_backend()}.")

        if "whisper_modelo" in dados and str(dados["whisper_modelo"]) in ("tiny", "base", "small", "medium"):
            self.whisper_modelo = str(dados["whisper_modelo"])
            self.stt.configurar(whisper_model_size=self.whisper_modelo)
            self._adicionar_log("INFO", "STT", f"Modelo Whisper atualizado para '{self.whisper_modelo}'.")

        if "roteador_modo" in dados and str(dados["roteador_modo"]) in ("auto", "gemini", "jev"):
            self.roteador_modo = str(dados["roteador_modo"])
            self._adicionar_log("INFO", "ROTA", f"Modo de roteamento alterado para '{self.roteador_modo}'.")

        if "exigir_wake_continuo" in dados:
            self.escuta_ativa_ctrl.exigir_wake_continuo = bool(dados["exigir_wake_continuo"])
            self._adicionar_log(
                "INFO",
                "WAKE",
                f"Wake-word 'PETeleco' no modo continuo: {'OBRIGATORIO' if self.escuta_ativa_ctrl.exigir_wake_continuo else 'LIVRE'}",
            )

        if "escuta_ativa" in dados:
            if bool(dados["escuta_ativa"]):
                self.escuta_ativa_ctrl.ativar()
                self._adicionar_log("OK", "WAKE", f"Escuta Ativa ligada manualmente por {int(self.escuta_ativa_ctrl.janela_s)}s.")
            else:
                self.escuta_ativa_ctrl.encerrar()
                self._adicionar_log("INFO", "WAKE", "Escuta Ativa encerrada manualmente; em Standby.")

        if "deteccao_continua" in dados and self.webcam is not None:
            ativo = bool(dados["deteccao_continua"])
            self.webcam.ativar_deteccao_continua(ativo)
            self._adicionar_log("INFO", "VISAO", f"Deteccao facial continua {'ON' if ativo else 'OFF'}.")

        if "pessoa" in dados:
            val = str(dados["pessoa"]).strip()
            if val in ("auto", "nenhuma", ""):
                self.face_simulada = "nenhuma"
            else:
                pid = self.memoria.mapear_identificador(val)
                if pid:
                    self.face_simulada = f"{self.memoria.obter_perfil(pid)['nome']} (95%)"
                    with self._state_lock:
                        self.ultima_pessoa_id = pid
            self._adicionar_log("INFO", "ID", f"Identificacao simulada: {self.face_simulada or 'auto'}.")

        self.emit("state", self.obter_estado_completo())
        return self.obter_estado_completo()

    def alternar_continuo(self, ativar: Optional[bool] = None) -> Dict[str, Any]:
        desejado = (not self.continuo_ativo) if ativar is None else bool(ativar)
        if desejado == self.continuo_ativo:
            return {"continuo_ativo": self.continuo_ativo}

        if desejado:
            self._stop_continuo.clear()
            self.continuo_ativo = True
            self._continuo_thread = threading.Thread(
                target=self._loop_continuo_worker, name="peteleco-ui-continuo", daemon=True
            )
            self._continuo_thread.start()
            self._adicionar_log("OK", "CONTINUO", "Escuta continua maos-livres ATIVADA.")
        else:
            self.continuo_ativo = False
            self._stop_continuo.set()
            self._adicionar_log("INFO", "CONTINUO", "Escuta continua pausada.")

        self.emit("state", self.obter_estado_completo())
        return {"continuo_ativo": self.continuo_ativo}

    def _loop_continuo_worker(self) -> None:
        while self.continuo_ativo and not self._stop_continuo.is_set() and not self._stop_all.is_set():
            det = self.executar_turno_ui(
                entrada_texto="",
                silencioso_timeout=True,
                origem_manual_ou_ui=False,
                stop_event=self._stop_continuo,
            )
            if det.get("respondeu") and self.tts_ativo:
                time.sleep(0.30)
            else:
                time.sleep(0.08)

    def iniciar_ollama_sob_demanda(self) -> Dict[str, Any]:
        online, versao, proc = garantir_ollama_local(self.roteador)
        self.jev_online = online
        self.jev_versao = versao
        if proc is not None:
            self.ollama_proc = proc
        self._adicionar_log("OK" if online else "ERRO", "JEV", f"Ollama/Jev agora: {'ONLINE (' + versao + ')' if online else 'OFFLINE'}")
        self.emit("state", self.obter_estado_completo())
        return {"online": online, "versao": versao}

    def encerrar(self) -> None:
        self._stop_all.set()
        self.continuo_ativo = False
        self._stop_continuo.set()
        with self._sub_lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
        if self._continuo_thread is not None and self._continuo_thread.is_alive():
            self._continuo_thread.join(timeout=2.0)
        if self.webcam is not None:
            self.webcam.fechar()
        if self.ollama_proc is not None:
            try:
                self.ollama_proc.terminate()
            except Exception:
                pass


def criar_handler(sessao: EstadoSessaoUI):
    """Fabrica o RequestHandler HTTP com acesso a sessao compartilhada."""

    class PETelecoUIHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            # Mantem o console limpo para os logs estruturados do PETeleco
            return

        def _enviar_json(self, dados: Any, status: int = 200) -> None:
            body = json.dumps(dados, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _ler_json(self) -> Dict[str, Any]:
            tam = int(self.headers.get("Content-Length", "0") or 0)
            if tam <= 0:
                return {}
            bruto = self.rfile.read(tam)
            if not bruto:
                return {}
            try:
                parsed = json.loads(bruto.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

        def do_GET(self) -> None:
            caminho = self.path.split("?", 1)[0]
            if caminho in ("/", "/index.html"):
                if not HTML_UI_PATH.exists():
                    self._enviar_json({"erro": f"Arquivo HTML ausente: {HTML_UI_PATH}"}, status=404)
                    return
                html = HTML_UI_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)
                return

            if caminho == "/api/state":
                self._enviar_json(sessao.obter_estado_completo())
                return

            if caminho == "/api/events":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                sub = sessao.subscribe()
                try:
                    inicial = json.dumps(sessao.obter_estado_completo(), ensure_ascii=False)
                    self.wfile.write(f"event: state\ndata: {inicial}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    while not sessao._stop_all.is_set():
                        try:
                            msg = sub.get(timeout=10.0)
                            if msg is None:
                                break
                            self.wfile.write(msg.encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    sessao.unsubscribe(sub)
                return

            if caminho == "/video.mjpg":
                if sessao.webcam is None or not sessao.webcam.disponivel:
                    self._enviar_json({"erro": "Webcam indisponivel."}, status=503)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                ultimo_seq = -1
                try:
                    while not sessao._stop_all.is_set():
                        novo_seq = sessao.webcam.aguardar_novo_frame(ultimo_seq, timeout_s=0.5)
                        if novo_seq == ultimo_seq:
                            continue
                        ultimo_seq = novo_seq
                        jpg = sessao.webcam.obter_frame_mjpeg_bytes(qualidade=80, anotar_faces=True)
                        if not jpg:
                            continue
                        cabecalho = (
                            f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n"
                        ).encode("ascii")
                        self.wfile.write(cabecalho)
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return

            self._enviar_json({"erro": "Rota nao encontrada."}, status=404)

        def do_POST(self) -> None:
            caminho = self.path.split("?", 1)[0]
            dados = self._ler_json()

            if caminho == "/api/config":
                self._enviar_json({"ok": True, "state": sessao.atualizar_config(dados)})
                return

            if caminho == "/api/continuo":
                res = sessao.alternar_continuo(dados.get("ativo"))
                self._enviar_json({"ok": True, **res})
                return

            if caminho == "/api/ollama_start":
                res = sessao.iniciar_ollama_sob_demanda()
                self._enviar_json({"ok": bool(res.get("online")), **res})
                return

            if caminho in ("/api/chat", "/api/mic", "/api/enroll_face", "/api/enroll_voice"):
                if sessao.ocupado:
                    self._enviar_json(
                        {"ok": False, "mensagem": "PETeleco esta processando outro turno agora. Aguarde um instante."},
                        status=409,
                    )
                    return

            if caminho == "/api/chat":
                texto = str(dados.get("texto", "")).strip()
                visao_forcada = bool(dados.get("incluir_webcam", False))
                if not texto and not visao_forcada:
                    self._enviar_json({"ok": False, "mensagem": "Texto vazio."}, status=400)
                    return
                if not texto and visao_forcada:
                    texto = "PETeleco, olha pela camera e descreve o que voce ta vendo agora."
                threading.Thread(
                    target=lambda: sessao.executar_turno_ui(
                        entrada_texto=texto, incluir_webcam_forcada=visao_forcada
                    ),
                    daemon=True,
                ).start()
                self._enviar_json({"ok": True})
                return

            if caminho == "/api/mic":
                threading.Thread(
                    target=lambda: sessao.executar_turno_ui(entrada_texto="", silencioso_timeout=False),
                    daemon=True,
                ).start()
                self._enviar_json({"ok": True})
                return

            if caminho == "/api/enroll_face":
                nome = str(dados.get("pessoa", "")).strip()
                if not nome:
                    self._enviar_json({"ok": False, "mensagem": "Informe o nome/id da pessoa."}, status=400)
                    return
                self._enviar_json(sessao.cadastrar_rosto_atual(nome))
                return

            if caminho == "/api/enroll_voice":
                nome = str(dados.get("pessoa", "")).strip()
                if not nome:
                    self._enviar_json({"ok": False, "mensagem": "Informe o nome/id da pessoa."}, status=400)
                    return
                threading.Thread(
                    target=lambda: sessao.cadastrar_voz_atual(nome),
                    daemon=True,
                ).start()
                self._enviar_json({"ok": True, "mensagem": "Gravando 4 segundos no microfone..."})
                return

            self._enviar_json({"erro": "Rota POST nao encontrada."}, status=404)

    return PETelecoUIHandler


def _iniciar_servidor_http(sessao: EstadoSessaoUI, porta_inicial: int) -> tuple[ThreadingHTTPServer, int]:
    handler_cls = criar_handler(sessao)
    for porta in range(int(porta_inicial), int(porta_inicial) + 15):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", porta), handler_cls)
            srv.daemon_threads = True
            return srv, porta
        except OSError:
            continue
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    return srv, int(srv.server_address[1])


def executar_ui(args: argparse.Namespace) -> None:
    """Inicia a UI Web local reutilizando os argumentos da bancada."""
    tts_inicial = True if getattr(args, "ui_tts_padrao", False) else bool(getattr(args, "tts", False))
    if getattr(args, "sem_tts", False):
        tts_inicial = False

    sessao = EstadoSessaoUI(
        camera_idx=getattr(args, "camera", 0),
        sem_webcam=getattr(args, "sem_webcam", False),
        sem_jev=getattr(args, "sem_jev", False),
        tts_ativo=tts_inicial,
        tts_modo=getattr(args, "tts_modo", "auto"),
        tts_voz=getattr(args, "tts_voz", DEFAULT_EDGE_VOICE),
        stt_modo=getattr(args, "stt_modo", "auto"),
        roteador_modo=getattr(args, "roteador", "auto"),
        exigir_wake_continuo=not bool(getattr(args, "sem_wake_word", False)),
        whisper_modelo=getattr(args, "whisper_modelo", "auto"),
        piper_voz=getattr(args, "piper_voz", DEFAULT_PIPER_VOICE),
        mic_idx=getattr(args, "mic", None),
        limiar_mic=getattr(args, "limiar_mic", 0.012),
        superficie=getattr(args, "superficie", "chao"),
        pessoa=getattr(args, "pessoa", "auto"),
        ollama_url=getattr(args, "ollama_url", "http://localhost:11434"),
        auto_ollama=getattr(args, "auto_ollama", True),
    )

    servidor, porta_real = _iniciar_servidor_http(sessao, getattr(args, "porta", 8765))
    url = f"http://127.0.0.1:{porta_real}"
    print("=" * 72)
    print(f"UI WEB PETELECO ONLINE: {url}")
    print("=" * 72)
    print(f"Webcam : {'ATIVA (YuNet+SFace)' if (sessao.webcam and sessao.webcam.disponivel) else 'OFF'}")
    print(f"Roteador: {sessao.roteador_modo} | Jev={'ONLINE (' + sessao.jev_versao + ')' if sessao.jev_online else 'OFFLINE'}")
    print(f"STT    : {sessao.stt.descrever_backend()}")
    print(f"TTS    : {'ON' if sessao.tts_ativo else 'OFF'} ({sessao.tts.descrever_backend()})")
    print("Pressione Ctrl+C neste terminal para encerrar.")

    if not getattr(args, "sem_browser", False):
        threading.Timer(0.45, lambda: webbrowser.open(url)).start()

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando UI Web do PETeleco...")
    finally:
        servidor.shutdown()
        servidor.server_close()
        sessao.encerrar()


def main() -> None:
    parser = argparse.ArgumentParser(description="UI Web PETeleco (Webcam + Reconhecimento Facial + Debaixo dos Panos)")
    parser.add_argument("--porta", type=int, default=8765, help="Porta HTTP local (padrao: 8765)")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam local (padrao: 0)")
    parser.add_argument("--sem-webcam", action="store_true", help="Inicia sem webcam")
    parser.add_argument("--sem-jev", action="store_true", help="Desativa roteador Jev")
    parser.add_argument("--sem-tts", action="store_true", help="Inicia com TTS mutado")
    parser.add_argument("--sem-browser", action="store_true", help="Nao abre navegador automaticamente")
    parser.add_argument("--sem-auto-ollama", action="store_true", help="Nao tenta iniciar `ollama serve` automaticamente")
    parser.add_argument("--sem-wake-word", action="store_true", help="No modo continuo, nao exige dizer 'PETeleco' antes")
    parser.add_argument("--tts-modo", choices=["auto", "edge", "gemini", "piper"], default="auto", help="Engine TTS (padrao: auto -> Edge Neural)")
    parser.add_argument("--tts-voz", default=DEFAULT_EDGE_VOICE, help=f"Voz Edge PT-BR (padrao: {DEFAULT_EDGE_VOICE})")
    parser.add_argument("--piper-voz", default=DEFAULT_PIPER_VOICE, help="Voz Piper PT-BR offline")
    parser.add_argument("--stt-modo", choices=["auto", "gemini", "whisper"], default="auto", help="Engine STT (padrao: auto)")
    parser.add_argument("--whisper-modelo", default="auto", help="Modelo faster-whisper (auto|tiny|base|small|medium)")
    parser.add_argument("--roteador", choices=["auto", "gemini", "jev"], default="auto", help="Roteador de intencoes (auto|gemini|jev)")
    parser.add_argument("--mic", type=int, default=None, help="Indice do microfone")
    parser.add_argument("--limiar-mic", type=float, default=0.012, help="Limiar RMS do VAD do microfone")
    parser.add_argument("--superficie", choices=["chao", "bancada", "no_colo"], default="chao")
    parser.add_argument("--pessoa", default="auto", help="Pessoa simulada inicial (padrao: auto)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    args.ui_tts_padrao = True
    args.auto_ollama = not args.sem_auto_ollama
    executar_ui(args)


if __name__ == "__main__":
    main()
