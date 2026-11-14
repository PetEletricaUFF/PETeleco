#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bancada Interativa de Teste — PETeleco (Voz + STT + Jev SystemOne + Persona + Memoria).

Permite testar ponta a ponta no PC:
- Captura de microfone e transcricao STT (`faster-whisper` ou Gemini Audio REST)
- Biometria de voz (`IdentificadorVoz`) mesmo sem camera/rosto
- Roteador Jev `/v1/systemone` (`tev1:0.8b` no Ollama local)
- Persona v3.1 adaptativa por pessoa e consciencia corporal (`chao` / `bancada` / `no_colo`)
- Memoria evolutiva em 3 camadas (`data/perfis_pessoas.json` + `SQLite`)
"""

import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.escuta_stt import EscutaSTT
from src.memoria_pessoas import GerenciadorMemoriaPessoas
from src.persona_peteleco import EstadoCorporal, MotorPersonaPETeleco, parece_pedido_visual
from src.router_systemone import ContextoRover, DecisaoRoteador, RoteadorSystemOne
from src.voz_biometria import IdentificadorVoz


def _estado_para_modo_jev(estado: EstadoCorporal) -> str:
    mapa = {
        "parado": "STANDBY",
        "andando": "APROXIMA",
        "seguindo_rosto": "CENTRALIZA",
        "radar": "RADAR",
    }
    return mapa.get(estado.movimento, "STANDBY")


def _imprimir_ajuda() -> None:
    print(
        "\nComandos da bancada:\n"
        "  [ENTER vazio]         -> Grava pelo microfone, identifica voz/rosto, STT, roda Jev e responde\n"
        "  <qualquer frase>      -> Envia texto direto para Jev + Persona + Memoria (+ webcam se pedir visao)\n"
        "  /voz <id_ou_apelido>  -> Grava 4s no mic para cadastrar voz (ex: /voz caio, /voz to2, /voz felipe_sass)\n"
        "  /testar_voz           -> Grava 4s no mic e mostra ranking de similaridade cosseno + STT\n"
        "  /cam                  -> Tira 1 foto da webcam agora e testa reconhecimento facial YuNet+SFace\n"
        "  /foto <arq> [frase]   -> Envia imagem JPEG/PNG com reconhecimento facial SFace + Gemini\n"
        "  /tts                  -> Liga/desliga leitura da resposta em voz alta no alto-falante (Windows)\n"
        "  /pessoa <id|auto>     -> Simula rosto SFace na camera (ou 'auto' para webcam/voz real)\n"
        "  /corpo <chao|bancada|no_colo> -> Muda superficie fisica simulada do robo\n"
        "  /modo <normal|serio>  -> Alterna MODO_PERSONA manualmente\n"
        "  /perfil [id]          -> Mostra ficha quente JSON + ultimos episodios no SQLite\n"
        "  /jev <frase>          -> Inspeciona todas as probabilidades do /v1/systemone para a frase\n"
        "  /mics                 -> Lista dispositivos de microfone disponiveis\n"
        "  /ajuda | /sair        -> Mostra comandos / encerra\n"
    )


def _capturar_webcam_b64(camera_idx: int = 0) -> Optional[str]:
    """Captura 1 frame JPEG base64 da webcam do PC (redimensionado ate 640px)."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    cap = None
    try:
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            return None

        frame = None
        for _ in range(3):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            return None

        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 640:
            scale = 640.0 / max_dim
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok_enc:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()


def _falar_tts_windows(texto_tts: str) -> None:
    """Sintetiza a resposta em voz alta usando SAPI nativo do Windows (zero VRAM)."""
    texto = (texto_tts or "").strip()
    if not texto or not sys.platform.startswith("win"):
        return
    env = os.environ.copy()
    env["PETELECO_TTS_TEXT"] = texto
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'pt*' } | Select-Object -First 1; "
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
        "$s.Rate = 1; "
        "$s.Speak($env:PETELECO_TTS_TEXT)"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=25,
        )
    except Exception:
        pass


def _verificar_jev(roteador: RoteadorSystemOne) -> tuple[bool, str]:
    try:
        versao = roteador.verificar_servidor()
        return True, versao
    except Exception:
        return False, "offline"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bancada de teste PETeleco (Voz + STT + Jev + Persona)")
    parser.add_argument("--sem-jev", action="store_true", help="Pula roteador /v1/systemone e vai direto para a Persona")
    parser.add_argument("--sem-webcam", action="store_true", help="Desativa captura automatica de webcam na bancada")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam local (padrao: 0)")
    parser.add_argument("--tts", action="store_true", help="Ativa leitura das respostas em voz alta (Windows SAPI)")
    parser.add_argument("--superficie", choices=["chao", "bancada", "no_colo"], default="chao")
    parser.add_argument("--pessoa", default="auto", help="Pessoa simulada no SFace inicial (padrao: auto)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    memoria = GerenciadorMemoriaPessoas()
    voz_bio = IdentificadorVoz()
    stt = EscutaSTT(aliases_stt=memoria.listar_aliases_stt())
    estado = EstadoCorporal(superficie=args.superficie)
    motor = MotorPersonaPETeleco(memoria=memoria, estado=estado, ollama_url=args.ollama_url)
    roteador = RoteadorSystemOne(base_url=args.ollama_url)

    jev_online = False
    jev_versao = "desativado"
    if not args.sem_jev:
        jev_online, jev_versao = _verificar_jev(roteador)

    webcam_ativa = False if args.sem_webcam else (_capturar_webcam_b64(args.camera) is not None)
    tts_ativo = bool(args.tts)

    face_simulada = "nenhuma"
    if args.pessoa != "auto":
        pid_arg = memoria.mapear_identificador(args.pessoa)
        if pid_arg:
            face_simulada = f"{memoria.obter_perfil(pid_arg)['nome']} (95%)"

    tem_gemini = bool(motor._env.get("GEMINI_API_KEY", "").strip())

    print("=" * 72)
    print("BANCADA PETELECO — Voice Recognition + STT + Jev SystemOne + Persona v3.1")
    print("=" * 72)
    print(f"Biometria de Voz : {voz_bio.backend} | vozes cadastradas: {list(voz_bio.vozes.keys()) or 'nenhuma'}")
    print(f"Perfis carregados: {len(memoria.perfis)} entradas ({memoria.perfis_path.name})")
    print(f"Webcam local     : {'ATIVA (YuNet+SFace)' if webcam_ativa else 'OFF (use /cam ou --camera N)'}")
    print(f"Saida TTS Windows: {'ON' if tts_ativo else 'OFF (use /tts para ativar)'}")
    print(f"Gemini API Key   : {'OK' if tem_gemini else 'NAO ENCONTRADA (.env)'}")
    if jev_online:
        print(f"Jev /v1/systemone: ONLINE (Ollama {jev_versao} | modelo {roteador.model})")
    else:
        print("Jev /v1/systemone: OFFLINE (para testar Jev junto: $env:OLLAMA_VULKAN='0'; ollama serve)")
    _imprimir_ajuda()

    while True:
        try:
            label_face = face_simulada if face_simulada != "nenhuma" else ("cam+voz" if webcam_ativa else "auto")
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
            print(f"[INFO] TTS em voz alta: {'ON' if tts_ativo else 'OFF'}")
            continue

        if entrada.lower() == "/cam":
            frame_cam = _capturar_webcam_b64(args.camera)
            if not frame_cam:
                print(f"[ERRO] Nao consegui capturar frame da webcam {args.camera}.")
            else:
                webcam_ativa = True
                face_cam = motor.reconhecer_rosto_imagem_b64(frame_cam)
                print(f"[CAM] Frame capturado! Reconhecimento YuNet+SFace: {face_cam}")
            continue

        if entrada.lower().startswith("/pessoa"):
            arg = entrada[7:].strip()
            if not arg or arg.lower() == "auto":
                face_simulada = "nenhuma"
                print("[INFO] Rosto simulado resetado para 'auto' (identifica por voz/fala).")
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
            if m in ("NORMAL", "SERIO"):
                motor.modo_persona = m
                print(f"[INFO] MODO_PERSONA = {motor.modo_persona}")
            else:
                print("[AVISO] Use: /modo normal | /modo serio")
            continue

        if entrada.lower().startswith("/perfil"):
            arg = entrada[7:].strip()
            pid = memoria.mapear_identificador(arg) if arg else (
                memoria.mapear_identificador(face_simulada.split("(")[0].strip())
                if face_simulada != "nenhuma"
                else "caio"
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
                print("[ERRO] Ollama/Jev esta offline. Inicie com: $env:OLLAMA_VULKAN='0'; ollama serve")
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

        # Fluxo principal: microfone (ENTER vazio), /foto ou texto digitado
        voice_id = "desconhecido"
        voice_conf = 0.0
        audio_mic: Any = None
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

        if not transcricao:
            print("[MIC] Ouvindo... fale agora (para sozinho apos 1,2s de silencio).")
            audio_mic = stt.gravar_microfone()
            if audio_mic.size == 0:
                print("[MIC] Nenhuma fala detectada.")
                continue
            voice_id, voice_conf, _ = voz_bio.identificar(audio_mic, sr=stt.sample_rate)
            transcricao, backend_stt, lat_stt = stt.transcrever(audio_mic, sr=stt.sample_rate)
            print(f"[STT {backend_stt} | {lat_stt:.0f} ms]: \"{transcricao}\"")
            if voice_id != "desconhecido":
                print(f"[VOZ] Locutor reconhecido pela voz: {voice_id} ({voice_conf:.2f})")
            if not transcricao:
                print("[AVISO] Nao consegui transcrever a fala.")
                continue

        frame_cam_turno: Optional[str] = None
        face_turno = face_simulada
        if face_turno == "nenhuma" and imagem_b64:
            face_auto = motor.reconhecer_rosto_imagem_b64(imagem_b64)
            if face_auto != "nenhuma":
                face_turno = face_auto
                print(f"[SFACE] Rosto detectado automaticamente na foto: {face_auto}")
        elif face_turno == "nenhuma" and webcam_ativa:
            frame_cam_turno = _capturar_webcam_b64(args.camera)
            if frame_cam_turno:
                face_auto = motor.reconhecer_rosto_imagem_b64(frame_cam_turno)
                if face_auto != "nenhuma":
                    face_turno = face_auto
                    print(f"[SFACE CAM] Rosto detectado ao vivo: {face_auto}")

        # Auto-cadastro cruzado: se SFace conhece o rosto sozinho e a voz ainda nao estava boa
        if audio_mic is not None and face_turno != "nenhuma" and voice_id == "desconhecido":
            pid_face = memoria.mapear_identificador(face_turno.split("(")[0].strip())
            if pid_face and pid_face != "desconhecido":
                n = voz_bio.cadastrar_amostra(pid_face, audio_mic, sr=stt.sample_rate)
                if n > 0:
                    print(f"[AUTO-VOZ] Amostra vocal salva automaticamente para {pid_face} ({n} total).")

        decisao_jev: Optional[DecisaoRoteador] = None
        if jev_online and not args.sem_jev:
            try:
                pessoa_ctx = face_turno
                if pessoa_ctx == "nenhuma" and voice_id != "desconhecido":
                    pessoa_ctx = f"{memoria.obter_perfil(voice_id)['nome']} ({int(round(voice_conf * 100))}% voz)"
                ctx = ContextoRover(
                    transcricao=transcricao,
                    estado_rover=_estado_para_modo_jev(estado),
                    pessoa_sface=pessoa_ctx,
                    objetos_yolo=estado.objetos_yolo,
                )
                decisao_jev = roteador.decidir(ctx)
                print(
                    f"[JEV {decisao_jev.latencia_ms:.1f} ms] rota={decisao_jev.rota_final}:{decisao_jev.subtipo} "
                    f"(conf={decisao_jev.confianca:.2f}, dir={decisao_jev.dirigido_ao_robo:.2f}, "
                    f"confirmar={decisao_jev.requer_confirmacao}, img={decisao_jev.precisa_imagem_camera})"
                )
            except Exception as exc:
                print(f"[AVISO JEV] Falha ao consultar SystemOne ({exc}); seguindo direto para Persona.")
                decisao_jev = None

        precisa_imagem = (
            usou_comando_foto
            or bool(decisao_jev and decisao_jev.precisa_imagem_camera)
            or parece_pedido_visual(transcricao)
        )
        imagem_para_llm = imagem_b64
        if precisa_imagem and not imagem_para_llm:
            if frame_cam_turno is None and webcam_ativa:
                frame_cam_turno = _capturar_webcam_b64(args.camera)
            imagem_para_llm = frame_cam_turno
            if imagem_para_llm:
                print("[CAM] Frame da webcam anexado ao turno visual.")

        decisao_para_motor = decisao_jev
        if (
            precisa_imagem
            and imagem_para_llm
            and decisao_jev is not None
            and decisao_jev.rota_final in ("ignorar", "pool_fala_pronta")
        ):
            decisao_para_motor = None

        resultado = motor.responder(
            transcricao=transcricao,
            decisao_jev=decisao_para_motor,
            face_id=face_turno,
            voice_id=voice_id,
            voice_conf=voice_conf,
            imagem_jpeg_b64=imagem_para_llm,
        )
        print(
            f"[CONTEXTO] pessoa={resultado['pessoa_display']} | id={resultado['modo_identificacao']} "
            f"| modo={resultado['modo_persona']} | origem={resultado['origem']}"
        )
        if resultado["resposta"]:
            print(f"PETeleco: {resultado['resposta']}")
            if resultado["resposta_tts"] != resultado["resposta"]:
                print(f"   [TTS]: {resultado['resposta_tts']}")
            if tts_ativo:
                _falar_tts_windows(resultado["resposta_tts"])
        else:
            print("[IGNORADO] Fala classificada como conversa paralela / ruido.")


if __name__ == "__main__":
    main()
