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
from pathlib import Path
import shlex
import sys
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.escuta_stt import EscutaSTT
from src.memoria_pessoas import GerenciadorMemoriaPessoas
from src.persona_peteleco import EstadoCorporal, MotorPersonaPETeleco
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
        "  [ENTER vazio]         -> Grava pelo microfone, identifica voz, transcreve (STT), roda Jev e responde\n"
        "  <qualquer frase>      -> Envia texto direto para Jev + Persona + Memoria\n"
        "  /voz <id_ou_apelido>  -> Grava 4s no mic para cadastrar voz (ex: /voz caio, /voz to2, /voz felipe_sass)\n"
        "  /testar_voz           -> Grava 4s no mic e mostra ranking de similaridade cosseno + STT\n"
        "  /pessoa <id|auto>     -> Simula rosto SFace na camera (ou 'auto' para identificar so por voz/fala)\n"
        "  /corpo <chao|bancada|no_colo> -> Muda superficie fisica simulada do robo\n"
        "  /modo <normal|serio>  -> Alterna MODO_PERSONA manualmente\n"
        "  /perfil [id]          -> Mostra ficha quente JSON + ultimos episodios no SQLite\n"
        "  /foto <arq> [frase]   -> Envia imagem JPEG/PNG com reconhecimento facial SFace + Gemini\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bancada de teste PETeleco (Voz + STT + Jev + Persona)")
    parser.add_argument("--sem-jev", action="store_true", help="Pula roteador /v1/systemone e vai direto para a Persona")
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
    print(f"Gemini API Key   : {'OK' if tem_gemini else 'NAO ENCONTRADA (.env)'}")
    if jev_online:
        print(f"Jev /v1/systemone: ONLINE (Ollama {jev_versao} | modelo {roteador.model})")
    else:
        print("Jev /v1/systemone: OFFLINE (para testar Jev junto: $env:OLLAMA_VULKAN='0'; ollama serve)")
    _imprimir_ajuda()

    while True:
        try:
            label_face = face_simulada if face_simulada != "nenhuma" else "auto"
            status_prompt = (
                f"\n[rosto={label_face} | corpo={estado.superficie}/{estado.movimento} "
                f"| modo={motor.modo_persona} | jev={'ON' if jev_online else 'OFF'}]\n> "
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
        imagem_b64: Optional[str] = None
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
            transcricao = " ".join(partes_cmd[1:]).strip() if len(partes_cmd) > 1 else "Olha aqui o que eu tô te mostrando!"
            print(f"[FOTO] Carregada: {caminho_foto.name} | pergunta: \"{transcricao}\"")

        if not transcricao:
            print("[MIC] Ouvindo... fale agora (para sozinho apos 1,2s de silencio).")
            audio = stt.gravar_microfone()
            if audio.size == 0:
                print("[MIC] Nenhuma fala detectada.")
                continue
            voice_id, voice_conf, _ = voz_bio.identificar(audio, sr=stt.sample_rate)
            transcricao, backend_stt, lat_stt = stt.transcrever(audio, sr=stt.sample_rate)
            print(f"[STT {backend_stt} | {lat_stt:.0f} ms]: \"{transcricao}\"")
            if voice_id != "desconhecido":
                print(f"[VOZ] Locutor reconhecido pela voz: {voice_id} ({voice_conf:.2f})")
            if not transcricao:
                print("[AVISO] Nao consegui transcrever a fala.")
                continue

            # Auto-cadastro cruzado: se SFace conhece o rosto sozinho e a voz ainda nao estava boa
            if face_simulada != "nenhuma":
                pid_face = memoria.mapear_identificador(face_simulada.split("(")[0].strip())
                if pid_face and pid_face != "desconhecido" and voice_id == "desconhecido":
                    n = voz_bio.cadastrar_amostra(pid_face, audio, sr=stt.sample_rate)
                    if n > 0:
                        print(f"[AUTO-VOZ] Amostra vocal salva automaticamente para {pid_face} ({n} total).")

        face_turno = face_simulada
        if face_turno == "nenhuma" and imagem_b64:
            face_auto = motor.reconhecer_rosto_imagem_b64(imagem_b64)
            if face_auto != "nenhuma":
                face_turno = face_auto
                print(f"[SFACE] Rosto detectado automaticamente na imagem: {face_auto}")

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

        resultado = motor.responder(
            transcricao=transcricao,
            decisao_jev=decisao_jev,
            face_id=face_turno,
            voice_id=voice_id,
            voice_conf=voice_conf,
            imagem_jpeg_b64=imagem_b64,
        )
        print(
            f"[CONTEXTO] pessoa={resultado['pessoa_display']} | id={resultado['modo_identificacao']} "
            f"| modo={resultado['modo_persona']} | origem={resultado['origem']}"
        )
        if resultado["resposta"]:
            print(f"PETeleco: {resultado['resposta']}")
            if resultado["resposta_tts"] != resultado["resposta"]:
                print(f"   [TTS]: {resultado['resposta_tts']}")
        else:
            print("[IGNORADO] Fala classificada como conversa paralela / ruido.")


if __name__ == "__main__":
    main()
