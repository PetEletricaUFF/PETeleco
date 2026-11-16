#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arena Multi-Chat de System Prompts & Avaliação (0–5) — PETeleco.

Permite rodar vários chats paralelos com System Prompts muito distintos entre si,
comparar respostas lado a lado para a mesma fala ou cenário completo, acompanhar
métricas objetivas anti-clichê em tempo real, dar nota de 0 a 5 para cada estilo
e exportar feedbacks em Markdown estruturado ou promover o vencedor ao `.md`.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import webbrowser

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.persona_peteleco import DEFAULT_PROMPT_PATH, carregar_system_prompt_base
from src.prompt_metrics import (
    CENARIOS_BENCHMARK_ARENA,
    VARIANTES_ARENA,
    avaliar_conversa_metricas,
)
from src.studio_prompt import (
    StudioPromptApp,
    abrir_servidor_porta_livre,
    enviar_json_http,
    ler_json_http,
    salvar_system_prompt_base,
    servir_html_arquivo,
)


HTML_ARENA_PATH = Path(__file__).resolve().parent / "static" / "arena_prompts.html"


def extrair_pares_turnos(
    messages: List[Dict[str, Any]],
    resposta_extra: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Converte histórico `[user, assistant, ...]` em pares `{user, assistant}` para métricas."""
    pares: List[Dict[str, str]] = []
    ultimo_user = ""
    for item in messages:
        role = str(item.get("role") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            if ultimo_user:
                pares.append({"user": ultimo_user, "assistant": ""})
            ultimo_user = text
        elif role in ("assistant", "model"):
            pares.append({"user": ultimo_user, "assistant": text})
            ultimo_user = ""
    if resposta_extra and resposta_extra.strip():
        pares.append({"user": ultimo_user, "assistant": resposta_extra.strip()})
    return [p for p in pares if p.get("assistant")]


class ArenaPromptsApp:
    """Controlador HTTP da Arena multi-chat (sem gravação em memória/SQLite)."""

    def __init__(
        self,
        *,
        prompt_path: Optional[Path] = None,
        perfis_path: Optional[Path] = None,
        ollama_url: str = "http://localhost:11434",
        pacing_seconds: float = 4.2,
    ) -> None:
        self.studio = StudioPromptApp(
            prompt_path=prompt_path,
            perfis_path=perfis_path,
            ollama_url=ollama_url,
        )
        self.pacing_seconds = max(0.0, float(pacing_seconds))

    def obter_estado_inicial(self) -> Dict[str, Any]:
        estado = self.studio.obter_estado_inicial()
        variantes = [
            {
                "id": "atual_md",
                "nome": "0 — Atual (persona_peteleco.md)",
                "resumo": "O System Prompt atualmente salvo em prompts/persona_peteleco.md.",
                "prompt": estado.get("prompt_base") or "",
            }
        ] + [dict(v) for v in VARIANTES_ARENA]

        return {
            **estado,
            "variantes": variantes,
            "cenarios": CENARIOS_BENCHMARK_ARENA,
            "pacing_gemini_ms": int(round(self.pacing_seconds * 1000)),
        }

    def executar_chat_unico(self, dados: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        status, payload = self.studio.executar_chat(dados)
        if status != 200 or not payload.get("ok"):
            return status, payload

        contexto_cfg = dados.get("contexto") if isinstance(dados.get("contexto"), dict) else {}
        pessoa_id = str(contexto_cfg.get("pessoa_id") or "caio").strip()
        raw_messages = dados.get("messages") if isinstance(dados.get("messages"), list) else []
        pares = extrair_pares_turnos(raw_messages, resposta_extra=payload.get("reply"))
        payload["metricas"] = avaliar_conversa_metricas(
            pares,
            modo_efetivo=str(payload.get("modo_efetivo") or "NORMAL"),
            pessoa_id=pessoa_id,
        )
        return 200, payload

    def executar_chat_batch(self, dados: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Executa a mesma fala em múltiplas variantes respeitando ritmo para evitar 429."""
        itens = dados.get("variantes")
        if not isinstance(itens, list) or not itens:
            return 400, {"ok": False, "error": "Envie uma lista nao vazia em 'variantes'."}

        provider = str(dados.get("provider") or "gemini").strip().lower()
        usar_pacing = provider == "gemini" and self.pacing_seconds > 0.0
        resultados: List[Dict[str, Any]] = []

        for idx, item in enumerate(itens):
            if not isinstance(item, dict):
                continue
            if idx > 0 and usar_pacing:
                time.sleep(self.pacing_seconds)

            req_item = {
                "prompt_base": item.get("prompt_base") or "",
                "incluir_contexto": dados.get("incluir_contexto", True),
                "contexto": dados.get("contexto") or {},
                "provider": provider,
                "model": dados.get("model") or "",
                "temperature": dados.get("temperature", 0.9),
                "max_tokens": dados.get("max_tokens", 240),
                "messages": item.get("messages") or [],
            }
            status, res = self.executar_chat_unico(req_item)
            resultados.append(
                {
                    "variant_id": str(item.get("variant_id") or f"v{idx + 1}"),
                    "status": status,
                    **res,
                }
            )

        return 200, {"ok": True, "resultados": resultados}

    def pontuar_metricas(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        contexto_cfg = dados.get("contexto") if isinstance(dados.get("contexto"), dict) else {}
        pessoa_id = str(contexto_cfg.get("pessoa_id") or "caio").strip()
        modo_efetivo = str(dados.get("modo_efetivo") or contexto_cfg.get("modo_persona") or "NORMAL").strip().upper()
        raw_messages = dados.get("messages") if isinstance(dados.get("messages"), list) else []
        pares = extrair_pares_turnos(raw_messages)
        return {
            "ok": True,
            "metricas": avaliar_conversa_metricas(
                pares,
                modo_efetivo=modo_efetivo,
                pessoa_id=pessoa_id,
            ),
        }

    def promover_prompt(self, dados: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        novo = str(dados.get("prompt_base") or "")
        try:
            salvo = salvar_system_prompt_base(novo, self.studio.prompt_path)
        except Exception as exc:
            return 400, {"ok": False, "error": str(exc)}
        return 200, {
            "ok": True,
            "prompt_base": salvo,
            "message": f"Promovido e salvo em {self.studio.prompt_path.name}.",
        }


def criar_handler_arena(app: ArenaPromptsApp):
    """Fabrica o RequestHandler HTTP da Arena Multi-Chat."""

    class ArenaPromptsHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            caminho = self.path.split("?", 1)[0]
            if caminho in ("/", "/index.html"):
                servir_html_arquivo(self, HTML_ARENA_PATH)
                return
            if caminho == "/api/state":
                enviar_json_http(self, app.obter_estado_inicial())
                return
            enviar_json_http(self, {"erro": "Rota nao encontrada."}, status=404)

        def do_POST(self) -> None:
            caminho = self.path.split("?", 1)[0]
            dados = ler_json_http(self)

            if caminho == "/api/chat":
                status, payload = app.executar_chat_unico(dados)
                enviar_json_http(self, payload, status=status)
                return

            if caminho == "/api/chat_batch":
                status, payload = app.executar_chat_batch(dados)
                enviar_json_http(self, payload, status=status)
                return

            if caminho == "/api/score_metrics":
                enviar_json_http(self, app.pontuar_metricas(dados))
                return

            if caminho == "/api/preview_prompt":
                enviar_json_http(self, app.studio.preview_prompt(dados))
                return

            if caminho == "/api/promote":
                status, payload = app.promover_prompt(dados)
                enviar_json_http(self, payload, status=status)
                return

            if caminho == "/api/prompt/reload":
                prompt_base = carregar_system_prompt_base(app.studio.prompt_path)
                enviar_json_http(self, {"ok": True, "prompt_base": prompt_base})
                return

            enviar_json_http(self, {"erro": "Rota POST nao encontrada."}, status=404)

    return ArenaPromptsHandler


def iniciar_servidor_arena(
    app: ArenaPromptsApp,
    host: str = "127.0.0.1",
    porta_inicial: int = 8775,
) -> Tuple[ThreadingHTTPServer, int]:
    """Inicia servidor HTTP da Arena escolhendo automaticamente porta livre."""
    return abrir_servidor_porta_livre(
        criar_handler_arena(app),
        host=host,
        porta_inicial=porta_inicial,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arena Multi-Chat de System Prompts (Avaliação 0–5) — PETeleco"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host HTTP (padrao: 127.0.0.1)")
    parser.add_argument("--porta", type=int, default=8775, help="Porta HTTP inicial (padrao: 8775)")
    parser.add_argument("--sem-browser", action="store_true", help="Nao abre o navegador automaticamente")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="URL do Ollama local")
    parser.add_argument(
        "--pacing-s",
        type=float,
        default=4.2,
        help="Intervalo entre chamadas Gemini em lote para evitar HTTP 429 (padrao: 4.2s)",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Caminho do Markdown com o bloco `## 3. System Prompt`",
    )
    args = parser.parse_args()

    app = ArenaPromptsApp(
        prompt_path=args.prompt_path,
        ollama_url=args.ollama_url,
        pacing_seconds=args.pacing_s,
    )
    estado = app.obter_estado_inicial()
    servidor, porta_real = iniciar_servidor_arena(app, host=args.host, porta_inicial=args.porta)
    url = f"http://{args.host}:{porta_real}"

    print("=" * 72)
    print(f"PETELECO ARENA MULTI-CHAT DE PROMPTS ONLINE: {url}")
    print("=" * 72)
    print(f"Variantes: {len(estado['variantes'])} | Cenários: {len(estado['cenarios'])}")
    print(f"Gemini   : {'OK' if estado['gemini_configured'] else 'SEM GEMINI_API_KEY'}")
    print(
        f"Ollama   : {'ONLINE (' + ', '.join(estado['ollama_models']) + ')' if estado['ollama_online'] and estado['ollama_models'] else ('ONLINE (sem modelo de chat)' if estado['ollama_online'] else 'OFFLINE')}"
    )
    print("Modo     : Multi-chat lado a lado com nota 0–5 e métricas (sem escrita em DB).")
    print("Pressione Ctrl+C neste terminal para encerrar.")

    if not args.sem_browser:
        threading_timer = threading.Timer(0.4, lambda: webbrowser.open(url))
        threading_timer.daemon = True
        threading_timer.start()

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando Arena de Prompts...")
    finally:
        servidor.shutdown()
        servidor.server_close()


if __name__ == "__main__":
    main()
