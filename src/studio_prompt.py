#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Studio de System Prompt (Text-to-Text) — PETeleco.

Ambiente web leve e isolado para testar, editar e iterar o System Prompt do
chatbot em modo puro texto-para-texto (sem webcam, microfone/STT, TTS, roteador
Jev ou escrita na memoria SQLite/JSON durante os testes).
"""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import webbrowser

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.memoria_pessoas import GerenciadorMemoriaPessoas, ROOT_DIR as MEM_ROOT_DIR, carregar_env
from src.persona_peteleco import (
    DEFAULT_PROMPT_PATH,
    EstadoCorporal,
    MotorPersonaPETeleco,
    carregar_system_prompt_base,
)


HTML_STUDIO_PATH = Path(__file__).resolve().parent / "static" / "studio_prompt.html"

DEFAULT_GEMINI_MODELS: Tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)

_PROMPT_BLOCK_RE = re.compile(
    r"(## 3\. System Prompt[^\n]*\n+```text\n)(.*?)(\n```)",
    flags=re.DOTALL,
)

_MODO_FORMAL_TOMS = frozenset({"serio", "formal", "respeitoso_formal_objetivo"})


def salvar_system_prompt_base(
    novo_prompt: str,
    prompt_path: Optional[Path] = None,
) -> str:
    """
    Atualiza apenas o corpo do System Prompt dentro do bloco `## 3. System Prompt`
    de `prompts/persona_peteleco.md`, preservando o restante do arquivo e a secao
    `=== CONTEXTO DO TURNO` intactos.
    """
    path = Path(prompt_path or DEFAULT_PROMPT_PATH)
    texto_limpo = (novo_prompt or "").strip()
    if not texto_limpo:
        raise ValueError("O System Prompt nao pode ser vazio.")
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de prompt nao encontrado: {path}")

    conteudo = path.read_text(encoding="utf-8")
    match = _PROMPT_BLOCK_RE.search(conteudo)
    if not match:
        raise ValueError(
            "Nao foi possivel localizar o bloco '## 3. System Prompt ... ```text' no arquivo Markdown."
        )

    bloco_antigo = match.group(2)
    marcador = "=== CONTEXTO DO TURNO"
    if marcador in bloco_antigo:
        _, sufixo = bloco_antigo.split(marcador, 1)
        novo_bloco = f"{texto_limpo}\n\n{marcador}{sufixo}"
    else:
        novo_bloco = texto_limpo

    novo_conteudo = (
        conteudo[: match.start(2)] + novo_bloco + conteudo[match.end(2) :]
    )

    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(novo_conteudo, encoding="utf-8")
    tmp_path.replace(path)
    return carregar_system_prompt_base(path)


def listar_modelos_ollama(
    ollama_url: str,
    session: Optional[requests.Session] = None,
    timeout_s: float = 1.5,
) -> Tuple[bool, List[str]]:
    """Consulta modelos de chat locais no Ollama (ignorando modelos `tev1` do SystemOne)."""
    sess = session or requests.Session()
    url = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        resp = sess.get(url, timeout=timeout_s)
        resp.raise_for_status()
        modelos = [
            str(m.get("name", "")).strip()
            for m in resp.json().get("models", [])
            if str(m.get("name", "")).strip()
            and not str(m.get("name", "")).startswith("tev1")
        ]
        return True, modelos
    except Exception:
        return False, []


def _eh_modo_formal_studio(modo_persona: str, perfil: Dict[str, Any]) -> bool:
    if str(modo_persona).upper() == "SERIO":
        return True
    if perfil.get("papel") == "professor_orientador":
        return True
    tom = str(perfil.get("tom_com_a_pessoa") or perfil.get("tom") or "").lower()
    return tom in _MODO_FORMAL_TOMS


def montar_bloco_contexto_studio(
    memoria: GerenciadorMemoriaPessoas,
    contexto_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Monta o bloco `=== CONTEXTO DO TURNO ===` simulado sem gravar nada em disco."""
    cfg = contexto_cfg or {}
    pessoa_id = str(cfg.get("pessoa_id") or "desconhecido").strip()
    if pessoa_id not in memoria.perfis:
        pessoa_id = "desconhecido"
    perfil = memoria.obter_perfil(pessoa_id)

    modo_req = str(cfg.get("modo_persona") or "NORMAL").strip().upper()
    if modo_req not in ("NORMAL", "SERIO", "DESGRACADO"):
        modo_req = "NORMAL"
    modo_efetivo = "SERIO" if _eh_modo_formal_studio(modo_req, perfil) else modo_req

    identificacao = str(cfg.get("identificacao") or "rosto+voz").strip()
    if identificacao not in (
        "rosto+voz",
        "apenas_rosto",
        "apenas_voz",
        "apresentacao_verbal",
        "desconhecido",
    ):
        identificacao = "rosto+voz"

    superficie = str(cfg.get("superficie") or "chao").strip()
    if superficie not in ("chao", "bancada", "no_colo"):
        superficie = "chao"

    movimento = str(cfg.get("movimento") or "parado").strip()
    if movimento not in ("parado", "andando", "seguindo_rosto", "radar"):
        movimento = "parado"

    estado = EstadoCorporal(superficie=superficie, movimento=movimento)
    nome = str(perfil.get("nome", pessoa_id))
    if pessoa_id == "desconhecido":
        pessoa_display = "desconhecido"
    elif identificacao in ("rosto+voz", "apenas_rosto"):
        pessoa_display = f"{nome} (94%)"
    elif identificacao == "apenas_voz":
        pessoa_display = f"{nome} (90% voz)"
    else:
        pessoa_display = nome

    perfil_str, memoria_padrao = memoria.formatar_contexto_prompt(pessoa_id)
    memoria_override = str(cfg.get("memoria_override") or "").strip()
    memoria_str = memoria_override or memoria_padrao

    telemetria_override = str(cfg.get("telemetria_override") or "").strip()
    telemetria_str = telemetria_override or estado.formatar_telemetria()

    sala_override = str(cfg.get("sala_override") or "").strip()
    sala_str = sala_override or "vazio"

    data_str, hora_str = MotorPersonaPETeleco._formatar_data_hora_atual()

    bloco = (
        "=== CONTEXTO DO TURNO (injetado dinamicamente) ===\n"
        f"DATA_HORA_ATUAL: {data_str}, {hora_str}\n"
        f"MODO_PERSONA: {modo_efetivo}\n"
        f"PESSOA_ATUAL: {pessoa_display}\n"
        f"IDENTIFICACAO: {identificacao}\n"
        f"PERFIL_PESSOA: {{{perfil_str}}}\n"
        f"MEMORIA_PESSOA: {memoria_str}\n"
        f"ESTADO_CORPORAL: {{{estado.formatar_prompt()}}}\n"
        f"TELEMETRIA: {telemetria_str}\n"
        f"ULTIMAS_FALAS_SALA: {sala_str}"
    )
    return bloco, modo_efetivo


def montar_system_instruction_completa(
    prompt_base: str,
    incluir_contexto: bool,
    contexto_cfg: Optional[Dict[str, Any]],
    memoria: GerenciadorMemoriaPessoas,
) -> Tuple[str, str]:
    """Retorna `(system_instruction, modo_efetivo)` pronta para envio ao LLM."""
    base = (prompt_base or "").strip()
    if not incluir_contexto:
        return base, "RAW"
    bloco, modo_efetivo = montar_bloco_contexto_studio(memoria, contexto_cfg)
    if not base:
        return bloco, modo_efetivo
    return f"{base}\n\n{bloco}", modo_efetivo


def _sanitizar_erro_http(texto: str, segredo: str = "") -> str:
    limpo = str(texto or "").strip()
    if segredo:
        limpo = limpo.replace(segredo, "***")
    limpo = re.sub(r"key=[^&\s]+", "key=***", limpo)
    return limpo


def _extrair_retry_after_429(detalhe: str, resp: Optional[requests.Response] = None) -> float:
    """Extrai atraso sugerido em segundos de headers Retry-After ou mensagem Gemini 429."""
    if resp is not None:
        hdr = str(resp.headers.get("Retry-After") or "").strip()
        if hdr:
            try:
                return min(65.0, max(2.0, float(hdr)))
            except ValueError:
                pass
    m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s", detalhe or "", flags=re.IGNORECASE)
    if m:
        try:
            return min(65.0, max(2.0, float(m.group(1)) + 1.0))
        except ValueError:
            pass
    return 12.0


def _chamar_gemini_studio(
    *,
    session: requests.Session,
    api_key: str,
    model: str,
    system_inst: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float = 35.0,
    max_retries_429: int = 2,
) -> Dict[str, Any]:
    contents: List[Dict[str, Any]] = []
    for item in messages:
        role = "model" if item.get("role") in ("assistant", "model") else "user"
        text = str(item.get("text", "")).strip()
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})

    if not contents or contents[-1]["role"] != "user":
        return {"ok": False, "error": "Envie uma mensagem de usuario valida."}

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_inst.strip():
        payload["systemInstruction"] = {"parts": [{"text": system_inst}]}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    tentativas = max(1, int(max_retries_429) + 1)
    resp: Optional[requests.Response] = None
    latency_ms = 0.0
    detalhe_ultimo = ""

    for tentativa in range(tentativas):
        t0 = time.perf_counter()
        try:
            resp = session.post(url, json=payload, timeout=timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except requests.RequestException as exc:
            return {
                "ok": False,
                "error": f"Falha de rede ao chamar Gemini ({model}): {_sanitizar_erro_http(str(exc), api_key)}",
            }

        if resp.status_code != 429:
            break

        try:
            detalhe_ultimo = str(resp.json().get("error", {}).get("message") or "").strip()
        except Exception:
            detalhe_ultimo = resp.text[:300].strip()
        detalhe_ultimo = _sanitizar_erro_http(detalhe_ultimo, api_key)
        # Cota diaria esgotada (ex.: "16h34m" ou "_free_tier_requests") nao resolve em 12s
        if re.search(r"retry in\s+\d+(?:\.\d+)?\s*[hm]\b", detalhe_ultimo, flags=re.IGNORECASE):
            break
        if tentativa + 1 < tentativas:
            espera_s = _extrair_retry_after_429(detalhe_ultimo, resp)
            time.sleep(espera_s)

    assert resp is not None
    if resp.status_code >= 400:
        detalhe = detalhe_ultimo
        if not detalhe:
            try:
                err_json = resp.json()
                detalhe = str(err_json.get("error", {}).get("message") or "").strip()
            except Exception:
                detalhe = resp.text[:300].strip()
            detalhe = _sanitizar_erro_http(detalhe, api_key)
        return {
            "ok": False,
            "error": f"Gemini HTTP {resp.status_code} ({model}): {detalhe or 'erro desconhecido'}",
        }

    data = resp.json()
    cands = data.get("candidates") or []
    if not cands:
        block_reason = str(data.get("promptFeedback", {}).get("blockReason") or "").strip()
        motivo = f" (blockReason={block_reason})" if block_reason else ""
        return {"ok": False, "error": f"Gemini nao retornou candidatos para {model}{motivo}."}

    partes = cands[0].get("content", {}).get("parts", [])
    texto = " ".join(
        str(p.get("text", "")).strip()
        for p in partes
        if not p.get("thought") and p.get("text")
    ).strip()
    if not texto:
        finish = str(cands[0].get("finishReason") or "desconhecido")
        return {"ok": False, "error": f"Resposta vazia do Gemini ({model}, finishReason={finish})."}

    usage_raw = data.get("usageMetadata") or {}
    usage = {
        "prompt_tokens": int(usage_raw.get("promptTokenCount") or 0),
        "output_tokens": int(usage_raw.get("candidatesTokenCount") or 0),
        "total_tokens": int(usage_raw.get("totalTokenCount") or 0),
    }
    return {
        "ok": True,
        "reply": texto,
        "provider": "gemini",
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "usage": usage,
    }


def _chamar_ollama_studio(
    *,
    session: requests.Session,
    ollama_url: str,
    model: str,
    system_inst: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    chat_msgs: List[Dict[str, str]] = []
    if system_inst.strip():
        chat_msgs.append({"role": "system", "content": system_inst})

    for item in messages:
        role = "assistant" if item.get("role") in ("assistant", "model") else "user"
        text = str(item.get("text", "")).strip()
        if text:
            chat_msgs.append({"role": role, "content": text})

    if not chat_msgs or chat_msgs[-1]["role"] != "user":
        return {"ok": False, "error": "Envie uma mensagem de usuario valida."}

    payload = {
        "model": model,
        "messages": chat_msgs,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    t0 = time.perf_counter()
    try:
        resp = session.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Falha ao chamar Ollama ({model}): {exc}"}

    data = resp.json()
    texto = str(data.get("message", {}).get("content", "")).strip()
    if not texto:
        return {"ok": False, "error": f"Ollama retornou resposta vazia para {model}."}

    prompt_tok = int(data.get("prompt_eval_count") or 0)
    out_tok = int(data.get("eval_count") or 0)
    return {
        "ok": True,
        "reply": texto,
        "provider": "ollama",
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "usage": {
            "prompt_tokens": prompt_tok,
            "output_tokens": out_tok,
            "total_tokens": prompt_tok + out_tok,
        },
    }


class StudioPromptApp:
    """Mantem configuracoes, leitura/escrita do prompt e chamadas text-to-text isoladas."""

    def __init__(
        self,
        *,
        prompt_path: Optional[Path] = None,
        perfis_path: Optional[Path] = None,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.prompt_path = Path(prompt_path or DEFAULT_PROMPT_PATH)
        self.memoria = GerenciadorMemoriaPessoas(perfis_path=perfis_path)
        self.ollama_url = ollama_url.rstrip("/")
        self._env = carregar_env()
        self._session = requests.Session()

    def _api_key(self) -> str:
        return self._env.get("GEMINI_API_KEY", "").strip()

    def _listar_modelos_gemini(self) -> List[str]:
        modelos: List[str] = []
        for cand in (
            self._env.get("GEMINI_PERSONA_MODEL", ""),
            self._env.get("GEMINI_MODEL", ""),
            *DEFAULT_GEMINI_MODELS,
        ):
            m = cand.strip()
            if m and m not in modelos:
                modelos.append(m)
        return modelos

    def _listar_perfis_ui(self) -> List[Dict[str, Any]]:
        ordem_prioridade = ["caio", "felipe_sass", "desconhecido"]
        chaves = sorted(
            self.memoria.perfis.keys(),
            key=lambda k: (ordem_prioridade.index(k) if k in ordem_prioridade else 99, k),
        )
        itens: List[Dict[str, Any]] = []
        for pid in chaves:
            p = self.memoria.obter_perfil(pid)
            itens.append(
                {
                    "id": pid,
                    "nome": str(p.get("nome", pid)),
                    "papel": str(p.get("papel", "")),
                    "nivel_intimidade": int(p.get("nivel_intimidade", 1)),
                    "tom_com_a_pessoa": str(p.get("tom_com_a_pessoa", "")),
                    "libera_palavrao": bool(p.get("libera_palavrao", False)),
                    "libera_zoeira_pesada": bool(p.get("libera_zoeira_pesada", False)),
                    "usa_bordao_chefe": bool(p.get("usa_bordao_chefe", True)),
                    "tracos_personalidade": list(p.get("tracos_personalidade") or []),
                    "resumo_convivencia": str(p.get("resumo_convivencia") or ""),
                    "ultimos_episodios": list(p.get("ultimos_episodios") or []),
                }
            )
        return itens

    def obter_estado_inicial(self) -> Dict[str, Any]:
        prompt_base = carregar_system_prompt_base(self.prompt_path)
        gemini_models = self._listar_modelos_gemini()
        gemini_ok = bool(self._api_key())
        ollama_online, ollama_models = listar_modelos_ollama(self.ollama_url, session=self._session)

        if gemini_ok:
            default_provider = "gemini"
            default_model = gemini_models[0]
        elif ollama_models:
            default_provider = "ollama"
            default_model = ollama_models[0]
        else:
            default_provider = "gemini"
            default_model = gemini_models[0]

        try:
            rel_prompt = str(self.prompt_path.relative_to(MEM_ROOT_DIR))
        except ValueError:
            rel_prompt = str(self.prompt_path)

        pessoa_padrao = "caio" if "caio" in self.memoria.perfis else "desconhecido"
        return {
            "prompt_base": prompt_base,
            "prompt_path": rel_prompt,
            "gemini_configured": gemini_ok,
            "gemini_models": gemini_models,
            "ollama_online": ollama_online,
            "ollama_url": self.ollama_url,
            "ollama_models": ollama_models,
            "perfis": self._listar_perfis_ui(),
            "defaults": {
                "provider": default_provider,
                "model": default_model,
                "temperature": 0.85,
                "max_tokens": 320,
                "incluir_contexto": True,
                "pessoa_id": pessoa_padrao,
                "modo_persona": "NORMAL",
                "identificacao": "rosto+voz",
                "superficie": "chao",
                "movimento": "parado",
            },
        }

    def preview_prompt(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        prompt_base = str(dados.get("prompt_base") or "")
        incluir_contexto = bool(dados.get("incluir_contexto", True))
        contexto_cfg = dados.get("contexto") if isinstance(dados.get("contexto"), dict) else {}
        system_inst, modo_efetivo = montar_system_instruction_completa(
            prompt_base=prompt_base,
            incluir_contexto=incluir_contexto,
            contexto_cfg=contexto_cfg,
            memoria=self.memoria,
        )
        chars = len(system_inst)
        return {
            "ok": True,
            "system_instruction": system_inst,
            "modo_efetivo": modo_efetivo,
            "chars": chars,
            "approx_tokens": max(1, round(chars / 4)) if chars else 0,
        }

    def executar_chat(self, dados: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        prompt_base = str(dados.get("prompt_base") or "")
        incluir_contexto = bool(dados.get("incluir_contexto", True))
        contexto_cfg = dados.get("contexto") if isinstance(dados.get("contexto"), dict) else {}
        provider = str(dados.get("provider") or "gemini").strip().lower()
        model = str(dados.get("model") or "").strip()
        raw_messages = dados.get("messages")
        messages = raw_messages if isinstance(raw_messages, list) else []

        try:
            temperature = float(dados.get("temperature", 0.85))
        except (TypeError, ValueError):
            temperature = 0.85
        temperature = min(2.0, max(0.0, temperature))

        try:
            max_tokens = int(dados.get("max_tokens", 320))
        except (TypeError, ValueError):
            max_tokens = 320
        max_tokens = min(2048, max(32, max_tokens))

        system_inst, modo_efetivo = montar_system_instruction_completa(
            prompt_base=prompt_base,
            incluir_contexto=incluir_contexto,
            contexto_cfg=contexto_cfg,
            memoria=self.memoria,
        )

        if provider == "gemini":
            api_key = self._api_key()
            if not api_key:
                return 400, {
                    "ok": False,
                    "error": "GEMINI_API_KEY nao configurada no .env (ou alterne para Ollama local).",
                }
            if not model:
                model = self._listar_modelos_gemini()[0]
            res = _chamar_gemini_studio(
                session=self._session,
                api_key=api_key,
                model=model,
                system_inst=system_inst,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider == "ollama":
            if not model:
                _, ol_models = listar_modelos_ollama(self.ollama_url, session=self._session)
                if not ol_models:
                    return 400, {
                        "ok": False,
                        "error": (
                            "Nenhum modelo de chat encontrado no Ollama local "
                            "(ex.: `ollama pull gemma3:4b`; `tev1` e exclusivo do roteador)."
                        ),
                    }
                model = ol_models[0]
            res = _chamar_ollama_studio(
                session=self._session,
                ollama_url=self.ollama_url,
                model=model,
                system_inst=system_inst,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            return 400, {"ok": False, "error": f"Provedor desconhecido: {provider}"}

        if not res.get("ok"):
            return 502, res

        res["modo_efetivo"] = modo_efetivo
        res["system_instruction"] = system_inst
        return 200, res


def enviar_json_http(handler: BaseHTTPRequestHandler, dados: Any, status: int = 200) -> None:
    """Envia resposta JSON UTF-8 sem cache para clientes locais."""
    body = json.dumps(dados, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def ler_json_http(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    """Lê corpo JSON de uma requisição HTTP POST retornando dict seguro."""
    tam = int(handler.headers.get("Content-Length", "0") or 0)
    if tam <= 0:
        return {}
    bruto = handler.rfile.read(tam)
    if not bruto:
        return {}
    try:
        parsed = json.loads(bruto.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def servir_html_arquivo(handler: BaseHTTPRequestHandler, html_path: Path) -> None:
    """Serve um arquivo HTML estático UTF-8 sem cache."""
    if not html_path.exists():
        enviar_json_http(handler, {"erro": f"HTML ausente: {html_path}"}, status=404)
        return
    html = html_path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(html)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(html)


def abrir_servidor_porta_livre(
    handler_cls: Any,
    host: str = "127.0.0.1",
    porta_inicial: int = 8770,
) -> Tuple[ThreadingHTTPServer, int]:
    """Abre um `ThreadingHTTPServer` tentando portas consecutivas antes de usar porta efêmera."""
    base_porta = int(porta_inicial)
    if base_porta > 0:
        for porta in range(base_porta, base_porta + 15):
            try:
                srv = ThreadingHTTPServer((host, porta), handler_cls)
                srv.daemon_threads = True
                return srv, int(srv.server_address[1])
            except OSError:
                continue
    srv = ThreadingHTTPServer((host, 0), handler_cls)
    srv.daemon_threads = True
    return srv, int(srv.server_address[1])


def criar_handler(app: StudioPromptApp):
    """Fabrica o RequestHandler HTTP do Studio de System Prompt."""

    class StudioPromptHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            caminho = self.path.split("?", 1)[0]
            if caminho in ("/", "/index.html"):
                servir_html_arquivo(self, HTML_STUDIO_PATH)
                return

            if caminho == "/api/state":
                enviar_json_http(self, app.obter_estado_inicial())
                return

            enviar_json_http(self, {"erro": "Rota nao encontrada."}, status=404)

        def do_POST(self) -> None:
            caminho = self.path.split("?", 1)[0]
            dados = ler_json_http(self)

            if caminho == "/api/preview_prompt":
                enviar_json_http(self, app.preview_prompt(dados))
                return

            if caminho == "/api/chat":
                status, payload = app.executar_chat(dados)
                enviar_json_http(self, payload, status=status)
                return

            if caminho == "/api/prompt/reload":
                prompt_base = carregar_system_prompt_base(app.prompt_path)
                enviar_json_http(self, {"ok": True, "prompt_base": prompt_base})
                return

            if caminho == "/api/prompt/save":
                novo = str(dados.get("prompt_base") or "")
                try:
                    salvo = salvar_system_prompt_base(novo, app.prompt_path)
                except Exception as exc:
                    enviar_json_http(self, {"ok": False, "error": str(exc)}, status=400)
                    return
                enviar_json_http(
                    self,
                    {
                        "ok": True,
                        "prompt_base": salvo,
                        "message": f"Salvo em {app.prompt_path.name} preservando o restante do documento.",
                    },
                )
                return

            enviar_json_http(self, {"erro": "Rota POST nao encontrada."}, status=404)

    return StudioPromptHandler


def iniciar_servidor_http(
    app: StudioPromptApp,
    host: str = "127.0.0.1",
    porta_inicial: int = 8770,
) -> Tuple[ThreadingHTTPServer, int]:
    """Inicia servidor HTTP escolhendo automaticamente a primeira porta livre."""
    return abrir_servidor_porta_livre(criar_handler(app), host=host, porta_inicial=porta_inicial)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Studio de System Prompt (Text-to-Text) — PETeleco"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host HTTP (padrao: 127.0.0.1)")
    parser.add_argument("--porta", type=int, default=8770, help="Porta HTTP inicial (padrao: 8770)")
    parser.add_argument("--sem-browser", action="store_true", help="Nao abre o navegador automaticamente")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="URL do Ollama local")
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Caminho do Markdown com o bloco `## 3. System Prompt`",
    )
    args = parser.parse_args()

    app = StudioPromptApp(prompt_path=args.prompt_path, ollama_url=args.ollama_url)
    estado = app.obter_estado_inicial()
    servidor, porta_real = iniciar_servidor_http(app, host=args.host, porta_inicial=args.porta)
    url = f"http://{args.host}:{porta_real}"

    print("=" * 72)
    print(f"PETELECO PROMPT STUDIO (TEXT-TO-TEXT) ONLINE: {url}")
    print("=" * 72)
    print(f"Prompt : {estado['prompt_path']}")
    print(f"Gemini : {'OK' if estado['gemini_configured'] else 'SEM GEMINI_API_KEY'}")
    print(
        f"Ollama : {'ONLINE (' + ', '.join(estado['ollama_models']) + ')' if estado['ollama_online'] and estado['ollama_models'] else ('ONLINE (sem modelo de chat)' if estado['ollama_online'] else 'OFFLINE')}"
    )
    print("Modo   : Puro text-to-text (sem webcam, mic/STT, TTS ou escrita em memoria).")
    print("Pressione Ctrl+C neste terminal para encerrar.")

    if not args.sem_browser:
        threading_timer = threading.Timer(0.4, lambda: webbrowser.open(url))
        threading_timer.daemon = True
        threading_timer.start()

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando Studio de System Prompt...")
    finally:
        servidor.shutdown()
        servidor.server_close()


if __name__ == "__main__":
    main()
