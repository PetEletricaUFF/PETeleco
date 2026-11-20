#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 6 — Busca Rápida na Web & Cotações 100% Grátis (Preserva Free Plan).

Ordem de resolução sem API key e sem usar grounding pago do Gemini:
1. Fast-path de cotações (`dólar`, `euro`, `bitcoin`) via AwesomeAPI (`economiabr`).
2. Snippets web atualizados via `ddgs` (`region="br-pt"`, `max_results=3`).
3. Fallback DuckDuckGo Instant Answer + Wikipedia PT-BR.
"""

import re
import urllib.parse
from typing import List, Optional

import requests

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


try:
    from ddgs import DDGS  # type: ignore

    _HAS_DDGS = True
except Exception:
    DDGS = None
    _HAS_DDGS = False


def _limpar_consulta_web(transcricao: str) -> str:
    limpa = re.sub(
        r"^(?:peteleco[,\s]+)?(?:pesquisa|pesquise|busca|busque|procura|procure|da uma olhada|olha na internet|ve no google|vê no google)\s+(?:sobre|por|na internet|no google|ai|aí)?\s*",
        "",
        (transcricao or "").strip(),
        flags=re.IGNORECASE,
    )
    return limpa.strip(" .,!?")


class SkillBuscaWeb(SkillBase):
    id = "busca_web"
    nome = "Busca Rápida na Web"
    descricao = "Busca cotações e fatos atualizados via AwesomeAPI + DDGS + Wikipedia PT (100% grátis)."
    criterio_jev = (
        "Asking to search the web/Google for a live external fact, definition, sports score, "
        "currency/stock quote, or quick factual lookup not covered by weather or news."
    )

    def __init__(
        self,
        timeout_s: float = 5.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self._session = session or requests.Session()

    def pode_atender_local(self, transcricao: str) -> bool:
        norm = normalizar_texto_skill(transcricao)
        padroes = (
            r"\b(pesquisa|pesquise|busca|busque|procura)\s+(na internet|no google|na web|ai)\b",
            r"\b(ve|olha)\s+no\s+google\b",
            r"\bquanto (ta|esta) o (dolar|bitcoin|euro)\b",
            r"\bcotacao\s+do\s+(dolar|bitcoin|euro)\b",
            r"\bplacar\s+do\s+jogo\b",
        )
        return any(re.search(p, norm) for p in padroes)

    def _buscar_cotacoes(self, consulta: str) -> str:
        norm = normalizar_texto_skill(consulta)
        if not any(k in norm for k in ("dolar", "euro", "bitcoin", "btc", "cotacao", "cambio")):
            return ""
        resp = self._session.get(
            "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL,BTC-BRL",
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        partes: List[str] = []
        if "dolar" in norm or "cambio" in norm or "cotacao" in norm:
            u = data.get("USDBRL") or {}
            if u.get("bid"):
                partes.append(f"Dólar R$ {float(u['bid']):.2f} (variação {u.get('pctChange', '0')}%)")
        if "euro" in norm or "cambio" in norm or "cotacao" in norm:
            e = data.get("EURBRL") or {}
            if e.get("bid"):
                partes.append(f"Euro R$ {float(e['bid']):.2f} (variação {e.get('pctChange', '0')}%)")
        if "bitcoin" in norm or "btc" in norm:
            b = data.get("BTCBRL") or {}
            if b.get("bid"):
                partes.append(f"Bitcoin R$ {float(b['bid']):,.0f}".replace(",", ".") + f" (variação {b.get('pctChange', '0')}%)")
        return "; ".join(partes)

    def _buscar_ddgs_snippets(self, consulta: str) -> str:
        if not _HAS_DDGS or DDGS is None:
            return ""
        resultados = list(DDGS().text(consulta, region="br-pt", safesearch="moderate", max_results=3))
        snippets: List[str] = []
        for it in resultados:
            if not isinstance(it, dict):
                continue
            titulo = str(it.get("title") or "").strip()
            corpo = str(it.get("body") or "").strip()
            trecho = f"{titulo}: {corpo}" if titulo and corpo else (corpo or titulo)
            if trecho:
                snippets.append(trecho[:220])
        return " | ".join(snippets[:3])[:620]

    def _buscar_ddg_instant(self, consulta: str) -> str:
        resp = self._session.get(
            "https://api.duckduckgo.com/",
            params={"q": consulta, "format": "json", "no_html": 1, "skip_disambig": 1, "kl": "br-pt"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        for campo in ("AbstractText", "Answer"):
            txt = str(data.get(campo) or "").strip()
            if txt:
                return txt[:420]
        for rel in data.get("RelatedTopics") or []:
            if isinstance(rel, dict) and rel.get("Text"):
                return str(rel["Text"]).strip()[:420]
        return ""

    def _buscar_wikipedia_pt(self, consulta: str) -> str:
        search_resp = self._session.get(
            "https://pt.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": consulta,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            headers={"User-Agent": "PETeleco-Bot/1.0"},
            timeout=self.timeout_s,
        )
        search_resp.raise_for_status()
        arr = search_resp.json()
        if not isinstance(arr, list) or len(arr) < 2 or not arr[1]:
            return ""
        titulo = str(arr[1][0])
        sum_resp = self._session.get(
            f"https://pt.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(titulo)}",
            headers={"User-Agent": "PETeleco-Bot/1.0"},
            timeout=self.timeout_s,
        )
        sum_resp.raise_for_status()
        extrato = str(sum_resp.json().get("extract") or "").strip()
        return extrato[:450]

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        consulta = _limpar_consulta_web(ctx.transcricao)
        if not consulta:
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=False,
                resumo_contexto="Consulta web vazia.",
                precisa_llm=True,
            )

        resumo = ""
        fonte = ""

        # 1) Fast-path Cotações (AwesomeAPI gratuita)
        try:
            resumo = self._buscar_cotacoes(consulta)
            if resumo:
                fonte = "AwesomeAPI"
        except Exception:
            resumo = ""

        # 2) DuckDuckGo Instant Answer (rápido para definições enciclopédicas)
        if not resumo:
            try:
                resumo = self._buscar_ddg_instant(consulta)
                if resumo:
                    fonte = "DuckDuckGo"
            except Exception:
                resumo = ""

        # 3) Snippets web reais via DDGS (placar, fatos atuais, datasheets)
        if not resumo:
            try:
                resumo = self._buscar_ddgs_snippets(consulta)
                if resumo:
                    fonte = "DuckDuckGo Search"
            except Exception:
                resumo = ""

        # 4) Fallback Wikipedia PT-BR
        if not resumo:
            try:
                resumo = self._buscar_wikipedia_pt(consulta)
                if resumo:
                    fonte = "Wikipedia PT"
            except Exception:
                resumo = ""

        if not resumo:
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=False,
                resumo_contexto=f"Busca rápida por '{consulta}' não retornou resposta direta instantânea.",
                precisa_llm=True,
                metadados={"consulta": consulta, "fonte": "nenhuma"},
            )

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=True,
            resumo_contexto=f"Resultado da busca ({fonte}) para '{consulta}': {resumo}",
            precisa_llm=True,
            metadados={"consulta": consulta, "fonte": fonte},
        )
