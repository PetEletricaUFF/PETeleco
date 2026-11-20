#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 3 — Notícias do Dia (RSS Leve com stdlib `xml.etree.ElementTree`).

Zero dependências pesadas e zero API key: lê feeds públicos (G1 Geral,
Tecnoblog para tecnologia, GE para esportes) com cache de 20 minutos e entrega
as manchetes limpas em `DADOS_SKILL` para o PETeleco resumir na própria voz.
"""

import re
import time
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import requests

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


_FEEDS_RSS: Dict[str, Tuple[str, str]] = {
    "geral": ("G1", "https://g1.globo.com/rss/g1/"),
    "tech": ("Tecnoblog", "https://tecnoblog.net/feed/"),
    "esportes": ("GE", "https://ge.globo.com/rss/ge/"),
}


def detectar_categoria_noticias(transcricao: str) -> str:
    norm = normalizar_texto_skill(transcricao)
    if any(k in norm for k in ("tech", "tecnologia", "ia ", "inteligencia artificial", "hardware", "ciencia", "computacao")):
        return "tech"
    if any(k in norm for k in ("esporte", "futebol", "flamengo", "vasco", "botafogo", "fluminense", "brasileirao", "jogo")):
        return "esportes"
    return "geral"


def extrair_manchetes_rss(xml_texto: str, limite: int = 4) -> List[str]:
    """Faz parse seguro de RSS 2.0 / Atom retornando títulos limpos."""
    raiz = ET.fromstring(xml_texto)
    titulos: List[str] = []

    for item in raiz.findall(".//item"):
        t = item.findtext("title")
        if t:
            limpo = re.sub(r"\s+", " ", t).strip()
            if limpo and limpo not in titulos:
                titulos.append(limpo)
        if len(titulos) >= limite:
            break

    if not titulos:
        # Fallback para Atom (<entry><title>)
        for entry in raiz.findall(".//{http://www.w3.org/2005/Atom}entry"):
            t = entry.findtext("{http://www.w3.org/2005/Atom}title")
            if t:
                limpo = re.sub(r"\s+", " ", t).strip()
                if limpo and limpo not in titulos:
                    titulos.append(limpo)
            if len(titulos) >= limite:
                break
    return titulos


class SkillNoticias(SkillBase):
    id = "noticias"
    nome = "Notícias do Dia"
    descricao = "Busca manchetes recentes via RSS (G1, Tecnoblog, GE) para resumo falado."
    criterio_jev = (
        "Asking for today's news, headlines, what is happening in the world, tech news, or sports news."
    )

    def __init__(
        self,
        cache_ttl_s: float = 1200.0,
        timeout_s: float = 6.0,
        limite_manchetes: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cache_ttl_s = float(cache_ttl_s)
        self.timeout_s = float(timeout_s)
        self.limite_manchetes = int(limite_manchetes)
        self._session = session or requests.Session()
        self._cache: Dict[str, Tuple[float, List[str]]] = {}

    def pode_atender_local(self, transcricao: str) -> bool:
        norm = normalizar_texto_skill(transcricao)
        padroes = (
            r"\bnoticias?\b",
            r"\bmanchetes?\b",
            r"\bo que (ta|esta) acontecendo no mundo\b",
            r"\bnovidades de hoje\b",
            r"\bresumo do dia\b",
        )
        return any(re.search(p, norm) for p in padroes)

    def _buscar_manchetes(self, categoria: str) -> Tuple[str, List[str]]:
        fonte, url = _FEEDS_RSS.get(categoria, _FEEDS_RSS["geral"])
        agora = time.monotonic()
        cached = self._cache.get(categoria)
        if cached and (agora - cached[0]) < self.cache_ttl_s:
            return fonte, list(cached[1])

        resp = self._session.get(
            url,
            headers={"User-Agent": "PETeleco-Bot/1.0 (+https://github.com/PetEletricaUFF/PETeleco)"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        manchetes = extrair_manchetes_rss(resp.text, limite=self.limite_manchetes)
        self._cache[categoria] = (agora, manchetes)
        return fonte, list(manchetes)

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        categoria = detectar_categoria_noticias(ctx.transcricao)
        try:
            fonte, manchetes = self._buscar_manchetes(categoria)
        except Exception as exc:
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=False,
                resumo_contexto=f"Não consegui puxar o feed de notícias ({categoria}) agora.",
                precisa_llm=True,
                metadados={"categoria": categoria, "erro": str(exc)},
            )

        if not manchetes:
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=False,
                resumo_contexto=f"O feed de notícias ({fonte}) veio vazio agora.",
                precisa_llm=True,
                metadados={"categoria": categoria, "fonte": fonte},
            )

        lista_fmt = " | ".join(f"{idx}. {m}" for idx, m in enumerate(manchetes, 1))
        return ResultadoSkill(
            skill_id=self.id,
            sucesso=True,
            resumo_contexto=f"Manchetes reais de hoje ({fonte} / {categoria}): {lista_fmt}",
            precisa_llm=True,
            metadados={"categoria": categoria, "fonte": fonte, "manchetes": manchetes},
        )
