#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 1 — Clima em Tempo Real (Open-Meteo, gratuito e sem API key).

Padrão configurado para Niterói/RJ (Praia Vermelha / UFF), personalizável via
variáveis de ambiente `PETELECO_CIDADE`, `PETELECO_LAT`, `PETELECO_LON`.
Usa cache em memória (TTL padrão 15 min) para respostas rápidas sem abuso de API.
"""

import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


_CODIGOS_WMO_PT = {
    0: "céu limpo",
    1: "predominantemente limpo",
    2: "parcialmente nublado",
    3: "nublado",
    45: "neblina",
    48: "neblina forte",
    51: "garoa fraca",
    53: "garoa moderada",
    55: "garoa forte",
    61: "chuva fraca",
    63: "chuva moderada",
    65: "chuva forte",
    80: "pancadas leves de chuva",
    81: "pancadas moderadas de chuva",
    82: "pancadas fortes de chuva",
    95: "tempestade com trovoadas",
    96: "tempestade com granizo leve",
    99: "tempestade forte com granizo",
}


class SkillClima(SkillBase):
    id = "clima"
    nome = "Clima ao Vivo"
    descricao = "Consulta temperatura, sensação térmica e chance de chuva via Open-Meteo (padrão Niterói/RJ)."
    criterio_jev = (
        "Asking for current weather, outside temperature, rain forecast, or if it is hot, cold, or sunny right now "
        "(NOT general science why-questions about the sky or physics)."
    )

    def __init__(
        self,
        cidade: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        cache_ttl_s: float = 900.0,
        timeout_s: float = 5.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cidade = (cidade or os.environ.get("PETELECO_CIDADE") or "Niterói").strip()
        self.latitude = float(latitude if latitude is not None else os.environ.get("PETELECO_LAT", "-22.8833"))
        self.longitude = float(longitude if longitude is not None else os.environ.get("PETELECO_LON", "-43.1036"))
        self.cache_ttl_s = float(cache_ttl_s)
        self.timeout_s = float(timeout_s)
        self._session = session or requests.Session()
        self._cache: Optional[Tuple[float, Dict[str, Any]]] = None

    def pode_atender_local(self, transcricao: str) -> bool:
        norm = normalizar_texto_skill(transcricao)
        padroes = (
            r"\bclima\b",
            r"\bprevisao do tempo\b",
            r"\bvai chover\b",
            r"\bta chovendo\b",
            r"\besta chovendo\b",
            r"\bquantos graus\b",
            r"\btemperatura (la fora|hoje|agora)\b",
            r"\bcomo (ta|esta) o tempo\b",
        )
        return any(re.search(p, norm) for p in padroes)

    def _buscar_open_meteo(self) -> Dict[str, Any]:
        agora = time.monotonic()
        if self._cache and (agora - self._cache[0]) < self.cache_ttl_s:
            return dict(self._cache[1])

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "America/Sao_Paulo",
            "forecast_days": 1,
        }
        resp = self._session.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        cur = data.get("current") or {}
        daily = data.get("daily") or {}

        code = int(cur.get("weather_code", 0) or 0)
        condicao = _CODIGOS_WMO_PT.get(code, "tempo variável")
        prob_chuva = (daily.get("precipitation_probability_max") or [0])[0]
        t_max = (daily.get("temperature_2m_max") or [cur.get("temperature_2m", 0)])[0]
        t_min = (daily.get("temperature_2m_min") or [cur.get("temperature_2m", 0)])[0]

        resumo = {
            "cidade": self.cidade,
            "temp_c": round(float(cur.get("temperature_2m", 0.0)), 1),
            "sensacao_c": round(float(cur.get("apparent_temperature", 0.0)), 1),
            "umidade_pct": int(round(float(cur.get("relative_humidity_2m", 0.0)))),
            "vento_kmh": round(float(cur.get("wind_speed_10m", 0.0)), 1),
            "chuva_mm": round(float(cur.get("precipitation", 0.0)), 1),
            "condicao": condicao,
            "prob_chuva_max_pct": int(round(float(prob_chuva or 0))),
            "temp_min_c": round(float(t_min or 0.0), 1),
            "temp_max_c": round(float(t_max or 0.0), 1),
        }
        self._cache = (agora, resumo)
        return dict(resumo)

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        try:
            dados = self._buscar_open_meteo()
        except Exception as exc:
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=False,
                resumo_contexto=f"Não consegui consultar a previsão do clima para {self.cidade} agora (sem resposta da API).",
                precisa_llm=True,
                metadados={"erro": str(exc)},
            )

        resumo_txt = (
            f"Clima real agora em {dados['cidade']}: {dados['temp_c']:.0f}°C "
            f"(sensação térmica de {dados['sensacao_c']:.0f}°C), {dados['condicao']}, "
            f"umidade {dados['umidade_pct']}%, chance máxima de chuva hoje de {dados['prob_chuva_max_pct']}%, "
            f"mínima {dados['temp_min_c']:.0f}°C e máxima {dados['temp_max_c']:.0f}°C."
        )
        return ResultadoSkill(
            skill_id=self.id,
            sucesso=True,
            resumo_contexto=resumo_txt,
            precisa_llm=True,
            metadados=dados,
        )
