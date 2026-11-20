#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 5 — Timers, Pomodoro e Lembretes Falados em Background.

Suporta pedidos naturais em PT-BR ("me avisa em 10 minutos pra olhar a placa",
"bota um timer de 30 segundos", "quanto falta no timer?", "cancela os timers").
Quando o tempo estoura, registra o alerta numa fila thread-safe e dispara o
callback opcional (para falar no TTS ou notificar a UI Web via SSE).
"""

from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


@dataclass
class _TimerAtivo:
    timer_id: int
    rotulo: str
    segundos: float
    dispara_em: float
    pessoa_nome: str
    handle: threading.Timer


def extrair_duracao_e_rotulo(transcricao: str) -> Tuple[str, float, str]:
    """
    Retorna `(acao, segundos, rotulo)`:
    - acao = "criar" | "status" | "cancelar" | "nenhuma"
    """
    bruto = (transcricao or "").strip()
    norm = normalizar_texto_skill(bruto)

    if any(k in norm for k in ("cancela o timer", "cancelar timer", "cancela os timers", "cancela o alarme", "cancela o lembrete")):
        return "cancelar", 0.0, ""
    if any(k in norm for k in ("quanto falta no timer", "status do timer", "quais timers", "tem algum timer")):
        return "status", 0.0, ""

    m = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*(segundos?|seg|s|minutos?|min|horas?|h)\b",
        norm,
    )
    if not m:
        return "nenhuma", 0.0, ""

    valor = float(m.group(1).replace(",", "."))
    unid = m.group(2)
    if unid.startswith("h"):
        segundos = valor * 3600.0
    elif unid.startswith("m"):
        segundos = valor * 60.0
    else:
        segundos = valor

    rotulo = ""
    m_rot = re.search(r"\b(?:pra|para|de)\s+(.+)$", bruto[m.end() :], flags=re.IGNORECASE)
    if m_rot:
        rotulo = m_rot.group(1).strip(" .,!?-")
    if not rotulo:
        rotulo = "timer concluído"

    return "criar", max(1.0, segundos), rotulo


def _formatar_duracao(segundos: float) -> str:
    s = int(round(max(0.0, segundos)))
    if s >= 3600:
        h = s // 3600
        m = (s % 3600) // 60
        return f"{h} hora{'s' if h > 1 else ''}" + (f" e {m} minutos" if m else "")
    if s >= 60:
        m = s // 60
        rem = s % 60
        return f"{m} minuto{'s' if m > 1 else ''}" + (f" e {rem} segundos" if rem else "")
    return f"{s} segundo{'s' if s != 1 else ''}"


class SkillTimer(SkillBase):
    id = "timer"
    nome = "Timer & Lembretes"
    descricao = "Cria contagens regressivas, Pomodoros e lembretes falados em background."
    criterio_jev = (
        "Asking to set a timer, countdown, Pomodoro, alarm, or timed reminder "
        "(e.g. 'me avisa em 10 minutos'), or asking to check/cancel active timers."
    )

    def __init__(self, on_trigger: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self._on_trigger = on_trigger
        self._lock = threading.Lock()
        self._seq = 0
        self._ativos: Dict[int, _TimerAtivo] = {}
        self._disparados: List[Dict[str, Any]] = []

    def definir_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self._on_trigger = callback

    def pode_atender_local(self, transcricao: str) -> bool:
        norm = normalizar_texto_skill(transcricao)
        if not any(k in norm for k in ("timer", "temporizador", "alarme", "pomodoro", "me avisa em", "lembra em", "lembrete em", "cronometro")):
            return False
        acao, _, _ = extrair_duracao_e_rotulo(transcricao)
        return acao in ("criar", "status", "cancelar")

    def _ao_disparar(self, timer_id: int) -> None:
        with self._lock:
            item = self._ativos.pop(timer_id, None)
            if item is None:
                return
            alerta = {
                "timer_id": item.timer_id,
                "rotulo": item.rotulo,
                "pessoa_nome": item.pessoa_nome,
                "segundos": item.segundos,
                "mensagem": f"Ei, deu o tempo do timer: {item.rotulo}!",
            }
            self._disparados.append(alerta)
        if self._on_trigger is not None:
            try:
                self._on_trigger(alerta)
            except Exception:
                pass

    def agendar(self, segundos: float, rotulo: str, pessoa_nome: str = "desconhecido") -> int:
        with self._lock:
            self._seq += 1
            tid = self._seq
            t = threading.Timer(segundos, self._ao_disparar, args=(tid,))
            t.daemon = True
            self._ativos[tid] = _TimerAtivo(
                timer_id=tid,
                rotulo=rotulo,
                segundos=segundos,
                dispara_em=time.monotonic() + segundos,
                pessoa_nome=pessoa_nome,
                handle=t,
            )
            t.start()
            return tid

    def cancelar_todos(self) -> int:
        with self._lock:
            qtd = len(self._ativos)
            for item in self._ativos.values():
                item.handle.cancel()
            self._ativos.clear()
            return qtd

    def listar_ativos(self) -> List[Dict[str, Any]]:
        agora = time.monotonic()
        with self._lock:
            return [
                {
                    "timer_id": it.timer_id,
                    "rotulo": it.rotulo,
                    "restante_s": max(0.0, round(it.dispara_em - agora, 1)),
                }
                for it in self._ativos.values()
            ]

    def consumir_alertas_disparados(self) -> List[Dict[str, Any]]:
        with self._lock:
            saida = list(self._disparados)
            self._disparados.clear()
            return saida

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        acao, segundos, rotulo = extrair_duracao_e_rotulo(ctx.transcricao)

        if acao == "cancelar":
            qtd = self.cancelar_todos()
            if ctx.eh_formal:
                direta = f"Foram cancelados {qtd} temporizadores." if qtd else "Não há temporizadores ativos no momento."
            else:
                direta = f"Cancelei os {qtd} timers que tavam rodando aqui." if qtd else "Não tinha nenhum timer rodando aqui não."
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=f"Cancelados {qtd} timers ativos.",
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": "cancelar", "cancelados": qtd},
            )

        if acao == "status":
            ativos = self.listar_ativos()
            if not ativos:
                direta = "Não há nenhum temporizador ativo." if ctx.eh_formal else "Não tem nenhum timer rodando agora."
            else:
                partes = [f"{a['rotulo']} ({_formatar_duracao(a['restante_s'])} restantes)" for a in ativos]
                direta = "Timers ativos: " + "; ".join(partes) + "."
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=direta,
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": "status", "ativos": ativos},
            )

        if acao == "criar":
            tid = self.agendar(segundos=segundos, rotulo=rotulo, pessoa_nome=ctx.pessoa_nome)
            dur_str = _formatar_duracao(segundos)
            if ctx.eh_formal:
                direta = f"Temporizador configurado para {dur_str} com o lembrete: {rotulo}."
            else:
                direta = f"Fechado! Botei o timer pra {dur_str}, te aviso na hora sobre {rotulo}."
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=f"Timer #{tid} agendado para {dur_str} ({rotulo}).",
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": "criar", "timer_id": tid, "segundos": segundos, "rotulo": rotulo},
            )

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=False,
            resumo_contexto="Não consegui identificar o tempo do timer na frase (ex.: 'em 10 minutos' ou '30 segundos').",
            precisa_llm=True,
            metadados={"acao": "indefinida"},
        )
