#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 9 — Rotinas Estilo Alexa (`Bom dia` & `Modo estudo`).

Orquestra múltiplas skills locais em sequência com um único comando:
- `bom_dia`: data/hora + clima de Niterói + manchetes + recados pendentes + tarefas do grafo.
- `modo_estudo`: Pomodoro de 25 min no `SkillTimer` + silencia sonoplastia + lo-fi baixinho (vol 0.35).
- `sair_modo_estudo`: encerra Pomodoro, para música e restaura sonoplastia.
"""

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from src.grafo_memoria import GrafoMemoriaSQLite

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


def identificar_rotina(transcricao: str) -> str:
    """Retorna `'bom_dia'`, `'sair_modo_estudo'`, `'modo_estudo'` ou `''`."""
    norm = normalizar_texto_skill(transcricao)
    if not norm:
        return ""

    if any(
        k in norm
        for k in (
            "sai do modo estudo",
            "sair do modo estudo",
            "encerrar modo estudo",
            "encerra o modo estudo",
            "desativa o modo estudo",
            "desativar modo estudo",
            "sai do modo foco",
            "desativa o modo foco",
        )
    ):
        return "sair_modo_estudo"

    if any(
        k in norm
        for k in (
            "modo estudo",
            "modo foco",
            "rotina de estudo",
            "hora de estudar",
            "bora estudar",
            "comecar a estudar",
            "iniciar estudo",
        )
    ):
        return "modo_estudo"

    if (
        re.search(r"\bbom\s+dia\s*,?\s*p[eé]\s*tele[ck]o\b", norm)
        or any(k in norm for k in ("rotina bom dia", "resumo da manha", "meu resumo matinal", "resumo matinal"))
    ):
        return "bom_dia"

    return ""


class SkillRotinas(SkillBase):
    id = "rotinas"
    nome = "Rotinas Alexa (Bom dia & Modo Estudo)"
    descricao = "Executa sequências multi-skill com 1 comando (Bom dia / Modo estudo)."
    criterio_jev = (
        "Explicitly triggering a multi-step routine like 'rotina bom dia', 'bom dia peteleco', "
        "'modo estudo', 'ativa o modo foco', or 'sai do modo estudo' (NOT a simple 'bom dia' greeting alone)."
    )

    def __init__(
        self,
        gerenciador_skills: Any = None,
        grafo: Optional[GrafoMemoriaSQLite] = None,
        sqlite_path: Optional[Path] = None,
    ) -> None:
        self._gerenciador = gerenciador_skills
        self.grafo = grafo or GrafoMemoriaSQLite(sqlite_path=sqlite_path)

    def vincular_gerenciador(self, gerenciador_skills: Any) -> None:
        self._gerenciador = gerenciador_skills

    def pode_atender_local(self, transcricao: str) -> bool:
        return bool(identificar_rotina(transcricao))

    def _obter_skill(self, skill_id: str) -> Any:
        if self._gerenciador is None:
            return None
        return self._gerenciador.obter(skill_id)

    def _executar_bom_dia(self, ctx: ContextoSkill) -> ResultadoSkill:
        agora = datetime.now()
        dias = (
            "segunda-feira",
            "terça-feira",
            "quarta-feira",
            "quinta-feira",
            "sexta-feira",
            "sábado",
            "domingo",
        )
        blocos: List[str] = [
            f"Data/hora atual: {dias[agora.weekday()]}, {agora.strftime('%d/%m/%Y %H:%M')}."
        ]
        meta: Dict[str, Any] = {"rotina": "bom_dia"}

        sk_clima = self._obter_skill("clima")
        if sk_clima is not None:
            res_clima = sk_clima.executar(ctx)
            blocos.append(f"Clima: {res_clima.resumo_contexto}")
            meta["clima"] = res_clima.metadados

        sk_not = self._obter_skill("noticias")
        if sk_not is not None:
            res_not = sk_not.executar(ctx)
            if res_not.sucesso and isinstance(res_not.metadados.get("manchetes"), list):
                duas = res_not.metadados["manchetes"][:2]
                blocos.append("Principais notícias: " + " | ".join(duas))
                meta["manchetes"] = duas
            else:
                blocos.append(f"Notícias: {res_not.resumo_contexto}")

        sk_rec = self._obter_skill("recados")
        if sk_rec is not None and hasattr(sk_rec, "consumir_recados_formatados"):
            rec_txt = sk_rec.consumir_recados_formatados(ctx.pessoa_id)
            if rec_txt:
                blocos.append(f"Recados pendentes para {ctx.pessoa_nome}:\n{rec_txt}")
                meta["tinha_recados"] = True

        tarefas = [o["destino_nome"] for o in self.grafo.listar_objetos_por_relacao("lista:tarefas", "contem")]
        if tarefas:
            blocos.append(f"Lista de tarefas pendentes ({len(tarefas)}): " + ", ".join(tarefas) + ".")
            meta["tarefas"] = tarefas

        resumo = (
            f"ROTINA BOM DIA ativada para {ctx.pessoa_nome}. Apresente este resumo matinal de forma fluida:\n"
            + "\n".join(blocos)
        )
        return ResultadoSkill(
            skill_id=self.id,
            sucesso=True,
            resumo_contexto=resumo,
            precisa_llm=True,
            metadados=meta,
        )

    def _executar_modo_estudo(self, ctx: ContextoSkill) -> ResultadoSkill:
        timer_id = None
        sk_timer = self._obter_skill("timer")
        if sk_timer is not None and hasattr(sk_timer, "agendar"):
            timer_id = sk_timer.agendar(
                1500.0,
                "Pomodoro de estudo concluído",
                pessoa_nome=ctx.pessoa_nome,
            )

        sk_sfx = self._obter_skill("sonoplastia")
        if sk_sfx is not None and hasattr(sk_sfx, "auto_ativo"):
            sk_sfx.auto_ativo = False

        faixa_info: Dict[str, Any] = {}
        sk_mus = self._obter_skill("musica")
        if sk_mus is not None and hasattr(sk_mus, "player"):
            faixa_info = sk_mus.player.tocar("lofi", volume=0.35)

        if ctx.eh_formal:
            direta = (
                "Modo estudo ativado. Configurei um temporizador Pomodoro de 25 minutos, "
                "silenciei os efeitos sonoros automáticos e iniciei música lo-fi em volume baixo."
            )
        else:
            direta = (
                "Modo estudo ligado! Botei um Pomodoro de 25 minutos, desliguei as gracinhas de sonoplastia "
                "e soltei um lo-fi baixinho pra tu focar."
            )

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=True,
            resumo_contexto=direta,
            resposta_direta=direta,
            precisa_llm=False,
            metadados={
                "rotina": "modo_estudo",
                "timer_id": timer_id,
                "pomodoro_s": 1500,
                "faixa": faixa_info,
            },
        )

    def _executar_sair_modo_estudo(self, ctx: ContextoSkill) -> ResultadoSkill:
        cancelados = 0
        sk_timer = self._obter_skill("timer")
        if sk_timer is not None and hasattr(sk_timer, "cancelar_todos"):
            cancelados = int(sk_timer.cancelar_todos())

        sk_mus = self._obter_skill("musica")
        if sk_mus is not None and hasattr(sk_mus, "player"):
            sk_mus.player.parar()

        sk_sfx = self._obter_skill("sonoplastia")
        if sk_sfx is not None and hasattr(sk_sfx, "auto_ativo"):
            sk_sfx.auto_ativo = True

        if ctx.eh_formal:
            direta = "Modo estudo encerrado. Música pausada, temporizadores cancelados e efeitos restaurados."
        else:
            direta = "Modo estudo encerrado! Parei a música, limpei o Pomodoro e já voltei ao normal."

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=True,
            resumo_contexto=direta,
            resposta_direta=direta,
            precisa_llm=False,
            metadados={"rotina": "sair_modo_estudo", "timers_cancelados": cancelados},
        )

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        rotina = identificar_rotina(ctx.transcricao)
        if rotina == "bom_dia":
            return self._executar_bom_dia(ctx)
        if rotina == "modo_estudo":
            return self._executar_modo_estudo(ctx)
        if rotina == "sair_modo_estudo":
            return self._executar_sair_modo_estudo(ctx)

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=False,
            resumo_contexto="Não identifiquei qual rotina executar (ex.: 'bom dia PETeleco' ou 'modo estudo').",
            resposta_direta="Tem duas rotinas prontas aqui: 'bom dia PETeleco' ou 'modo estudo'. Qual tu quer?",
            precisa_llm=False,
            metadados={"rotina": "desconhecida"},
        )
