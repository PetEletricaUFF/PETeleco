#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrato Base & Gerenciador de Skills Modulares do PETeleco.

Cada skill é um módulo isolado em `src/skills/` que:
- declara um `criterio_jev` para roteamento automático via `/v1/systemone`;
- executa sem travar o loop conversacional (timeouts curtos + captura de erros);
- devolve `ResultadoSkill` com `resumo_contexto` (injetado em `DADOS_SKILL`)
  e/ou `resposta_direta` (execução determinística em ~0 ms sem gastar LLM).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional
import unicodedata


def normalizar_texto_skill(texto: str) -> str:
    decomp = unicodedata.normalize("NFKD", texto or "")
    sem = "".join(c for c in decomp if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem.lower()).strip()


@dataclass
class ContextoSkill:
    """Snapshot leve passado para qualquer skill durante a execução do turno."""

    transcricao: str
    pessoa_id: str = "desconhecido"
    pessoa_nome: str = "desconhecido"
    modo_persona: str = "NORMAL"
    eh_formal: bool = False
    cerebro_id: str = "resenha"
    memoria: Any = None


@dataclass
class ResultadoSkill:
    """Retorno padronizado de uma skill modular."""

    skill_id: str
    sucesso: bool
    resumo_contexto: str = ""
    resposta_direta: str = ""
    precisa_llm: bool = True
    metadados: Dict[str, Any] = field(default_factory=dict)

    def para_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "sucesso": self.sucesso,
            "resumo_contexto": self.resumo_contexto,
            "resposta_direta": self.resposta_direta,
            "precisa_llm": self.precisa_llm,
            "metadados": self.metadados,
        }


class SkillBase(ABC):
    """Interface base de uma skill plugável do PETeleco."""

    id: str = ""
    nome: str = ""
    descricao: str = ""
    criterio_jev: str = ""
    habilitada: bool = True

    def pode_atender_local(self, transcricao: str) -> bool:
        """Gatilho leve opcional para fallback sem Jev ou comandos inequívocos."""
        return False

    @abstractmethod
    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        """Executa a ação ou coleta de dados da skill."""

    def descrever(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "nome": self.nome,
            "descricao": self.descricao,
            "habilitada": self.habilitada,
            "criterio_jev": self.criterio_jev,
        }


class GerenciadorSkills:
    """Registro central e executor seguro das skills modulares do PETeleco."""

    def __init__(self, skills: Optional[List[SkillBase]] = None) -> None:
        self._skills: Dict[str, SkillBase] = {}
        for sk in skills or []:
            self.registrar(sk)

    def registrar(self, skill: SkillBase) -> None:
        if not skill.id:
            raise ValueError("Skill precisa definir atributo `id`.")
        self._skills[skill.id] = skill

    def obter(self, skill_id: str) -> Optional[SkillBase]:
        return self._skills.get((skill_id or "").strip().lower())

    def listar(self) -> List[Dict[str, Any]]:
        return [sk.descrever() for sk in self._skills.values()]

    def configurar_habilitada(self, skill_id: str, habilitada: bool) -> bool:
        sk = self.obter(skill_id)
        if sk is None:
            return False
        sk.habilitada = bool(habilitada)
        return True

    def construir_criterios_jev(self) -> Dict[str, str]:
        """Gera dinamicamente o mapa `skill_id -> criterio_jev` para o roteador Jev."""
        criterios: Dict[str, str] = {}
        for sk in self._skills.values():
            if sk.habilitada and sk.criterio_jev:
                criterios[sk.id] = sk.criterio_jev
        criterios["nenhuma"] = (
            "Regular conversation, greeting, robot movement, or pure conceptual/technical question "
            "that needs no external utility tool."
        )
        return criterios

    def detectar_skill_local(self, transcricao: str) -> Optional[str]:
        """Fallback local determinístico quando Jev está offline ou o comando é explícito."""
        for sk in self._skills.values():
            if not sk.habilitada:
                continue
            try:
                if sk.pode_atender_local(transcricao):
                    return sk.id
            except Exception:
                continue
        return None

    def executar(self, skill_id: str, ctx: ContextoSkill) -> Optional[ResultadoSkill]:
        """Executa uma skill isolando falhas externas para nunca quebrar a conversa."""
        sk = self.obter(skill_id)
        if sk is None or not sk.habilitada:
            return None
        try:
            return sk.executar(ctx)
        except Exception as exc:
            return ResultadoSkill(
                skill_id=sk.id,
                sucesso=False,
                resumo_contexto=f"A ferramenta '{sk.nome}' falhou temporariamente ({exc.__class__.__name__}).",
                precisa_llm=True,
                metadados={"erro": str(exc)},
            )

    def coletar_recados_pendentes(self, pessoa_id: str) -> str:
        """Consulta e consome recados pendentes caso a skill `recados` esteja registrada."""
        sk = self.obter("recados")
        if sk is None or not sk.habilitada or not hasattr(sk, "consumir_recados_formatados"):
            return ""
        try:
            return str(sk.consumir_recados_formatados(pessoa_id) or "")
        except Exception:
            return ""

    def coletar_timers_disparados(self) -> List[Dict[str, Any]]:
        """Retorna e limpa lembretes/timers já disparados em background."""
        sk = self.obter("timer")
        if sk is None or not hasattr(sk, "consumir_alertas_disparados"):
            return []
        try:
            return list(sk.consumir_alertas_disparados())
        except Exception:
            return []

    def avaliar_sonoplastia(
        self,
        *,
        transcricao: str,
        resposta: str,
        cerebro_id: str,
        modo_persona: str,
        eh_formal: bool,
        reproduzir_audio: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Aciona o avaliador anti-cringe de sonoplastia caso a skill esteja ativa."""
        sk = self.obter("sonoplastia")
        if sk is None or not sk.habilitada or not hasattr(sk, "avaliar_e_tocar"):
            return None
        try:
            return sk.avaliar_e_tocar(
                transcricao=transcricao,
                resposta=resposta,
                cerebro_id=cerebro_id,
                modo_persona=modo_persona,
                eh_formal=eh_formal,
                reproduzir_audio=reproduzir_audio,
            )
        except Exception:
            return None
