#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 4 — Mural de Recados por Pessoa (SQLite + Entrega Automática por Rosto/Voz).

Permite deixar recados para colegas do PET ("avisa o Heitor que deixei a fonte
no armário") e entregá-los automaticamente assim que essa pessoa aparecer na
câmera ou falar com o PETeleco.
"""

from contextlib import closing
from datetime import datetime
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memoria_longa.sqlite"


def extrair_recado(transcricao: str) -> Tuple[str, str, str]:
    """
    Analisa a fala e retorna `(acao, destinatario_raw, mensagem)`:
    - acao = "deixar" | "ler" | "nenhuma"
    """
    bruto = (transcricao or "").strip()
    norm = normalizar_texto_skill(bruto)

    if any(
        k in norm
        for k in (
            "tem recado pra mim",
            "tem algum recado",
            "meus recados",
            "deixaram recado",
            "alguem deixou recado",
            "ler recados",
            "quais sao os recados",
        )
    ):
        return "ler", "", ""

    padroes = (
        r"(?:avisa|avise|fala\s+pro|fala\s+pra|diz\s+pro|diz\s+pra)\s+(?:o\s+|a\s+|ao\s+|pro\s+|pra\s+)?([A-Za-zÀ-ÿ0-9_]+)\s+que\s+(.+)",
        r"(?:deixa|deixe|manda|anota)\s+(?:um\s+)?recado\s+(?:pro|pra|para\s+o|para\s+a|para)\s+([A-Za-zÀ-ÿ0-9_]+)\s*(?:dizendo\s+que|que|:|-)?\s*(.+)",
    )
    for padrao in padroes:
        m = re.search(padrao, bruto, flags=re.IGNORECASE)
        if m:
            dest = m.group(1).strip(" ,.:;!?")
            msg = m.group(2).strip(" ,.:;!?")
            if dest and msg:
                return "deixar", dest, msg

    return "nenhuma", "", ""


class SkillRecados(SkillBase):
    id = "recados"
    nome = "Mural de Recados do PET"
    descricao = "Salva recados em SQLite por pessoa e entrega automaticamente quando o destinatário interagir."
    criterio_jev = (
        "Asking the robot to leave, pass, or tell a message/note to another colleague/person "
        "(e.g. 'avisa o Heitor que...', 'fala pro...', 'deixa um recado para...'), "
        "or asking if anyone left messages for them."
    )

    def __init__(self, sqlite_path: Optional[Path] = None) -> None:
        self.sqlite_path = Path(sqlite_path) if sqlite_path else DEFAULT_SQLITE_PATH
        self._init_db()

    def _init_db(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS recados_pendentes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    de_pessoa_id TEXT NOT NULL,
                    de_nome TEXT NOT NULL,
                    para_pessoa_id TEXT NOT NULL,
                    para_nome TEXT NOT NULL,
                    mensagem TEXT NOT NULL,
                    criado_em TEXT NOT NULL,
                    entregue_em TEXT DEFAULT NULL
                )
                """
            )
            con.commit()

    def pode_atender_local(self, transcricao: str) -> bool:
        acao, _, _ = extrair_recado(transcricao)
        return acao in ("deixar", "ler")

    def salvar_recado(
        self,
        *,
        de_pessoa_id: str,
        de_nome: str,
        para_pessoa_id: str,
        para_nome: str,
        mensagem: str,
    ) -> int:
        agora = datetime.now().strftime("%Y-%m-%d %H:%M")
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            cur = con.execute(
                """
                INSERT INTO recados_pendentes
                (de_pessoa_id, de_nome, para_pessoa_id, para_nome, mensagem, criado_em)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (de_pessoa_id, de_nome, para_pessoa_id, para_nome, mensagem.strip(), agora),
            )
            con.commit()
            return int(cur.lastrowid or 0)

    def listar_pendentes(self, para_pessoa_id: str) -> List[Dict[str, Any]]:
        pid = (para_pessoa_id or "").strip().lower()
        if not pid or pid == "desconhecido":
            return []
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            rows = con.execute(
                """
                SELECT id, de_pessoa_id, de_nome, mensagem, criado_em
                FROM recados_pendentes
                WHERE para_pessoa_id = ? AND entregue_em IS NULL
                ORDER BY id ASC
                """,
                (pid,),
            ).fetchall()
        return [
            {
                "id": int(r[0]),
                "de_pessoa_id": str(r[1]),
                "de_nome": str(r[2]),
                "mensagem": str(r[3]),
                "criado_em": str(r[4]),
            }
            for r in rows
        ]

    def marcar_entregues(self, ids: List[int]) -> None:
        if not ids:
            return
        agora = datetime.now().strftime("%Y-%m-%d %H:%M")
        placeholders = ",".join("?" for _ in ids)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            con.execute(
                f"UPDATE recados_pendentes SET entregue_em = ? WHERE id IN ({placeholders})",
                [agora, *ids],
            )
            con.commit()

    def consumir_recados_formatados(self, pessoa_id: str) -> str:
        """Retorna texto dos recados pendentes para `pessoa_id` e marca como entregues."""
        pendentes = self.listar_pendentes(pessoa_id)
        if not pendentes:
            return ""
        self.marcar_entregues([item["id"] for item in pendentes])
        linhas = [
            f"- De {item['de_nome']} ({item['criado_em']}): \"{item['mensagem']}\""
            for item in pendentes
        ]
        return "\n".join(linhas)

    def _resolver_destinatario(self, dest_raw: str, memoria: Any) -> Tuple[str, str]:
        if memoria is not None and hasattr(memoria, "mapear_identificador"):
            pid = memoria.mapear_identificador(dest_raw)
            if pid:
                perfil = memoria.obter_perfil(pid)
                return pid, str(perfil.get("nome") or dest_raw)
        limpo = normalizar_texto_skill(dest_raw).replace(" ", "_")
        return limpo, dest_raw.capitalize()

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        acao, dest_raw, msg = extrair_recado(ctx.transcricao)

        if acao == "ler":
            texto_recados = self.consumir_recados_formatados(ctx.pessoa_id)
            if not texto_recados:
                return ResultadoSkill(
                    skill_id=self.id,
                    sucesso=True,
                    resumo_contexto=f"Nenhum recado pendente no mural para {ctx.pessoa_nome}.",
                    precisa_llm=True,
                    metadados={"acao": "ler", "qtd": 0},
                )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=f"Recados pendentes entregues agora para {ctx.pessoa_nome}:\n{texto_recados}",
                precisa_llm=True,
                metadados={"acao": "ler", "recados": texto_recados},
            )

        if acao == "deixar" and dest_raw and msg:
            para_id, para_nome = self._resolver_destinatario(dest_raw, ctx.memoria)
            rid = self.salvar_recado(
                de_pessoa_id=ctx.pessoa_id,
                de_nome=ctx.pessoa_nome,
                para_pessoa_id=para_id,
                para_nome=para_nome,
                mensagem=msg,
            )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=(
                    f"Recado #{rid} salvo no mural para {para_nome} (id={para_id}), "
                    f"enviado por {ctx.pessoa_nome}: \"{msg}\". Será entregue automaticamente quando {para_nome} falar ou aparecer."
                ),
                precisa_llm=True,
                metadados={"acao": "deixar", "recado_id": rid, "para_id": para_id, "para_nome": para_nome},
            )

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=False,
            resumo_contexto="Não identifiquei claramente para quem é o recado e qual é a mensagem.",
            precisa_llm=True,
            metadados={"acao": "indefinida"},
        )
