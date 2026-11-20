#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 8 — Listas Persistentes no Grafo de Conhecimento (Compras, Tarefas, PET).

Persiste itens diretamente no `GrafoMemoriaSQLite` usando arestas:
`(lista:<slug>, "contem", item:<slug>)` (ex.: `lista:compras`, `lista:tarefas`).
Executa localmente em ~0 ms (`precisa_llm=False`), sem gastar cota da API.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.grafo_memoria import GrafoMemoriaSQLite, _no_id, _norm, _slug

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


def _normalizar_nome_lista(nome_raw: str) -> str:
    limpo = _norm(nome_raw)
    limpo = re.sub(r"^(?:de|do|da|dos|das)\s+", "", limpo).strip()
    if not limpo:
        return "compras"
    if limpo in ("compra", "mercado", "supermercado"):
        return "compras"
    if limpo in ("tarefa", "afazeres", "fazer", "to do", "todo"):
        return "tarefas"
    return limpo


def extrair_comando_lista(transcricao: str) -> Tuple[str, str, str]:
    """
    Analisa a fala PT-BR e retorna `(acao, nome_lista, item)`:
    - acao: "adicionar" | "remover" | "limpar" | "ler" | "listar_todas" | "nenhuma"
    """
    bruto = re.sub(r"\s+", " ", (transcricao or "").strip())
    norm = normalizar_texto_skill(bruto)

    if any(k in norm for k in ("minhas listas", "quais listas", "listas salvas", "ver todas as listas")):
        return "listar_todas", "", ""

    # Limpar lista
    m_limp = re.search(
        r"\b(?:limpa|limpar|zera|zerar|apaga|esvazia)\s+(?:a\s+)?(?:minha\s+)?lista(?:\s+(?:de|do|da)\s+([a-z0-9_\- ]+?))?(?:[.!?]|$)",
        norm,
    )
    if m_limp:
        return "limpar", _normalizar_nome_lista(m_limp.group(1) or "compras"), ""

    # Marca como feito: "marca resistor 10k como feito na lista de tarefas"
    m_feito = re.search(
        r"\b(?:marca|marcar|risca)\s+(.+?)\s+(?:como\s+)?(?:feito|concluido|comprado|ok)(?:\s+(?:na|da)\s+(?:minha\s+)?lista(?:\s+(?:de|do|da)\s+([A-Za-zÀ-ÿ0-9_\- ]+))?)?(?:[.!?]|$)",
        bruto,
        flags=re.IGNORECASE,
    )
    if m_feito:
        item = m_feito.group(1).strip(" .,!?-")
        nome_lista = _normalizar_nome_lista(m_feito.group(2) or "tarefas")
        if item:
            return "remover", nome_lista, item

    # Remover: "tira/remove X da lista de Y"
    m_rem = re.search(
        r"\b(?:tira|tirar|remove|remover|apaga|apagar|deleta|deletar|exclui)\s+(?:o\s+|a\s+)?(.+?)\s+(?:da|na)\s+(?:minha\s+)?lista(?:\s+(?:de|do|da)\s+([A-Za-zÀ-ÿ0-9_\- ]+?))?(?:[.!?]|$)",
        bruto,
        flags=re.IGNORECASE,
    )
    if m_rem:
        item = m_rem.group(1).strip(" .,!?-")
        nome_lista = _normalizar_nome_lista(m_rem.group(2) or "compras")
        if item:
            return "remover", nome_lista, item

    # Adicionar: "bota/adiciona/coloca/anota X na lista de Y"
    m_add = re.search(
        r"\b(?:bota|botar|coloca|colocar|adiciona|adicionar|poe|põe|anota|anotar|inclui|incluir)\s+(?:o\s+|a\s+)?(.+?)\s+na\s+(?:minha\s+)?lista(?:\s+(?:de|do|da)\s+([A-Za-zÀ-ÿ0-9_\- ]+?))?(?:[.!?]|$)",
        bruto,
        flags=re.IGNORECASE,
    )
    if m_add:
        item = m_add.group(1).strip(" .,!?-")
        nome_lista = _normalizar_nome_lista(m_add.group(2) or "compras")
        if item:
            return "adicionar", nome_lista, item

    # Ler: "o que tem na lista de compras" / "lê minha lista de tarefas"
    m_ler = re.search(
        r"(?:o que tem|quais itens|le|lê|ler|mostra|ver|consulta|quais sao os itens)\s+(?:na|a|da)\s+(?:minha\s+)?lista(?:\s+(?:de|do|da)\s+([A-Za-zÀ-ÿ0-9_\- ]+?))?(?:[.!?]|$)",
        norm,
    )
    if m_ler:
        return "ler", _normalizar_nome_lista(m_ler.group(1) or "compras"), ""

    if re.search(r"\blista\s+de\s+(compras|tarefas|pet)\b", norm):
        m_sub = re.search(r"\blista\s+de\s+([a-z0-9_\-]+)", norm)
        return "ler", _normalizar_nome_lista(m_sub.group(1) if m_sub else "compras"), ""

    return "nenhuma", "", ""


class SkillListas(SkillBase):
    id = "listas"
    nome = "Listas no Grafo"
    descricao = "Gerencia listas de compras, tarefas e materiais no grafo SQLite em 0 ms."
    criterio_jev = (
        "Asking to add, remove, mark done, clear, or read items on a shopping list, to-do list, or lab materials list."
    )

    def __init__(
        self,
        grafo: Optional[GrafoMemoriaSQLite] = None,
        sqlite_path: Optional[Path] = None,
    ) -> None:
        self.grafo = grafo or GrafoMemoriaSQLite(sqlite_path=sqlite_path)

    def pode_atender_local(self, transcricao: str) -> bool:
        acao, _, _ = extrair_comando_lista(transcricao)
        return acao != "nenhuma"

    def listar_itens(self, nome_lista: str) -> List[str]:
        slug_lista = _slug(_normalizar_nome_lista(nome_lista))
        objs = self.grafo.listar_objetos_por_relacao(f"lista:{slug_lista}", "contem")
        return [o["destino_nome"] for o in objs]

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        acao, nome_lista, item = extrair_comando_lista(ctx.transcricao)

        if acao == "listar_todas":
            listas = self.grafo.listar_nos_por_tipo("lista")
            if not listas:
                direta = "Não há listas salvas no momento." if ctx.eh_formal else "Tu ainda não tem nenhuma lista salva aqui."
                return ResultadoSkill(
                    skill_id=self.id,
                    sucesso=True,
                    resumo_contexto="Nenhuma lista cadastrada no grafo.",
                    resposta_direta=direta,
                    precisa_llm=False,
                    metadados={"acao": acao, "listas": []},
                )
            resumos = []
            for ls in listas:
                itens = [o["destino_nome"] for o in self.grafo.listar_objetos_por_relacao(ls["id"], "contem")]
                resumos.append(f"{ls['nome']} ({len(itens)})")
            txt = "Listas salvas: " + ", ".join(resumos) + "."
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=txt,
                resposta_direta=txt,
                precisa_llm=False,
                metadados={"acao": acao, "listas": [l["nome"] for l in listas]},
            )

        lista_norm = _normalizar_nome_lista(nome_lista)
        origem_id = f"lista:{_slug(lista_norm)}"

        if acao == "adicionar" and item:
            dest_id = _no_id("item", item)
            self.grafo.upsert_aresta(
                origem_id=origem_id,
                origem_tipo="lista",
                origem_nome=f"lista de {lista_norm}",
                relacao="contem",
                destino_id=dest_id,
                destino_tipo="item",
                destino_nome=item,
                metodo="skill_listas",
            )
            direta = (
                f"Adicionei {item} à lista de {lista_norm}."
                if ctx.eh_formal
                else f"Anotado! Botei {item} na lista de {lista_norm}."
            )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=f"Item '{item}' adicionado em lista de {lista_norm}.",
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": acao, "lista": lista_norm, "item": item},
            )

        if acao == "remover" and item:
            dest_id = _no_id("item", item)
            removidos = self.grafo.remover_aresta(origem_id, "contem", dest_id)
            if removidos == 0:
                # Tenta busca parcial nos itens existentes
                atuais = self.grafo.listar_objetos_por_relacao(origem_id, "contem")
                norm_item = _norm(item)
                for obj in atuais:
                    if norm_item in _norm(obj["destino_nome"]):
                        removidos += self.grafo.remover_aresta(origem_id, "contem", obj["destino_id"])
            if removidos > 0:
                direta = (
                    f"Removi {item} da lista de {lista_norm}."
                    if ctx.eh_formal
                    else f"Feito, tirei {item} da lista de {lista_norm}!"
                )
            else:
                direta = (
                    f"Não encontrei {item} na lista de {lista_norm}."
                    if ctx.eh_formal
                    else f"Nem achei {item} na lista de {lista_norm} não."
                )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=removidos > 0,
                resumo_contexto=direta,
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": acao, "lista": lista_norm, "item": item, "removidos": removidos},
            )

        if acao == "limpar":
            qtd = self.grafo.remover_arestas_por_origem_relacao(origem_id, "contem")
            direta = (
                f"A lista de {lista_norm} foi limpa ({qtd} itens removidos)."
                if ctx.eh_formal
                else (f"Zerei a lista de {lista_norm}, tirei {qtd} itens." if qtd else f"A lista de {lista_norm} já tava vazia.")
            )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=direta,
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": acao, "lista": lista_norm, "removidos": qtd},
            )

        if acao == "ler":
            itens = self.listar_itens(lista_norm)
            if not itens:
                direta = (
                    f"A lista de {lista_norm} está vazia."
                    if ctx.eh_formal
                    else f"A lista de {lista_norm} tá zerada por enquanto."
                )
            else:
                corpo = ", ".join(itens)
                direta = (
                    f"Na lista de {lista_norm} há {len(itens)} itens: {corpo}."
                    if ctx.eh_formal
                    else f"Na lista de {lista_norm} tem: {corpo}."
                )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=direta,
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": acao, "lista": lista_norm, "itens": itens},
            )

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=False,
            resumo_contexto="Não identifiquei a ação na lista.",
            resposta_direta="Não entendi o que você quer mexer na lista.",
            precisa_llm=False,
            metadados={"acao": "indefinida"},
        )
