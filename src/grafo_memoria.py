#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memória em Grafo Embarcada no SQLite (`data/memoria_longa.sqlite`).

Relaciona pessoas, matérias, projetos, objetos do laboratório, recados e listas
em triplas `(Sujeito) --[relacao]--> (Objeto)` com custo zero de servidor externo
e zero VRAM. Permite injetar apenas um subgrafo enxuto de 1–2 saltos relevante ao
turno (~40–90 tokens) sem inflar o System Prompt.

Implementa dois extratores + modo híbrido para benchmark de eficiência vs. custo:
- `local`: regex PT-BR determinístico (`0 tokens`, `< 2 ms`).
- `llm`: extrator JSON via Gemini Flash-Lite (`responseMimeType="application/json"`).
- `hibrido`: tenta `local` primeiro e só escala ao `llm` em gatilhos explícitos complexos.
"""

from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import requests


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = ROOT_DIR / "data" / "memoria_longa.sqlite"


def _norm(texto: str) -> str:
    decomp = unicodedata.normalize("NFKD", texto or "")
    sem = "".join(c for c in decomp if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem.lower()).strip()


def _slug(texto: str) -> str:
    limpo = re.sub(r"[^a-z0-9]+", "_", _norm(texto)).strip("_")
    return limpo or "item"


def _no_id(tipo: str, nome: str) -> str:
    return f"{tipo.strip().lower()}:{_slug(nome)}"


def _limpar_termo(texto: str) -> str:
    return re.sub(r"\s+", " ", (texto or "").strip(" .,!?;:\"'"))


class GrafoMemoriaSQLite:
    """Grafo de conhecimento leve persistido em SQLite para contexto por subgrafo."""

    def __init__(self, sqlite_path: Optional[Path] = None) -> None:
        self.sqlite_path = Path(sqlite_path) if sqlite_path else DEFAULT_SQLITE_PATH
        self._init_db()

    def _init_db(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS grafo_nos (
                    id TEXT PRIMARY KEY,
                    tipo TEXT NOT NULL,
                    nome TEXT NOT NULL,
                    props_json TEXT DEFAULT '{}',
                    atualizado_em TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS grafo_arestas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origem_id TEXT NOT NULL,
                    relacao TEXT NOT NULL,
                    destino_id TEXT NOT NULL,
                    destino_nome TEXT NOT NULL,
                    peso REAL DEFAULT 1.0,
                    metodo TEXT DEFAULT 'local',
                    atualizado_em TEXT NOT NULL,
                    UNIQUE(origem_id, relacao, destino_id)
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_grafo_arestas_origem ON grafo_arestas (origem_id, relacao)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_grafo_arestas_destino ON grafo_arestas (destino_id)"
            )
            con.commit()

    def upsert_no(
        self,
        no_id: str,
        tipo: str,
        nome: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> str:
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        props_str = json.dumps(props or {}, ensure_ascii=False)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            con.execute(
                """
                INSERT INTO grafo_nos (id, tipo, nome, props_json, atualizado_em)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    tipo = excluded.tipo,
                    nome = excluded.nome,
                    props_json = excluded.props_json,
                    atualizado_em = excluded.atualizado_em
                """,
                (no_id, tipo.strip().lower(), nome.strip(), props_str, agora),
            )
            con.commit()
        return no_id

    def upsert_aresta(
        self,
        origem_id: str,
        origem_tipo: str,
        origem_nome: str,
        relacao: str,
        destino_id: str,
        destino_tipo: str,
        destino_nome: str,
        peso: float = 1.0,
        metodo: str = "local",
    ) -> None:
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rel_norm = _slug(relacao)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            con.execute(
                """
                INSERT INTO grafo_nos (id, tipo, nome, props_json, atualizado_em)
                VALUES (?, ?, ?, '{}', ?)
                ON CONFLICT(id) DO UPDATE SET
                    nome = excluded.nome,
                    atualizado_em = excluded.atualizado_em
                """,
                (origem_id, origem_tipo.strip().lower(), origem_nome.strip(), agora),
            )
            con.execute(
                """
                INSERT INTO grafo_nos (id, tipo, nome, props_json, atualizado_em)
                VALUES (?, ?, ?, '{}', ?)
                ON CONFLICT(id) DO UPDATE SET
                    nome = excluded.nome,
                    atualizado_em = excluded.atualizado_em
                """,
                (destino_id, destino_tipo.strip().lower(), destino_nome.strip(), agora),
            )
            con.execute(
                """
                INSERT INTO grafo_arestas (origem_id, relacao, destino_id, destino_nome, peso, metodo, atualizado_em)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(origem_id, relacao, destino_id) DO UPDATE SET
                    destino_nome = excluded.destino_nome,
                    peso = excluded.peso,
                    metodo = excluded.metodo,
                    atualizado_em = excluded.atualizado_em
                """,
                (origem_id, rel_norm, destino_id, destino_nome.strip(), float(peso), metodo, agora),
            )
            con.commit()

    def remover_aresta(self, origem_id: str, relacao: str, destino_id: str) -> int:
        rel_norm = _slug(relacao)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            cur = con.execute(
                "DELETE FROM grafo_arestas WHERE origem_id = ? AND relacao = ? AND destino_id = ?",
                (origem_id, rel_norm, destino_id),
            )
            con.commit()
            return int(cur.rowcount or 0)

    def remover_arestas_por_origem_relacao(self, origem_id: str, relacao: str) -> int:
        rel_norm = _slug(relacao)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            cur = con.execute(
                "DELETE FROM grafo_arestas WHERE origem_id = ? AND relacao = ?",
                (origem_id, rel_norm),
            )
            con.commit()
            return int(cur.rowcount or 0)

    def listar_objetos_por_relacao(self, origem_id: str, relacao: str) -> List[Dict[str, Any]]:
        rel_norm = _slug(relacao)
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            rows = con.execute(
                """
                SELECT a.destino_id, a.destino_nome, a.peso, a.metodo, a.atualizado_em
                FROM grafo_arestas a
                WHERE a.origem_id = ? AND a.relacao = ?
                ORDER BY a.id ASC
                """,
                (origem_id, rel_norm),
            ).fetchall()
        return [
            {
                "destino_id": str(r[0]),
                "destino_nome": str(r[1]),
                "peso": float(r[2]),
                "metodo": str(r[3]),
                "atualizado_em": str(r[4]),
            }
            for r in rows
        ]

    def listar_nos_por_tipo(self, tipo: str) -> List[Dict[str, Any]]:
        t = (tipo or "").strip().lower()
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            rows = con.execute(
                "SELECT id, nome, atualizado_em FROM grafo_nos WHERE tipo = ? ORDER BY atualizado_em DESC, nome ASC",
                (t,),
            ).fetchall()
        return [{"id": str(r[0]), "nome": str(r[1]), "atualizado_em": str(r[2])} for r in rows]

    def buscar_subgrafo(
        self,
        pessoa_id: str,
        transcricao: str,
        max_saltos: int = 2,
        limite: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Busca arestas relevantes (1 a 2 saltos) ancoradas na `pessoa_id` atual
        e em quaisquer entidades cujo nome apareça na `transcricao`.
        """
        pid = (pessoa_id or "desconhecido").strip().lower()
        sementes = set()
        if pid and pid != "desconhecido":
            sementes.add(f"pessoa:{_slug(pid)}")

        norm_texto = f" {_norm(transcricao)} "
        with closing(sqlite3.connect(self.sqlite_path)) as con:
            todos_nos = con.execute("SELECT id, tipo, nome FROM grafo_nos").fetchall()
            mapa_nomes: Dict[str, str] = {}
            for nid, _tipo, nome in todos_nos:
                sid = str(nid)
                snome = str(nome)
                mapa_nomes[sid] = snome
                norm_nome = _norm(snome)
                if len(norm_nome) >= 3 and re.search(rf"\b{re.escape(norm_nome)}\b", norm_texto):
                    sementes.add(sid)

            if not sementes:
                return []

            visitados = set(sementes)
            fronteira = set(sementes)
            coletadas: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

            for _salto in range(max(1, int(max_saltos))):
                if not fronteira:
                    break
                placeholders = ",".join("?" for _ in fronteira)
                params = list(fronteira) + list(fronteira)
                rows = con.execute(
                    f"""
                    SELECT id, origem_id, relacao, destino_id, destino_nome, peso, metodo, atualizado_em
                    FROM grafo_arestas
                    WHERE origem_id IN ({placeholders}) OR destino_id IN ({placeholders})
                    ORDER BY atualizado_em DESC, id DESC
                    """,
                    params,
                ).fetchall()

                proxima_fronteira = set()
                for row_id, oid, rel, did, dnome, peso, metodo, ts in rows:
                    chave = (str(oid), str(rel), str(did))
                    if chave not in coletadas:
                        coletadas[chave] = {
                            "row_id": int(row_id),
                            "origem_id": str(oid),
                            "origem_nome": mapa_nomes.get(str(oid), str(oid).split(":", 1)[-1]),
                            "relacao": str(rel),
                            "destino_id": str(did),
                            "destino_nome": mapa_nomes.get(str(did), str(dnome)),
                            "peso": float(peso),
                            "metodo": str(metodo),
                            "atualizado_em": str(ts),
                        }
                    if str(oid) not in visitados:
                        visitados.add(str(oid))
                        proxima_fronteira.add(str(oid))
                    if str(did) not in visitados:
                        visitados.add(str(did))
                        proxima_fronteira.add(str(did))
                fronteira = proxima_fronteira

        ordenado = sorted(
            coletadas.values(),
            key=lambda x: (x["atualizado_em"], x["row_id"]),
            reverse=True,
        )
        return ordenado[: max(1, int(limite))]

    def formatar_subgrafo_prompt(
        self,
        pessoa_id: str,
        transcricao: str,
        limite: int = 8,
    ) -> str:
        """Formata o subgrafo em triplas compactas (~40–80 tokens) para o prompt."""
        triplas = self.buscar_subgrafo(pessoa_id=pessoa_id, transcricao=transcricao, limite=limite)
        if not triplas:
            return "vazio"
        partes = [
            f"({t['origem_nome']}) --[{t['relacao']}]--> ({t['destino_nome']})"
            for t in triplas
        ]
        return "; ".join(partes)

    @staticmethod
    def extrair_triplas_local(
        transcricao: str,
        pessoa_id: str,
        pessoa_nome: str,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Extrator determinístico PT-BR (`0 tokens`, `< 2 ms`).
        Retorna `(triplas, telemetria)`.
        """
        t0 = time.perf_counter()
        texto = re.sub(r"\s+", " ", (transcricao or "").strip())
        suj_pessoa = pessoa_nome.strip() if pessoa_nome and pessoa_nome != "desconhecido" else pessoa_id
        triplas: List[Dict[str, str]] = []
        vistos = set()

        def _add(suj: str, tipo_s: str, rel: str, obj: str, tipo_o: str) -> None:
            s = _limpar_termo(suj)
            o = _limpar_termo(obj)
            if not s or not o or len(o) < 2:
                return
            chave = (_norm(s), rel, _norm(o))
            if chave in vistos:
                return
            vistos.add(chave)
            triplas.append(
                {
                    "sujeito": s,
                    "tipo_sujeito": tipo_s,
                    "relacao": rel,
                    "objeto": o,
                    "tipo_objeto": tipo_o,
                }
            )

        # Perguntas não devem gravar fatos especulativos como estado
        eh_pergunta = texto.rstrip().endswith("?")

        if not eh_pergunta:
            # 1) Fatos explícitos ("lembra que o Heitor mora em Icaraí", "anota que a chave fica com o Caio")
            for m in re.finditer(
                r"\b(?:lembra|anota|guarda|salva)\s+que\s+(?:o\s+|a\s+)?(.+?)\s+(?:[eé]|est[aá]|ta|t[aá]|fica)\s+(?:no\s+|na\s+|em\s+)?(.+?)(?:[.!;]|$)",
                texto,
                flags=re.IGNORECASE,
            ):
                _add(m.group(1), "entidade", "fato", m.group(2), "conceito")

            # 2) Localização de objetos/ferramentas ("guardei o multímetro no armário 2" / "a fonte tá na bancada")
            for m in re.finditer(
                r"(?:guardei|deixei|botei|coloquei)\s+(?:o|a)\s+(.+?)\s+(?:no|na|em|dentro\s+do|dentro\s+da)\s+(.+?)(?:[.!,;]|$)",
                texto,
                flags=re.IGNORECASE,
            ):
                _add(m.group(1), "objeto", "esta_em", m.group(2), "local")

            if not any(t["relacao"] == "esta_em" for t in triplas):
                for m in re.finditer(
                    r"(?:^|[.!?,]\s*)(?:o|a)\s+([A-Za-zÀ-ÿ0-9_\- ]{2,40}?)\s+(?:est[aá]|ta|t[aá]|ficou)\s+(?:no|na|em|dentro\s+do|dentro\s+da)\s+(.+?)(?:[.!,;]|$)",
                    texto,
                    flags=re.IGNORECASE,
                ):
                    suj_cand = _norm(m.group(1))
                    if suj_cand not in ("clima", "tempo", "jogo", "gente", "voce", "tu"):
                        _add(m.group(1), "objeto", "esta_em", m.group(2), "local")

            # 3) Matérias / estudos
            for m in re.finditer(
                r"(?:eu\s+)?(?:curso|estudo|t[oô]\s+fazendo|estou\s+fazendo|tenho\s+prova\s+de)\s+([A-Za-zÀ-ÿ0-9_\- ]{2,50}?)(?:\s+(?:amanh[aã]|hoje|agora|nessa|na|pela|esse\s+per[ií]odo|este\s+per[ií]odo|nesse\s+per[ií]odo)|[.,!;]|$)",
                texto,
                flags=re.IGNORECASE,
            ):
                _add(suj_pessoa, "pessoa", "estuda", m.group(1), "materia")

            # 4) Projetos
            for m in re.finditer(
                r"(?:meu\s+projeto\s+(?:[eé]|eh)|trabalho\s+n[oa]|desenvolvo\s+[oa]|t[oô]\s+mexendo\s+n[oa]|estou\s+mexendo\s+n[oa])\s+([A-Za-zÀ-ÿ0-9_\- ]{2,50}?)(?:[.!,;]|$)",
                texto,
                flags=re.IGNORECASE,
            ):
                _add(suj_pessoa, "pessoa", "trabalha_em", m.group(1), "projeto")

            # 5) Preferências / hábitos
            for m in re.finditer(
                r"(?:eu\s+)?(?:gosto\s+de|curto|amo|prefiro|vivo\s+[àa]\s+base\s+de|vivo\s+a\s+base\s+de)\s+([A-Za-zÀ-ÿ0-9_\- ]{2,45}?)(?:[.!,;]|$)",
                texto,
                flags=re.IGNORECASE,
            ):
                _add(suj_pessoa, "pessoa", "gosta_de", m.group(1), "preferencia")

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return triplas, {
            "metodo": "local",
            "latencia_ms": round(lat_ms, 3),
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def extrair_triplas_llm(
        transcricao: str,
        pessoa_id: str,
        pessoa_nome: str,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        session: Optional[requests.Session] = None,
        timeout_s: float = 12.0,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Extrator semântico via Gemini Flash-Lite JSON para benchmark e casos complexos."""
        suj_pessoa = pessoa_nome.strip() if pessoa_nome and pessoa_nome != "desconhecido" else pessoa_id
        key = (api_key or "").strip()
        if not key or not (transcricao or "").strip():
            return [], {
                "metodo": "llm",
                "latencia_ms": 0.0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "erro": "sem_api_key_ou_texto",
            }

        prompt = (
            "Extraia fatos objetivos em triplas de grafo de conhecimento a partir da fala em PT-BR.\n"
            f"FALANTE_ATUAL: {suj_pessoa}\n"
            f'FALA: "{transcricao.strip()}"\n\n'
            "Regras:\n"
            "- Se a fala for apenas saudação, pergunta simples ou conversa sem fatos duráveis sobre pessoas/objetos/projetos/preferências, retorne [].\n"
            "- Relações preferidas: esta_em, estuda, trabalha_em, gosta_de, mora_em, fato.\n"
            '- Retorne JSON exatamente como lista de objetos: [{"sujeito": str, "tipo_sujeito": str, "relacao": str, "objeto": str, "tipo_objeto": str}]'
        )
        sess = session or requests.Session()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 180,
                "responseMimeType": "application/json",
            },
        }
        t0 = time.perf_counter()
        try:
            resp = sess.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            data = resp.json()
            usage = data.get("usageMetadata") or {}
            partes = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            bruto = "".join(str(p.get("text", "")) for p in partes if not p.get("thought")).strip()
            parsed = json.loads(bruto)
            itens = parsed if isinstance(parsed, list) else []
            triplas: List[Dict[str, str]] = []
            for it in itens:
                if not isinstance(it, dict):
                    continue
                suj = _limpar_termo(str(it.get("sujeito") or ""))
                rel = _slug(str(it.get("relacao") or "fato"))
                obj = _limpar_termo(str(it.get("objeto") or ""))
                if suj and obj:
                    triplas.append(
                        {
                            "sujeito": suj,
                            "tipo_sujeito": _slug(str(it.get("tipo_sujeito") or "entidade")),
                            "relacao": rel,
                            "objeto": obj,
                            "tipo_objeto": _slug(str(it.get("tipo_objeto") or "conceito")),
                        }
                    )
            ptok = int(usage.get("promptTokenCount") or 0)
            otok = int(usage.get("candidatesTokenCount") or 0)
            return triplas, {
                "metodo": "llm",
                "latencia_ms": round(lat_ms, 2),
                "prompt_tokens": ptok,
                "output_tokens": otok,
                "total_tokens": int(usage.get("totalTokenCount") or (ptok + otok)),
            }
        except Exception as exc:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return [], {
                "metodo": "llm",
                "latencia_ms": round(lat_ms, 2),
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "erro": str(exc),
            }

    @staticmethod
    def _tem_gatilho_complexo_memoria(transcricao: str) -> bool:
        norm = _norm(transcricao)
        return any(
            k in norm
            for k in (
                "lembra que",
                "anota que",
                "guarda que",
                "guarda ai",
                "na verdade",
                "mora em",
                "namora",
                "orientador",
                "matricula",
            )
        )

    def _gravar_triplas(self, triplas: List[Dict[str, str]], metodo: str) -> int:
        gravadas = 0
        for tr in triplas:
            suj = tr.get("sujeito", "").strip()
            obj = tr.get("objeto", "").strip()
            if not suj or not obj:
                continue
            tsuj = tr.get("tipo_sujeito", "entidade")
            tobj = tr.get("tipo_objeto", "conceito")
            rel = tr.get("relacao", "fato")
            self.upsert_aresta(
                origem_id=_no_id(tsuj, suj),
                origem_tipo=tsuj,
                origem_nome=suj,
                relacao=rel,
                destino_id=_no_id(tobj, obj),
                destino_tipo=tobj,
                destino_nome=obj,
                metodo=metodo,
            )
            gravadas += 1
        return gravadas

    def ingerir_texto(
        self,
        transcricao: str,
        pessoa_id: str,
        pessoa_nome: str,
        modo: str = "local",
        api_key: str = "",
        model: str = "gemini-3.5-flash-lite",
        session: Optional[requests.Session] = None,
    ) -> Dict[str, Any]:
        """
        Extrai e grava triplas no grafo usando `local`, `llm` ou `hibrido`.
        """
        modo_norm = (modo or "local").strip().lower()
        triplas: List[Dict[str, str]] = []
        telemetria: Dict[str, Any] = {
            "metodo": modo_norm,
            "latencia_ms": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        if modo_norm == "llm":
            triplas, telemetria = self.extrair_triplas_llm(
                transcricao=transcricao,
                pessoa_id=pessoa_id,
                pessoa_nome=pessoa_nome,
                api_key=api_key,
                model=model,
                session=session,
            )
            gravadas = self._gravar_triplas(triplas, metodo="llm")
            return {"triplas": triplas, "gravadas": gravadas, **telemetria}

        # local ou hibrido
        triplas, tel_local = self.extrair_triplas_local(
            transcricao=transcricao,
            pessoa_id=pessoa_id,
            pessoa_nome=pessoa_nome,
        )
        telemetria = dict(tel_local)
        metodo_final = "local"

        if modo_norm == "hibrido" and not triplas and self._tem_gatilho_complexo_memoria(transcricao):
            triplas_llm, tel_llm = self.extrair_triplas_llm(
                transcricao=transcricao,
                pessoa_id=pessoa_id,
                pessoa_nome=pessoa_nome,
                api_key=api_key,
                model=model,
                session=session,
            )
            triplas = triplas_llm
            metodo_final = "llm"
            telemetria = {
                **tel_llm,
                "metodo": "hibrido(llm)",
                "latencia_ms": round(tel_local["latencia_ms"] + tel_llm["latencia_ms"], 2),
            }

        gravadas = self._gravar_triplas(triplas, metodo=metodo_final)
        return {"triplas": triplas, "gravadas": gravadas, **telemetria}
