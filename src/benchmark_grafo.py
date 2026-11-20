#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de Eficiência vs. Custo — Extração para Grafo de Memória (`PETeleco`).

Compara os modos `local` (regex PT-BR 0 tokens), `hibrido` e `llm` (Gemini Flash-Lite JSON)
em 12 frases reais do cotidiano do PET Elétrica, medindo Precisão, Recall, F1,
latência por turno e gasto de tokens da cota Free.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.grafo_memoria import GrafoMemoriaSQLite, _norm
from src.memoria_pessoas import carregar_env


@dataclass(frozen=True)
class CasoBenchmarkGrafo:
    id: str
    pessoa_id: str
    pessoa_nome: str
    fala: str
    esperadas: Tuple[Tuple[str, str, str], ...]  # (sujeito, relacao, objeto)


CASOS_BENCHMARK_GRAFO: Tuple[CasoBenchmarkGrafo, ...] = (
    CasoBenchmarkGrafo(
        "obj_guardado",
        "caio",
        "Caio",
        "Guardei o multímetro no armário 2.",
        (("multimetro", "esta_em", "armario 2"),),
    ),
    CasoBenchmarkGrafo(
        "obj_estado",
        "heitor",
        "Heitor",
        "A fonte de bancada tá na mesa do fundo.",
        (("fonte de bancada", "esta_em", "mesa do fundo"),),
    ),
    CasoBenchmarkGrafo(
        "estudo_materia",
        "heitor",
        "Heitor",
        "Tô fazendo Eletromagnetismo esse período.",
        (("heitor", "estuda", "eletromagnetismo"),),
    ),
    CasoBenchmarkGrafo(
        "estudo_prova",
        "caio",
        "Caio",
        "Tenho prova de Cálculo 3 amanhã cedo.",
        (("caio", "estuda", "calculo 3"),),
    ),
    CasoBenchmarkGrafo(
        "projeto_atual",
        "caio",
        "Caio",
        "Tô mexendo na placa V2 do Rover.",
        (("caio", "trabalha_em", "placa v2 do rover"),),
    ),
    CasoBenchmarkGrafo(
        "gosto_energetico",
        "heitor",
        "Heitor",
        "Eu vivo à base de Monster antes da prova.",
        (("heitor", "gosta_de", "monster antes da prova"),),
    ),
    CasoBenchmarkGrafo(
        "gosto_musica",
        "luigi",
        "Luigi",
        "Eu curto rock alternativo e lo-fi pra programar.",
        (("luigi", "gosta_de", "rock alternativo e lo fi pra programar"),),
    ),
    CasoBenchmarkGrafo(
        "fato_explicito",
        "caio",
        "Caio",
        "Lembra que o sensor DHT22 fica na gaveta azul.",
        (("sensor dht22", "fato", "gaveta azul"),),
    ),
    CasoBenchmarkGrafo(
        "complexo_mora",
        "nathan",
        "Nathan",
        "Na verdade eu moro em Icaraí e só pego o ônibus pro Gragoatá.",
        (("nathan", "mora_em", "icarai"),),
    ),
    CasoBenchmarkGrafo(
        "pergunta_nao_grava",
        "caio",
        "Caio",
        "Como tá o clima hoje em Niterói?",
        (),
    ),
    CasoBenchmarkGrafo(
        "saudacao_nao_grava",
        "manuel",
        "Manuel",
        "Fala PETeleco, beleza?",
        (),
    ),
    CasoBenchmarkGrafo(
        "zoeira_sem_fato",
        "jhonatan",
        "Jhonatan",
        "Dormi três horas hoje, tô só o pó da rabiola.",
        (),
    ),
)


def _limpar_cmp(texto: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", _norm(texto))).strip()


def _match_tripla(pred: Dict[str, str], esp: Tuple[str, str, str]) -> bool:
    ps = _limpar_cmp(pred.get("sujeito", ""))
    pr = _limpar_cmp(pred.get("relacao", ""))
    po = _limpar_cmp(pred.get("objeto", ""))
    es, er, eo = (_limpar_cmp(esp[0]), _limpar_cmp(esp[1]), _limpar_cmp(esp[2]))

    suj_ok = es in ps or ps in es
    obj_ok = eo in po or po in eo
    if not (suj_ok and obj_ok):
        return False
    if er == pr:
        return True
    # Equivalência semântica aceita entre fato e esta_em
    return {er, pr} <= {"fato", "esta_em"}


def avaliar_modo_grafo(
    modo: str,
    api_key: str = "",
    model: str = "gemini-3.5-flash-lite",
) -> Dict[str, object]:
    tp = 0
    fp = 0
    fn = 0
    total_lat_ms = 0.0
    total_tokens = 0
    detalhes: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmp:
        grafo = GrafoMemoriaSQLite(sqlite_path=Path(tmp) / "bench_grafo.sqlite")
        for caso in CASOS_BENCHMARK_GRAFO:
            res = grafo.ingerir_texto(
                transcricao=caso.fala,
                pessoa_id=caso.pessoa_id,
                pessoa_nome=caso.pessoa_nome,
                modo=modo,
                api_key=api_key,
                model=model,
            )
            preds: List[Dict[str, str]] = list(res.get("triplas") or [])
            total_lat_ms += float(res.get("latencia_ms") or 0.0)
            total_tokens += int(res.get("total_tokens") or 0)

            casos_esp = list(caso.esperadas)
            usados_pred = set()
            caso_tp = 0
            for esp in casos_esp:
                achou = False
                for idx, p in enumerate(preds):
                    if idx not in usados_pred and _match_tripla(p, esp):
                        usados_pred.add(idx)
                        achou = True
                        break
                if achou:
                    caso_tp += 1
            caso_fp = max(0, len(preds) - len(usados_pred))
            caso_fn = max(0, len(casos_esp) - caso_tp)
            tp += caso_tp
            fp += caso_fp
            fn += caso_fn
            detalhes.append(
                {
                    "id": caso.id,
                    "tp": caso_tp,
                    "fp": caso_fp,
                    "fn": caso_fn,
                    "lat_ms": res.get("latencia_ms", 0.0),
                    "tokens": res.get("total_tokens", 0),
                }
            )

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    n = len(CASOS_BENCHMARK_GRAFO)
    return {
        "modo": modo,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precisao": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "lat_media_ms": round(total_lat_ms / max(1, n), 2),
        "tokens_total": total_tokens,
        "detalhes": detalhes,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Grafo SQLite (local vs hibrido vs llm)")
    parser.add_argument(
        "--modo",
        choices=("local", "hibrido", "llm", "todos"),
        default="local",
        help="Modo de extração a avaliar (padrão: local)",
    )
    parser.add_argument("--model", default="gemini-3.5-flash-lite", help="Modelo Gemini caso rode llm/hibrido")
    args = parser.parse_args(argv)

    env = carregar_env()
    api_key = env.get("GEMINI_API_KEY", "").strip()
    modos = ("local", "hibrido", "llm") if args.modo == "todos" else (args.modo,)

    print("=== BENCHMARK DE EXTRAÇÃO DO GRAFO DE MEMÓRIA (12 CASOS) ===")
    print(f"{'Modo':10s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'Lat média':10s} | {'Tokens':8s} | TP/FP/FN")
    print("-" * 74)
    for m in modos:
        if m in ("hibrido", "llm") and not api_key:
            print(f"{m:10s} | (ignorado: GEMINI_API_KEY ausente no .env)")
            continue
        r = avaliar_modo_grafo(m, api_key=api_key, model=args.model)
        print(
            f"{str(r['modo']):10s} | {r['precisao']:6.3f} | {r['recall']:6.3f} | {r['f1']:6.3f} | "
            f"{r['lat_media_ms']:7.2f} ms | {r['tokens_total']:8d} | {r['tp']}/{r['fp']}/{r['fn']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
