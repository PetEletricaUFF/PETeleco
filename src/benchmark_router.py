#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark isolado do Roteador SystemOne (`tev1:0.8b` via Ollama /v1/systemone).

Mede sem microfone nem robo fisico:
  - Latencia cold-start vs warm (min, p50, p95, max em ms)
  - Uso de VRAM antes e depois via `nvidia-smi` e `ollama ps`
  - Acuracia das rotas esperadas (ignorar, fala_pronta, fala_parametrizada, acao_rover, cognicao_gemini)
  - Taxa de economia local (chamadas resolvidas localmente sem API Gemini)
"""

import argparse
from dataclasses import dataclass
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional

try:
    from src.router_systemone import ContextoRover, DEFAULT_MODEL, DEFAULT_OLLAMA_URL, RoteadorSystemOne
except ImportError:
    from router_systemone import ContextoRover, DEFAULT_MODEL, DEFAULT_OLLAMA_URL, RoteadorSystemOne


@dataclass
class CasoTeste:
    """Caso rotulado de benchmark para o roteador cognitivo."""

    id_caso: str
    transcricao: str
    estado_rover: str
    pessoa_sface: str
    objetos_yolo: str
    rota_esperada: str
    subtipo_esperado: Optional[str] = None
    espera_imagem: Optional[bool] = None


CASOS_BENCHMARK: List[CasoTeste] = [
    # 1. Pool de falas prontas (0 ms TTS / R$ 0)
    CasoTeste("social_oi", "Oi robo, tudo bem por ai?", "STANDBY", "Caio (94%)", "person", "pool_fala_pronta", "saudacao", False),
    CasoTeste("social_bom_dia", "Bom dia meu amigo!", "STANDBY", "Caio (91%)", "person", "pool_fala_pronta", "saudacao", False),
    CasoTeste("social_tchau", "Tchau robo, ate amanha.", "STANDBY", "Caio (92%)", "person", "pool_fala_pronta", "despedida", False),
    CasoTeste("social_ok", "Beleza, pode confirmar, entendi.", "AGUARDANDO_CONFIRMACAO", "Caio (93%)", "person", "pool_fala_pronta", "ok_confirmacao", False),
    CasoTeste("social_obrigado", "Valeu, mandou muito bem!", "STANDBY", "Caio (95%)", "person", "pool_fala_pronta", "elogio_agradecimento", False),

    # 2. Funcoes deterministicas locais + Piper TTS (R$ 0)
    CasoTeste("local_hora", "Que horas sao agora?", "STANDBY", "Caio (94%)", "person", "fala_parametrizada", "hora_atual", False),
    CasoTeste("local_clima", "Vai chover hoje? Como ta o tempo?", "STANDBY", "Caio (94%)", "person", "fala_parametrizada", "previsao_clima", False),
    CasoTeste("local_bateria", "Como esta a sua bateria e temperatura?", "STANDBY", "Caio (90%)", "person", "fala_parametrizada", "bateria_temp", False),
    CasoTeste("local_quem", "Voce sabe quem sou eu na sua frente?", "STANDBY", "Maria (88%)", "person", "fala_parametrizada", "quem_vejo_agora", False),

    # 3. Comandos fisicos do rover (trava de seguranca motora)
    CasoTeste("acao_seguir", "Robo, me segue agora.", "STANDBY", "Caio (94%)", "person", "acao_rover", "seguir_rosto", False),
    CasoTeste("acao_ir_objeto", "Vai ate aquela garrafa ali na frente.", "STANDBY", "Caio (94%)", "person, bottle", "acao_rover", "ir_ate_objeto", None),
    CasoTeste("acao_parar", "Para tudo agora! Freia!", "APROXIMA", "Caio (92%)", "person, bottle", "acao_rover", "parar", False),
    CasoTeste("acao_radar", "Comeca a procurar uma garrafa pela sala, ativa o radar.", "STANDBY", "desconhecido", "chair", "acao_rover", "radar", False),

    # 4. Conversa paralela entre terceiros / ruido (deve ignorar)
    CasoTeste("ruido_terceiros_1", "Fulano, voce ja almocou hoje ou vai pedir marmita?", "STANDBY", "desconhecido", "person, person", "ignorar", None, False),
    CasoTeste("ruido_terceiros_2", "Passa o carregador do celular pra mim por favor, Joao.", "STANDBY", "desconhecido", "person, cell phone", "ignorar", None, False),
    CasoTeste("ruido_tv", "E o placar final ficou tres a um no segundo tempo.", "STANDBY", "nenhuma", "tv", "ignorar", None, False),

    # 5. Cognicao pesada (Gemini API com ou sem imagem da camera)
    CasoTeste("gemini_visao_1", "O que voce esta vendo na mesa na sua frente agora?", "STANDBY", "Caio (94%)", "person, cup, bottle, laptop", "cognicao_gemini", "descrever_cena", True),
    CasoTeste("gemini_visao_2", "Le o que esta escrito nesse papel aqui.", "STANDBY", "Caio (93%)", "person, book", "cognicao_gemini", "descrever_cena", True),
    CasoTeste("gemini_aberto_1", "Me explica de forma simples como funciona um filtro de Kalman em robotica.", "STANDBY", "Caio (94%)", "person", "cognicao_gemini", "conversa_aberta", False),
    CasoTeste("gemini_aberto_2", "Qual a diferenca pratica entre controlador PID e controle por espaco de estados?", "STANDBY", "Caio (94%)", "person", "cognicao_gemini", "conversa_aberta", False),
]


CASOS_GENERALIZACAO: List[CasoTeste] = [
    # Casos ineditos (out-of-sample, sem palavras-chave dos criterios do prompt)
    CasoTeste("gen_social_1", "E ai parceiro, tudo tranquilo?", "STANDBY", "Caio (94%)", "person", "pool_fala_pronta", "saudacao", False),
    CasoTeste("gen_social_2", "Ate mais tarde, vou sair agora.", "STANDBY", "Caio (92%)", "person", "pool_fala_pronta", "despedida", False),
    CasoTeste("gen_social_3", "Fechou, pode mandar ver.", "AGUARDANDO_CONFIRMACAO", "Caio (93%)", "person", "pool_fala_pronta", "ok_confirmacao", False),
    CasoTeste("gen_social_4", "Show de bola, curti demais.", "STANDBY", "Caio (95%)", "person", "pool_fala_pronta", "elogio_agradecimento", False),
    CasoTeste("gen_local_1", "Me diz a data e o horario de hoje.", "STANDBY", "Caio (94%)", "person", "fala_parametrizada", "hora_atual", False),
    CasoTeste("gen_local_2", "Ta muito frio la fora ou vai abrir sol?", "STANDBY", "Caio (94%)", "person", "fala_parametrizada", "previsao_clima", False),
    CasoTeste("gen_local_3", "Quantos volts ainda tem no pack do chassi e quanto ta esquentando?", "STANDBY", "Caio (90%)", "person", "fala_parametrizada", "bateria_temp", False),
    CasoTeste("gen_local_4", "Qual o nome dessa pessoa parada na sua frente?", "STANDBY", "Maria (88%)", "person", "fala_parametrizada", "quem_vejo_agora", False),
    CasoTeste("gen_acao_1", "Acompanha a Maria enquanto ela anda pela sala.", "STANDBY", "Maria (88%)", "person", "acao_rover", "seguir_rosto", False),
    CasoTeste("gen_acao_2", "Chega perto daquela cadeira verde ali.", "STANDBY", "Caio (94%)", "person, chair", "acao_rover", "ir_ate_objeto", None),
    CasoTeste("gen_acao_3", "Corta os motores imediatamente!", "APROXIMA", "Caio (92%)", "person, chair", "acao_rover", "parar", False),
    CasoTeste("gen_acao_4", "Gira a cabeca devagar pra achar algum copo perdido no chao.", "STANDBY", "desconhecido", "chair", "acao_rover", "radar", False),
    CasoTeste("gen_ruido_1", "Lucas, pega uma garrafa de agua na geladeira pra mim por favor?", "STANDBY", "desconhecido", "person, bottle", "ignorar", None, False),
    CasoTeste("gen_ruido_2", "O dolar subiu dois por cento no fechamento da bolsa hoje.", "STANDBY", "nenhuma", "tv", "ignorar", None, False),
    CasoTeste("gen_gemini_1", "Tem algum cabo desconectado em cima dessa bancada?", "STANDBY", "Caio (94%)", "person, laptop", "cognicao_gemini", "descrever_cena", True),
    CasoTeste("gen_gemini_2", "Por que o ceu fica alaranjado no por do sol?", "STANDBY", "Caio (94%)", "person", "cognicao_gemini", "conversa_aberta", False),
]


def consultar_vram_nvidia() -> Optional[Dict[str, int]]:
    """Le VRAM usada, livre e total em MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        primeira_linha = out.splitlines()[0]
        usada, livre, total = [int(x.strip()) for x in primeira_linha.split(",")]
        return {"usada_mib": usada, "livre_mib": livre, "total_mib": total}
    except Exception:
        return None


def consultar_ollama_ps(base_url: str) -> List[Dict[str, object]]:
    """Consulta modelos carregados e VRAM reportada pelo proprio Ollama (`/api/ps`)."""
    try:
        import requests

        resp = requests.get(f"{base_url.rstrip('/')}/api/ps", timeout=3.0)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception:
        return []


def percentil(valores: List[float], p: float) -> float:
    """Calcula percentil linear simples em lista ja ordenada."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    k = (len(ordenados) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordenados) - 1)
    if f == c:
        return ordenados[f]
    return ordenados[f] + (ordenados[c] - ordenados[f]) * (k - f)


def executar_benchmark(
    base_url: str,
    model: str,
    rodadas: int,
    salvar_json: Optional[str],
    suite: str = "todas",
) -> int:
    """Executa warmup/cold-start, bateria rotulada e imprime relatorio consolidado."""
    if suite == "padrao":
        casos_ativos = CASOS_BENCHMARK
    elif suite == "generalizacao":
        casos_ativos = CASOS_GENERALIZACAO
    else:
        casos_ativos = CASOS_BENCHMARK + CASOS_GENERALIZACAO

    roteador = RoteadorSystemOne(base_url=base_url, model=model)

    try:
        versao = roteador.verificar_servidor()
    except Exception as exc:
        print(f"[ERRO] Nao foi possivel conectar ao Ollama em {base_url}: {exc}")
        return 2

    print(f"=== BENCHMARK ROTEADOR SYSTEMONE ({model}) ===")
    print(f"Ollama URL: {base_url} | Versao servidor: {versao}")

    vram_antes = consultar_vram_nvidia()
    if vram_antes:
        print(
            f"VRAM antes: {vram_antes['usada_mib']} MiB usada / "
            f"{vram_antes['livre_mib']} MiB livre (total {vram_antes['total_mib']} MiB)"
        )

    # Medicao separada do cold-start (primeira inferencia carregando pesos)
    primeiro_ctx = ContextoRover(
        transcricao="Oi robo, tudo bem?",
        estado_rover="STANDBY",
        pessoa_sface="Caio (94%)",
        objetos_yolo="person",
    )
    cold = roteador.decidir(primeiro_ctx, timeout_override_s=120.0)
    print(f"Cold-start (1a chamada): {cold.latencia_ms:.1f} ms -> {cold.rota_final}:{cold.subtipo}")

    vram_depois = consultar_vram_nvidia()
    ollama_ps = consultar_ollama_ps(base_url)
    if vram_depois and vram_antes:
        delta_vram = vram_depois["usada_mib"] - vram_antes["usada_mib"]
        print(
            f"VRAM apos carga: {vram_depois['usada_mib']} MiB usada "
            f"(delta observado: {delta_vram:+d} MiB) | Livre: {vram_depois['livre_mib']} MiB"
        )
    for m in ollama_ps:
        size_vram_mb = float(m.get("size_vram", 0)) / (1024 * 1024)
        size_mb = float(m.get("size", 0)) / (1024 * 1024)
        print(f"ollama ps -> {m.get('name')}: size={size_mb:.1f} MiB | size_vram={size_vram_mb:.1f} MiB")

    print("\n--- Resultados por Caso (Ultima Rodada) ---")
    latencias_warm: List[float] = []
    acertos_rota = 0
    acertos_subtipo = 0
    total_subtipo_verificado = 0
    acertos_imagem = 0
    total_imagem_verificada = 0
    resolvidos_localmente = 0
    registros_json: List[Dict[str, object]] = []

    for r in range(rodadas):
        for caso in casos_ativos:
            ctx = ContextoRover(
                transcricao=caso.transcricao,
                estado_rover=caso.estado_rover,
                pessoa_sface=caso.pessoa_sface,
                objetos_yolo=caso.objetos_yolo,
            )
            dec = roteador.decidir(ctx)
            latencias_warm.append(dec.latencia_ms)

            if r == rodadas - 1:
                ok_rota = dec.rota_final == caso.rota_esperada
                acertos_rota += int(ok_rota)

                ok_sub = True
                if caso.subtipo_esperado is not None:
                    total_subtipo_verificado += 1
                    ok_sub = dec.subtipo == caso.subtipo_esperado
                    acertos_subtipo += int(ok_sub)

                ok_img = True
                if caso.espera_imagem is not None:
                    total_imagem_verificada += 1
                    ok_img = dec.precisa_imagem_camera == caso.espera_imagem
                    acertos_imagem += int(ok_img)

                if dec.rota_final != "cognicao_gemini":
                    resolvidos_localmente += 1

                status = "OK" if (ok_rota and ok_sub and ok_img) else "FALHA"
                trava = " [CONFIRMAR]" if dec.requer_confirmacao else ""
                img_tag = f" img={int(dec.precisa_imagem_camera)}({dec.prob_imagem_camera:.2f})"
                print(
                    f"[{status:5s}] {caso.id_caso:18s} | {dec.latencia_ms:6.1f} ms | "
                    f"dirig={dec.dirigido_ao_robo:.2f} | "
                    f"pred={dec.rota_final}:{dec.subtipo} ({dec.confianca:.2f}){trava}{img_tag} | "
                    f"esp={caso.rota_esperada}:{caso.subtipo_esperado}"
                )

                registros_json.append(
                    {
                        "id": caso.id_caso,
                        "transcricao": caso.transcricao,
                        "esperado": {
                            "rota": caso.rota_esperada,
                            "subtipo": caso.subtipo_esperado,
                            "imagem": caso.espera_imagem,
                        },
                        "predito": {
                            "rota": dec.rota_final,
                            "subtipo": dec.subtipo,
                            "confianca": round(dec.confianca, 4),
                            "dirigido_ao_robo": round(dec.dirigido_ao_robo, 4),
                            "requer_confirmacao": dec.requer_confirmacao,
                            "precisa_imagem_camera": dec.precisa_imagem_camera,
                            "prob_imagem_camera": round(dec.prob_imagem_camera, 4),
                            "latencia_ms": round(dec.latencia_ms, 2),
                        },
                        "ok_rota": ok_rota,
                        "ok_subtipo": ok_sub,
                        "ok_imagem": ok_img,
                        "respostas_brutas": dec.respostas_brutas,
                    }
                )

    total = len(casos_ativos)
    resumo = {
        "modelo": model,
        "ollama_versao": versao,
        "cold_start_ms": round(cold.latencia_ms, 2),
        "rodadas": rodadas,
        "total_chamadas_warm": len(latencias_warm),
        "latencia_ms": {
            "min": round(min(latencias_warm), 2),
            "media": round(statistics.mean(latencias_warm), 2),
            "p50": round(percentil(latencias_warm, 50), 2),
            "p95": round(percentil(latencias_warm, 95), 2),
            "max": round(max(latencias_warm), 2),
        },
        "vram_antes": vram_antes,
        "vram_depois": vram_depois,
        "ollama_ps": ollama_ps,
        "acuracia_rota": f"{acertos_rota}/{total} ({(acertos_rota / total) * 100:.1f}%)",
        "acuracia_subtipo": (
            f"{acertos_subtipo}/{total_subtipo_verificado} "
            f"({(acertos_subtipo / total_subtipo_verificado) * 100:.1f}%)"
        ),
        "acuracia_imagem": (
            f"{acertos_imagem}/{total_imagem_verificada} "
            f"({(acertos_imagem / total_imagem_verificada) * 100:.1f}%)"
        ),
        "economia_nuvem_local": f"{resolvidos_localmente}/{total} ({(resolvidos_localmente / total) * 100:.1f}%)",
    }

    print("\n--- Resumo Estatistico ---")
    print(
        f"Latencia Warm (n={len(latencias_warm)}): "
        f"min={resumo['latencia_ms']['min']} ms | "
        f"media={resumo['latencia_ms']['media']} ms | "
        f"p50={resumo['latencia_ms']['p50']} ms | "
        f"p95={resumo['latencia_ms']['p95']} ms | "
        f"max={resumo['latencia_ms']['max']} ms"
    )
    print(f"Acuracia Rota:    {resumo['acuracia_rota']}")
    print(f"Acuracia Subtipo: {resumo['acuracia_subtipo']}")
    print(f"Acuracia Imagem:  {resumo['acuracia_imagem']}")
    print(f"Resolvido Local:  {resumo['economia_nuvem_local']} (sem chamar Gemini)")

    if salvar_json:
        os.makedirs(os.path.dirname(os.path.abspath(salvar_json)), exist_ok=True)
        with open(salvar_json, "w", encoding="utf-8") as f:
            json.dump({"resumo": resumo, "casos": registros_json}, f, ensure_ascii=False, indent=2)
        print(f"Relatorio salvo em: {salvar_json}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark isolado do Roteador SystemOne (Ollama).")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="URL base do servidor Ollama.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo SystemOne (ex: tev1:0.8b).")
    parser.add_argument("--rodadas", type=int, default=2, help="Quantidade de rodadas sobre a suite escolhida.")
    parser.add_argument(
        "--suite",
        choices=["padrao", "generalizacao", "todas"],
        default="todas",
        help="Conjunto de frases: padrao (20), generalizacao inedita (16) ou todas (36).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Caminho opcional para salvar o relatorio completo em JSON (ex: results/benchmark_systemone.json).",
    )
    args = parser.parse_args()
    return executar_benchmark(args.url, args.model, max(1, args.rodadas), args.json_out, suite=args.suite)


if __name__ == "__main__":
    sys.exit(main())
