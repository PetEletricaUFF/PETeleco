#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes automatizados para métricas, variantes da Arena multi-chat e modo DESGRACADO."""

from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import requests

from src.arena_prompts import (
    ArenaPromptsApp,
    extrair_pares_turnos,
    iniciar_servidor_arena,
)
from src.escuta_stt import EscutaSTT
from src.memoria_pessoas import GerenciadorMemoriaPessoas
from src.persona_peteleco import (
    DEFAULT_PROMPT_PATH,
    MotorPersonaPETeleco,
    carregar_system_prompt_base,
    normalizar_para_tts,
)
from src.prompt_metrics import (
    CENARIOS_BENCHMARK_ARENA,
    VARIANTES_ARENA,
    avaliar_conversa_metricas,
)


class ArenaAndMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="peteleco_arena_test_"))
        self.prompt_copy = self.tmpdir / "persona_peteleco.md"
        shutil.copy2(DEFAULT_PROMPT_PATH, self.prompt_copy)
        self.perfis_copy = self.tmpdir / "perfis_pessoas.json"
        shutil.copy2(Path("data/perfis_pessoas.json"), self.perfis_copy)
        self.sqlite_tmp = self.tmpdir / "memoria_longa.sqlite"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_metricas_penalizam_cliches_sabia_que_e_quebra_formal(self) -> None:
        conversa_ruim = [
            {"user": "Fala peteleco", "assistant": "Fala chefe! Tô com as quatro rodas firmes no chão da Praia Vermelha."},
            {"user": "tenho prova", "assistant": "Caraca, merece patrimônio histórico! Sabia que vai precisar? Boa sorte ou vai chorar?"},
            {"user": "valeu", "assistant": "Caraca, tamo junto chefe! Sabia que o professor é lenda viva? Vai desistir ou tentar?"},
        ]
        met_ruim = avaliar_conversa_metricas(conversa_ruim, modo_efetivo="NORMAL", pessoa_id="caio")
        self.assertLess(met_ruim["anti_cliche_score"], 50.0)
        self.assertEqual(met_ruim["chefe_count"], 2)
        self.assertEqual(met_ruim["sabia_que_count"], 2)
        self.assertEqual(met_ruim["binary_question_count"], 2)
        self.assertTrue(len(met_ruim["penalties"]) >= 5)

        conversa_seria_suja = [
            {"user": "Bom dia", "assistant": "Bom dia chefe, essa prova é sacanagem demais."}
        ]
        met_seria = avaliar_conversa_metricas(conversa_seria_suja, modo_efetivo="SERIO", pessoa_id="felipe_sass")
        self.assertFalse(met_seria["formal_safety_ok"])
        self.assertLess(met_seria["overall_heuristic"], 40.0)

        conversa_boa = [
            {"user": "Fala peteleco", "assistant": "E aí Caio, qual foi a missão caótica de hoje?"},
            {"user": "tenho prova de calculo 3", "assistant": "Cálculo três cobra caro mesmo, lembra que até o teorema de Stokes veio de carta alheia."},
            {"user": "valeu", "assistant": "Se a integral não fechar, culpa as coordenadas esféricas e segue firme."},
        ]
        met_boa = avaliar_conversa_metricas(conversa_boa, modo_efetivo="NORMAL", pessoa_id="caio")
        self.assertTrue(met_boa["formal_safety_ok"])
        self.assertGreaterEqual(met_boa["overall_heuristic"], 90.0)
        self.assertEqual(met_boa["penalties"], [])

    def test_variantes_arena_e_system_prompt_atual_v4(self) -> None:
        self.assertEqual(len(VARIANTES_ARENA), 5)
        ids = [v["id"] for v in VARIANTES_ARENA]
        self.assertEqual(len(ids), len(set(ids)))
        for v in VARIANTES_ARENA:
            self.assertIn("DESGRACADO", v["prompt"])
            self.assertIn("SERIO", v["prompt"])
        self.assertTrue(len(CENARIOS_BENCHMARK_ARENA) >= 4)

        base = carregar_system_prompt_base(DEFAULT_PROMPT_PATH)
        self.assertIn("DESGRACADO", base)
        self.assertIn("NORMAL", base)
        self.assertIn("SERIO", base)
        self.assertIn("professor_orientador", base)
        self.assertNotIn("=== CONTEXTO DO TURNO", base)
        self.assertLess(len(base), 5500)

    def test_interceptadores_modo_desgracado_e_saida_limpa(self) -> None:
        mem = GerenciadorMemoriaPessoas(perfis_path=self.perfis_copy, sqlite_path=self.sqlite_tmp)
        motor = MotorPersonaPETeleco(memoria=mem, prompt_path=self.prompt_copy)

        resp1, origem1 = motor._verificar_interceptadores_fisicos_e_modo(
            "entra no modo desgracado", mem.obter_perfil("caio")
        )
        self.assertEqual(motor.modo_persona, "DESGRACADO")
        self.assertEqual(origem1, "local:troca_modo_desgracado")
        self.assertTrue(resp1)

        # Substring "modo desgracado" dentro de "sai do modo desgracado" nao pode reativar DESGRACADO
        resp2, origem2 = motor._verificar_interceptadores_fisicos_e_modo(
            "sai do modo desgracado", mem.obter_perfil("caio")
        )
        self.assertEqual(motor.modo_persona, "NORMAL")
        self.assertEqual(origem2, "local:troca_modo_normal")
        self.assertTrue(resp2)

        # Professor orientador trava automaticamente em SERIO
        resp_prof, origem_prof = motor._verificar_interceptadores_fisicos_e_modo(
            "entra no modo desgracado", mem.obter_perfil("felipe_sass")
        )
        self.assertEqual(motor.modo_persona, "SERIO")
        self.assertEqual(origem_prof, "local:troca_modo_serio")
        self.assertIn("Professor Felipe", resp_prof.replace("professor Felipe", "Professor Felipe"))

    def test_extrair_pares_e_rotas_http_arena(self) -> None:
        pares = extrair_pares_turnos(
            [
                {"role": "user", "text": "Oi"},
                {"role": "assistant", "text": "Fala!"},
                {"role": "user", "text": "Tudo certo?"},
            ],
            resposta_extra="Tudo ótimo.",
        )
        self.assertEqual(
            pares,
            [
                {"user": "Oi", "assistant": "Fala!"},
                {"user": "Tudo certo?", "assistant": "Tudo ótimo."},
            ],
        )

        app = ArenaPromptsApp(
            prompt_path=self.prompt_copy,
            perfis_path=self.perfis_copy,
            pacing_seconds=0.0,
        )
        app.studio._env["GEMINI_API_KEY"] = "SECRET_ARENA_KEY_999"

        with patch("src.studio_prompt.listar_modelos_ollama", return_value=(True, ["gemma3:4b"])):
            srv, port = iniciar_servidor_arena(app, porta_inicial=0)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            base_url = f"http://127.0.0.1:{port}"
            try:
                r_html = requests.get(f"{base_url}/", timeout=3)
                self.assertEqual(r_html.status_code, 200)
                self.assertIn("Arena Multi-Chat", r_html.text)

                r_state = requests.get(f"{base_url}/api/state", timeout=3)
                self.assertEqual(r_state.status_code, 200)
                st = r_state.json()
                self.assertEqual(len(st["variantes"]), 6)  # atual_md + 5 variantes
                self.assertTrue(len(st["cenarios"]) >= 4)

                fake_gem = MagicMock()
                fake_gem.status_code = 200
                fake_gem.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "Resposta teste da Arena sem clichês."}]}}],
                    "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 8, "totalTokenCount": 98},
                }
                with patch.object(app.studio._session, "post", return_value=fake_gem):
                    r_chat = requests.post(
                        f"{base_url}/api/chat",
                        json={
                            "provider": "gemini",
                            "model": "gemini-3.5-flash-lite",
                            "prompt_base": VARIANTES_ARENA[0]["prompt"],
                            "contexto": {"pessoa_id": "caio", "modo_persona": "DESGRACADO"},
                            "messages": [{"role": "user", "text": "Fala peteleco"}],
                        },
                        timeout=3,
                    )
                    self.assertEqual(r_chat.status_code, 200)
                    chat_data = r_chat.json()
                    self.assertTrue(chat_data["ok"])
                    self.assertEqual(chat_data["modo_efetivo"], "DESGRACADO")
                    self.assertIn("metricas", chat_data)
                    self.assertGreaterEqual(chat_data["metricas"]["anti_cliche_score"], 90.0)

                    r_batch = requests.post(
                        f"{base_url}/api/chat_batch",
                        json={
                            "provider": "gemini",
                            "model": "gemini-3.5-flash-lite",
                            "contexto": {"pessoa_id": "caio", "modo_persona": "NORMAL"},
                            "variantes": [
                                {"variant_id": "a", "prompt_base": "P1", "messages": [{"role": "user", "text": "Oi"}]},
                                {"variant_id": "b", "prompt_base": "P2", "messages": [{"role": "user", "text": "Oi"}]},
                            ],
                        },
                        timeout=3,
                    )
                    self.assertEqual(r_batch.status_code, 200)
                    self.assertEqual(len(r_batch.json()["resultados"]), 2)

                r_prom = requests.post(
                    f"{base_url}/api/promote",
                    json={"prompt_base": "Prompt vencedor promovido pela Arena."},
                    timeout=3,
                )
                self.assertEqual(r_prom.status_code, 200)
                self.assertEqual(
                    carregar_system_prompt_base(self.prompt_copy),
                    "Prompt vencedor promovido pela Arena.",
                )
            finally:
                srv.shutdown()
                srv.server_close()

    def test_normalizar_para_tts_unidades_eletronicas_e_moeda(self) -> None:
        bruto = (
            "Liga 4k7 no 3V3, usa 22k com 82nF pra 20kHz, "
            "470uF nos 5A, 15mA no LED, buffer so_rcvbuf no ADC GPIO I2C UDP UART por R$1,99."
        )
        tts = normalizar_para_tts(bruto)
        self.assertIn("4 vírgula 7 quilo-ohms", tts)
        self.assertIn("3 vírgula 3 volts", tts)
        self.assertIn("22 quilo-ohms", tts)
        self.assertIn("82 nanofarads", tts)
        self.assertIn("20 quilohertz", tts)
        self.assertIn("470 microfarads", tts)
        self.assertIn("5 amperes", tts)
        self.assertIn("15 miliamperes", tts)
        self.assertIn("so rcvbuf", tts)
        self.assertIn("A Dê Cê", tts)
        self.assertIn("Gê Pê I Ó", tts)
        self.assertIn("I Dois Cê", tts)
        self.assertIn("U Dê Pê", tts)
        self.assertIn("Ú art", tts)
        self.assertIn("1 real e 99", tts)
        self.assertNotIn("_", tts)

    def test_memoria_limpa_sem_bordoes_e_stt_vocab_bancada(self) -> None:
        resumo = GerenciadorMemoriaPessoas._resumir_episodio_local(
            "Medi 14 ohms na linha de alimentacao",
            "Fala chefe, tamo junto aqui na bancada!",
        )
        self.assertTrue(resumo.startswith("Assunto: "))
        self.assertNotIn("PETeleco:", resumo)
        self.assertNotIn("chefe", resumo.lower())

        prompt_atual = carregar_system_prompt_base(self.prompt_copy)
        self.assertIn("Jarvis", prompt_atual)
        self.assertIn("TELEMETRIA", prompt_atual)
        self.assertIn("apenas_voz", prompt_atual)

        stt = EscutaSTT()
        vocab = stt.construir_initial_prompt()
        for termo in ("ESP32", "I2C", "DS18B20", "KiCad", "Santisteban"):
            self.assertIn(termo, vocab)

    def test_motor_persona_pool_desgracado_e_trava_professor(self) -> None:
        mem = GerenciadorMemoriaPessoas(
            perfis_path=self.perfis_copy,
            sqlite_path=self.sqlite_tmp,
        )
        motor = MotorPersonaPETeleco(memoria=mem, prompt_path=self.prompt_copy)
        motor.modo_persona = "DESGRACADO"
        decisao_pool = MagicMock(rota_final="pool_fala_pronta", subtipo="saudacao")

        res_caio = motor.responder(
            "Fala",
            decisao_jev=decisao_pool,
            face_id="caio",
        )
        self.assertEqual(res_caio["modo_persona"], "DESGRACADO")
        self.assertTrue(len(res_caio["resposta"]) > 5)
        self.assertNotIn("chefe", res_caio["resposta"].lower())

        res_prof = motor.responder(
            "Bom dia",
            decisao_jev=decisao_pool,
            face_id="felipe_sass",
        )
        self.assertEqual(res_prof["modo_persona"], "SERIO")
        self.assertIn("Felipe", res_prof["resposta"])


if __name__ == "__main__":
    unittest.main()
