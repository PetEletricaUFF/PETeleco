#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes automatizados para o Studio de System Prompt (Text-to-Text)."""

import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import requests

from src.memoria_pessoas import GerenciadorMemoriaPessoas
from src.persona_peteleco import DEFAULT_PROMPT_PATH, carregar_system_prompt_base
from src.studio_prompt import (
    StudioPromptApp,
    _chamar_gemini_studio,
    iniciar_servidor_http,
    montar_system_instruction_completa,
    salvar_system_prompt_base,
)


class StudioPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="peteleco_studio_test_"))
        self.prompt_copy = self.tmpdir / "persona_peteleco.md"
        shutil.copy2(DEFAULT_PROMPT_PATH, self.prompt_copy)
        self.perfis_copy = self.tmpdir / "perfis_pessoas.json"
        shutil.copy2(Path("data/perfis_pessoas.json"), self.perfis_copy)
        self.sqlite_tmp = self.tmpdir / "memoria_longa.sqlite"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_salvar_e_recarregar_preserva_estrutura_markdown(self) -> None:
        original_doc = self.prompt_copy.read_text(encoding="utf-8")
        novo_texto = "Você é o PETeleco vTest.\nFale sempre em 1 frase curta."
        salvo = salvar_system_prompt_base(novo_texto, self.prompt_copy)

        self.assertEqual(salvo, novo_texto)
        self.assertEqual(carregar_system_prompt_base(self.prompt_copy), novo_texto)

        atualizado_doc = self.prompt_copy.read_text(encoding="utf-8")
        self.assertIn("## 1. Os 3 Pilares da Persona v3", atualizado_doc)
        self.assertIn("=== CONTEXTO DO TURNO (injetado dinamicamente) ===", atualizado_doc)
        self.assertIn("ULTIMAS_FALAS_SALA: {ULTIMAS_FALAS_SALA}", atualizado_doc)
        self.assertIn("## 4. Simulação de Diálogos Multi-Turno", atualizado_doc)
        self.assertNotEqual(original_doc, atualizado_doc)

    def test_salvar_vazio_rejeita_sem_corromper(self) -> None:
        antes = self.prompt_copy.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            salvar_system_prompt_base("   \n ", self.prompt_copy)
        self.assertEqual(self.prompt_copy.read_text(encoding="utf-8"), antes)

    def test_montar_system_instruction_raw_e_contexto_formal(self) -> None:
        mem = GerenciadorMemoriaPessoas(
            perfis_path=self.perfis_copy,
            sqlite_path=self.sqlite_tmp,
        )
        raw_inst, raw_mode = montar_system_instruction_completa(
            prompt_base="Prompt puro de teste",
            incluir_contexto=False,
            contexto_cfg={"pessoa_id": "felipe_sass", "modo_persona": "NORMAL"},
            memoria=mem,
        )
        self.assertEqual(raw_inst, "Prompt puro de teste")
        self.assertEqual(raw_mode, "RAW")

        full_inst, eff_mode = montar_system_instruction_completa(
            prompt_base="Prompt base",
            incluir_contexto=True,
            contexto_cfg={
                "pessoa_id": "felipe_sass",
                "modo_persona": "NORMAL",
                "identificacao": "apenas_voz",
                "superficie": "bancada",
                "movimento": "parado",
                "memoria_override": "Memória custom",
            },
            memoria=mem,
        )
        self.assertEqual(eff_mode, "SERIO")
        self.assertIn("MODO_PERSONA: SERIO", full_inst)
        self.assertIn("PESSOA_ATUAL: Felipe Sass (90% voz)", full_inst)
        self.assertIn("IDENTIFICACAO: apenas_voz", full_inst)
        self.assertIn("superficie=bancada", full_inst)
        self.assertIn("MEMORIA_PESSOA: Memória custom", full_inst)

    def test_rotas_http_chat_gemini_ollama_e_sanitizacao_erro(self) -> None:
        app = StudioPromptApp(prompt_path=self.prompt_copy, perfis_path=self.perfis_copy)
        app._env["GEMINI_API_KEY"] = "SECRET_TEST_KEY_12345"

        with patch("src.studio_prompt.listar_modelos_ollama", return_value=(True, ["gemma3:4b"])):
            srv, port = iniciar_servidor_http(app, porta_inicial=0)
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{port}"
            try:
                # 1) HTML principal
                r_html = requests.get(f"{base_url}/", timeout=3)
                self.assertEqual(r_html.status_code, 200)
                self.assertIn("Studio de System Prompt", r_html.text)

                # 2) /api/state
                r_state = requests.get(f"{base_url}/api/state", timeout=3)
                self.assertEqual(r_state.status_code, 200)
                st = r_state.json()
                self.assertTrue(st["gemini_configured"])
                self.assertIn("gemma3:4b", st["ollama_models"])
                self.assertTrue(len(st["perfis"]) >= 3)

                # 3) /api/preview_prompt
                r_prev = requests.post(
                    f"{base_url}/api/preview_prompt",
                    json={
                        "prompt_base": "Teste preview",
                        "incluir_contexto": True,
                        "contexto": {"pessoa_id": "caio", "superficie": "no_colo"},
                    },
                    timeout=3,
                )
                self.assertEqual(r_prev.status_code, 200)
                self.assertIn("superficie=no_colo", r_prev.json()["system_instruction"])

                # 4) /api/chat via Gemini (mockado)
                fake_gem_resp = MagicMock()
                fake_gem_resp.status_code = 200
                fake_gem_resp.json.return_value = {
                    "candidates": [
                        {"content": {"parts": [{"text": "Fala chefe, tudo certo aqui na bancada!"}]}}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 120,
                        "candidatesTokenCount": 12,
                        "totalTokenCount": 132,
                    },
                }
                with patch.object(app._session, "post", return_value=fake_gem_resp) as mock_post:
                    r_chat = requests.post(
                        f"{base_url}/api/chat",
                        json={
                            "provider": "gemini",
                            "model": "gemini-3.5-flash-lite",
                            "prompt_base": "Prompt customizado",
                            "incluir_contexto": False,
                            "temperature": 0.7,
                            "max_tokens": 200,
                            "messages": [
                                {"role": "user", "text": "Oi"},
                                {"role": "assistant", "text": "Fala!"},
                                {"role": "user", "text": "Tudo bem?"},
                            ],
                        },
                        timeout=3,
                    )
                    self.assertEqual(r_chat.status_code, 200)
                    body = r_chat.json()
                    self.assertTrue(body["ok"])
                    self.assertEqual(body["reply"], "Fala chefe, tudo certo aqui na bancada!")
                    self.assertEqual(body["usage"]["total_tokens"], 132)
                    sent_payload = mock_post.call_args.kwargs["json"]
                    self.assertEqual(
                        sent_payload["systemInstruction"]["parts"][0]["text"],
                        "Prompt customizado",
                    )
                    self.assertEqual([c["role"] for c in sent_payload["contents"]], ["user", "model", "user"])

                # 5) Sanitizacao da API key em erro HTTP
                fake_err_resp = MagicMock()
                fake_err_resp.status_code = 403
                fake_err_resp.json.return_value = {
                    "error": {"message": "Forbidden for key SECRET_TEST_KEY_12345"}
                }
                with patch.object(app._session, "post", return_value=fake_err_resp):
                    r_err = requests.post(
                        f"{base_url}/api/chat",
                        json={
                            "provider": "gemini",
                            "model": "gemini-3.5-flash-lite",
                            "prompt_base": "Prompt",
                            "messages": [{"role": "user", "text": "Teste"}],
                        },
                        timeout=3,
                    )
                    self.assertEqual(r_err.status_code, 502)
                    self.assertNotIn("SECRET_TEST_KEY_12345", r_err.json()["error"])

                # 6) /api/chat via Ollama (mockado)
                fake_ol_resp = MagicMock()
                fake_ol_resp.status_code = 200
                fake_ol_resp.json.return_value = {
                    "message": {"content": "Resposta local Ollama"},
                    "prompt_eval_count": 55,
                    "eval_count": 8,
                }
                with patch.object(app._session, "post", return_value=fake_ol_resp) as mock_ol:
                    r_ol = requests.post(
                        f"{base_url}/api/chat",
                        json={
                            "provider": "ollama",
                            "model": "gemma3:4b",
                            "prompt_base": "Prompt Ollama",
                            "incluir_contexto": False,
                            "messages": [{"role": "user", "text": "E ai"}],
                        },
                        timeout=3,
                    )
                    self.assertEqual(r_ol.status_code, 200)
                    self.assertEqual(r_ol.json()["reply"], "Resposta local Ollama")
                    self.assertEqual(
                        mock_ol.call_args.kwargs["json"]["messages"],
                        [
                            {"role": "system", "content": "Prompt Ollama"},
                            {"role": "user", "content": "E ai"},
                        ],
                    )

                # 7) Salvar e recarregar via HTTP
                r_save = requests.post(
                    f"{base_url}/api/prompt/save",
                    json={"prompt_base": "Prompt salvo via HTTP"},
                    timeout=3,
                )
                self.assertEqual(r_save.status_code, 200)
                r_rel = requests.post(f"{base_url}/api/prompt/reload", json={}, timeout=3)
                self.assertEqual(r_rel.status_code, 200)
                self.assertEqual(r_rel.json()["prompt_base"], "Prompt salvo via HTTP")
            finally:
                srv.shutdown()
                srv.server_close()

    def test_modo_desgracado_e_trava_professor_e_retry_429(self) -> None:
        mem = GerenciadorMemoriaPessoas(
            perfis_path=self.perfis_copy,
            sqlite_path=self.sqlite_tmp,
        )
        # Caio aceita modo DESGRACADO
        inst_caio, eff_caio = montar_system_instruction_completa(
            prompt_base="Prompt teste",
            incluir_contexto=True,
            contexto_cfg={"pessoa_id": "caio", "modo_persona": "DESGRACADO"},
            memoria=mem,
        )
        self.assertEqual(eff_caio, "DESGRACADO")
        self.assertIn("MODO_PERSONA: DESGRACADO", inst_caio)

        # Felipe Sass trava obrigatoriamente em SERIO mesmo pedindo DESGRACADO
        inst_prof, eff_prof = montar_system_instruction_completa(
            prompt_base="Prompt teste",
            incluir_contexto=True,
            contexto_cfg={"pessoa_id": "felipe_sass", "modo_persona": "DESGRACADO"},
            memoria=mem,
        )
        self.assertEqual(eff_prof, "SERIO")
        self.assertIn("MODO_PERSONA: SERIO", inst_prof)

        # Retry automatico em HTTP 429
        r429 = MagicMock()
        r429.status_code = 429
        r429.headers = {}
        r429.json.return_value = {"error": {"message": "Quota exceeded. Please retry in 2.5s."}}

        r200 = MagicMock()
        r200.status_code = 200
        r200.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Resposta após backoff"}]}}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 5, "totalTokenCount": 55},
        }
        sess = MagicMock()
        sess.post.side_effect = [r429, r200]
        with patch("src.studio_prompt.time.sleep") as mock_sleep:
            res = _chamar_gemini_studio(
                session=sess,
                api_key="KEY123",
                model="gemini-3.5-flash-lite",
                system_inst="Sys",
                messages=[{"role": "user", "text": "Oi"}],
                temperature=0.8,
                max_tokens=100,
                max_retries_429=2,
            )
        self.assertTrue(res["ok"])
        self.assertEqual(res["reply"], "Resposta após backoff")
        mock_sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
