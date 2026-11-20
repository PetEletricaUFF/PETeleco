#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes automatizados da Arquitetura Multi-Cérebro (Jev) & 7 Skills Modulares."""

import json
from pathlib import Path
import random
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock
import wave

import numpy as np

from src.cerebros import GerenciadorCerebros
from src.grafo_memoria import GrafoMemoriaSQLite
from src.memoria_pessoas import GerenciadorMemoriaPessoas
from src.persona_peteleco import EstadoCorporal, MotorPersonaPETeleco
from src.router_systemone import DecisaoRoteador, RoteadorSystemOne, construir_perguntas_rover
from src.skills import (
    ContextoSkill,
    SkillBuscaWeb,
    SkillClima,
    SkillListas,
    SkillMusica,
    SkillNoticias,
    SkillRecados,
    SkillRotinas,
    SkillSonoplastia,
    SkillTimer,
    criar_gerenciador_skills,
)
from src.skills.musica import set_music_ducking
from src.skills.sonoplastia import DEFAULT_SFX_DIR, gerar_sfx_padrao
from src.studio_prompt import StudioPromptApp
from src.voz_biometria import IdentificadorVoz


class _FakeResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json = json_data
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class ArquiteturaModularTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.sqlite_tmp = self.tmp_dir / "mem_test.sqlite"
        self.cerebros_copy = self.tmp_dir / "cerebros"
        shutil.copytree(Path(__file__).resolve().parent.parent / "prompts" / "cerebros", self.cerebros_copy)
        self.perfis_copy = self.tmp_dir / "perfis.json"
        shutil.copy(Path(__file__).resolve().parent.parent / "data" / "perfis_pessoas.json", self.perfis_copy)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_carregamento_e_arbitragem_segura_multi_cerebro(self) -> None:
        ger = GerenciadorCerebros(diretorio=self.cerebros_copy)
        ids = [c.id for c in ger.listar()]
        for esperado in ("resenha", "desgracado", "tecnico", "serio", "utilitario"):
            self.assertIn(esperado, ids)
        self.assertGreaterEqual(len(ger.obter("resenha").exemplos_few_shot), 2)
        inst_res = GerenciadorCerebros.compor_system_instruction(
            cerebro=ger.obter("resenha"),
            bloco_contexto="MODO_PERSONA: NORMAL",
        )
        self.assertNotIn("=== FEW_SHOT_DIALOGO ===", inst_res)

        # Trava absoluta para professor orientador mesmo se Jev sugerir desgracado (0.99)
        prof = {"nome": "Felipe Sass", "papel": "professor_orientador", "libera_zoeira_pesada": False}
        spec_prof, motivo_prof = ger.resolver_cerebro(
            modo_persona="DESGRACADO",
            perfil=prof,
            cerebro_jev="desgracado",
            confianca_jev=0.99,
            cerebro_forcado="desgracado",
        )
        self.assertEqual(spec_prof.id, "serio")
        self.assertEqual(motivo_prof, "trava_professor_orientador")

        # Em NORMAL, Jev desgracado com confianca moderada (0.58) fica protegido em resenha
        colega_liberado = {"nome": "Caio", "papel": "membro_pet", "libera_zoeira_pesada": True}
        spec_guard, _ = ger.resolver_cerebro(
            modo_persona="NORMAL",
            perfil=colega_liberado,
            cerebro_jev="desgracado",
            confianca_jev=0.58,
        )
        self.assertEqual(spec_guard.id, "resenha")

        # Em NORMAL, Jev desgracado com alta confianca (0.85) + perfil liberado ativa desgracado
        spec_auto_d, _ = ger.resolver_cerebro(
            modo_persona="NORMAL",
            perfil=colega_liberado,
            cerebro_jev="desgracado",
            confianca_jev=0.85,
        )
        self.assertEqual(spec_auto_d.id, "desgracado")

        # Em NORMAL, desconhecido sem zoeira liberada nunca ativa desgracado automaticamente
        desconhecido = {"nome": "desconhecido", "papel": "visitante", "libera_zoeira_pesada": False}
        spec_desc, _ = ger.resolver_cerebro(
            modo_persona="NORMAL",
            perfil=desconhecido,
            cerebro_jev="desgracado",
            confianca_jev=0.95,
        )
        self.assertEqual(spec_desc.id, "resenha")

        # Pergunta tecnica via Jev ativa cerebro 'tecnico' com temperatura baixa
        spec_tec, _ = ger.resolver_cerebro(
            modo_persona="NORMAL",
            perfil=colega_liberado,
            cerebro_jev="tecnico",
            confianca_jev=0.91,
        )
        self.assertEqual(spec_tec.id, "tecnico")
        self.assertLessEqual(spec_tec.temperature, 0.45)

        # Dados de skill promovem para 'utilitario' (exceto professor)
        spec_ut, _ = ger.resolver_cerebro(
            modo_persona="NORMAL",
            perfil=colega_liberado,
            cerebro_jev="resenha",
            confianca_jev=0.70,
            tem_dados_skill=True,
        )
        self.assertEqual(spec_ut.id, "utilitario")

        # Salvar prompt customizado de um cerebro persiste no .md
        salvo = ger.salvar_prompt_cerebro("resenha", "Prompt resenha customizado de teste.", temperature=1.05)
        self.assertEqual(salvo.temperature, 1.05)
        ger2 = GerenciadorCerebros(diretorio=self.cerebros_copy)
        self.assertIn("Prompt resenha customizado", ger2.obter("resenha").prompt_base)

    def test_perguntas_jev_incluem_cerebros_e_skills(self) -> None:
        ger_c = GerenciadorCerebros(diretorio=self.cerebros_copy)
        ger_s = criar_gerenciador_skills(sqlite_path=self.sqlite_tmp, dry_run_os=True)
        qs = construir_perguntas_rover(
            criterios_cerebros=ger_c.construir_criterios_jev(),
            criterios_skills=ger_s.construir_criterios_jev(),
        )
        self.assertIn("cerebro_persona", qs)
        self.assertIn("skill_modular", qs)
        self.assertIn("tecnico", qs["cerebro_persona"]["criteria"])
        self.assertIn("clima", qs["skill_modular"]["criteria"])
        self.assertIn("recados", qs["skill_modular"]["criteria"])

    def test_skill_clima_com_cache(self) -> None:
        fake_session = MagicMock()
        fake_session.get.return_value = _FakeResponse(
            json_data={
                "current": {
                    "temperature_2m": 28.4,
                    "apparent_temperature": 31.2,
                    "relative_humidity_2m": 68,
                    "precipitation": 0.0,
                    "weather_code": 2,
                    "wind_speed_10m": 12.0,
                },
                "daily": {
                    "temperature_2m_max": [32.0],
                    "temperature_2m_min": [22.0],
                    "precipitation_probability_max": [20],
                },
            }
        )
        sk = SkillClima(cidade="Niterói", session=fake_session, cache_ttl_s=600.0)
        self.assertTrue(sk.pode_atender_local("Como tá o clima hoje?"))
        res1 = sk.executar(ContextoSkill(transcricao="como tá o clima hoje?"))
        res2 = sk.executar(ContextoSkill(transcricao="vai chover hoje?"))
        self.assertTrue(res1.sucesso)
        self.assertTrue(res2.sucesso)
        self.assertIn("Niterói", res1.resumo_contexto)
        self.assertIn("28°C", res1.resumo_contexto)
        self.assertEqual(fake_session.get.call_count, 1)  # segundo turno veio do cache

    def test_skill_musica_dry_run(self) -> None:
        sk = SkillMusica(dry_run=True)
        self.assertTrue(sk.pode_atender_local("Pausa a música aí"))
        r1 = sk.executar(ContextoSkill(transcricao="Toca Linkin Park aí"))
        self.assertTrue(r1.sucesso)
        self.assertFalse(r1.precisa_llm)
        self.assertEqual(r1.metadados["acao"], "tocar_busca")
        self.assertIn("Linkin Park", r1.metadados["consulta"])

        r2 = sk.executar(ContextoSkill(transcricao="Pausa a música", eh_formal=True))
        self.assertTrue(r2.sucesso)
        self.assertEqual(r2.metadados["acao"], "play_pause")
        self.assertEqual(r2.resposta_direta, "Reprodução pausada.")

    def test_skill_noticias_rss_mockado(self) -> None:
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
          <item><title>Novo chip de IA consome metade da energia</title></item>
          <item><title>Python 3.14 traz compilador JIT mais rápido</title></item>
        </channel></rss>"""
        fake_session = MagicMock()
        fake_session.get.return_value = _FakeResponse(text=rss_xml)
        sk = SkillNoticias(session=fake_session)
        self.assertTrue(sk.pode_atender_local("Quais são as notícias de tecnologia de hoje?"))
        res = sk.executar(ContextoSkill(transcricao="Quais são as notícias de tecnologia de hoje?"))
        self.assertTrue(res.sucesso)
        self.assertEqual(res.metadados["categoria"], "tech")
        self.assertIn("Novo chip de IA", res.resumo_contexto)

    def test_skill_recados_sqlite_e_entrega_automatica(self) -> None:
        mem = GerenciadorMemoriaPessoas(perfis_path=self.perfis_copy, sqlite_path=self.sqlite_tmp)
        sk = SkillRecados(sqlite_path=self.sqlite_tmp)
        self.assertTrue(sk.pode_atender_local("Avisa o Heitor que deixei a fonte no armário"))
        res = sk.executar(
            ContextoSkill(
                transcricao="Avisa o Heitor que deixei a fonte no armário",
                pessoa_id="caio",
                pessoa_nome="Caio",
                memoria=mem,
            )
        )
        self.assertTrue(res.sucesso)
        self.assertEqual(res.metadados["para_id"], "heitor")

        # Entrega automática na primeira interação do Heitor
        entregue = sk.consumir_recados_formatados("heitor")
        self.assertIn("deixei a fonte no armário", entregue)
        self.assertIn("Caio", entregue)
        # Segunda consulta não repete o recado já entregue
        self.assertEqual(sk.consumir_recados_formatados("heitor"), "")

    def test_skill_timer_disparo_e_cancelamento(self) -> None:
        disparados = []
        ev = threading.Event()

        def _cb(alerta):
            disparados.append(alerta)
            ev.set()

        sk = SkillTimer(on_trigger=_cb)
        self.assertTrue(sk.pode_atender_local("Bota um timer de 5 segundos pra olhar a placa"))
        sk.agendar(0.08, "tirar a placa", pessoa_nome="Caio")
        self.assertTrue(ev.wait(timeout=1.5))
        self.assertEqual(len(disparados), 1)
        alertas = sk.consumir_alertas_disparados()
        self.assertEqual(len(alertas), 1)
        self.assertIn("tirar a placa", alertas[0]["mensagem"])

        # Criar e cancelar
        res_cr = sk.executar(ContextoSkill(transcricao="Bota um timer de 10 minutos pra estudar"))
        self.assertTrue(res_cr.sucesso)
        self.assertFalse(res_cr.precisa_llm)
        self.assertEqual(len(sk.listar_ativos()), 1)
        res_can = sk.executar(ContextoSkill(transcricao="Cancela os timers"))
        self.assertEqual(res_can.metadados["cancelados"], 1)
        self.assertEqual(len(sk.listar_ativos()), 0)

    def test_skill_busca_web_mockada(self) -> None:
        fake_session = MagicMock()
        fake_session.get.return_value = _FakeResponse(
            json_data={"AbstractText": "A Lei de Ohm relaciona tensão, corrente e resistência elétrica."}
        )
        sk = SkillBuscaWeb(session=fake_session)
        self.assertTrue(sk.pode_atender_local("Pesquisa no Google o que é Lei de Ohm"))
        res = sk.executar(ContextoSkill(transcricao="Pesquisa no Google o que é Lei de Ohm"))
        self.assertTrue(res.sucesso)
        self.assertIn("Lei de Ohm", res.resumo_contexto)

    def test_skill_sonoplastia_trava_anti_cringe(self) -> None:
        sfx = SkillSonoplastia(
            sfx_dir=self.tmp_dir / "sfx",
            cooldown_s=60.0,
            min_turnos_entre_auto=3,
            prob_auto=1.0,
            auto_ativo=True,
            dry_run=True,
            rng=random.Random(42),
        )
        # Bloqueio absoluto no SERIO e no cérebro técnico
        self.assertIsNone(
            sfx.avaliar_e_tocar(
                transcricao="manda um ba dum tss",
                resposta="kkkk",
                cerebro_id="serio",
                modo_persona="SERIO",
                eh_formal=True,
            )
        )
        self.assertIsNone(
            sfx.avaliar_e_tocar(
                transcricao="qual resistor?",
                resposta="vai queimar kkkk",
                cerebro_id="tecnico",
                modo_persona="NORMAL",
                eh_formal=False,
            )
        )
        # Dispara uma vez em punchline no cérebro resenha
        hit1 = sfx.avaliar_e_tocar(
            transcricao="vou gabaritar",
            resposta="vai direto pra DP kkkk",
            cerebro_id="desgracado",
            modo_persona="DESGRACADO",
            eh_formal=False,
        )
        self.assertIsNotNone(hit1)
        self.assertEqual(hit1["efeito"], "trombone")

        # Turno imediatamente seguinte fica BLOQUEADO pelo cooldown anti-cringe!
        hit2 = sfx.avaliar_e_tocar(
            transcricao="outra piada",
            resposta="essa foi boa kkkk",
            cerebro_id="resenha",
            modo_persona="NORMAL",
            eh_formal=False,
        )
        self.assertIsNone(hit2)

    def test_motor_persona_integracao_skills_e_cerebro_por_voz(self) -> None:
        mem = GerenciadorMemoriaPessoas(perfis_path=self.perfis_copy, sqlite_path=self.sqlite_tmp)
        ger_c = GerenciadorCerebros(diretorio=self.cerebros_copy)
        ger_s = criar_gerenciador_skills(sqlite_path=self.sqlite_tmp, dry_run_os=True)
        motor = MotorPersonaPETeleco(
            memoria=mem,
            estado=EstadoCorporal(),
            cerebros=ger_c,
            skills=ger_s,
        )

        # 1) Comando de timer via Jev executa skill direta sem precisar de LLM
        dec_timer = DecisaoRoteador(
            rota_final="cognicao_gemini",
            subtipo="conversa_aberta",
            confianca=0.90,
            dirigido_ao_robo=0.95,
            cerebro_sugerido="utilitario",
            confianca_cerebro=0.88,
            skill_sugerida="timer",
            confianca_skill=0.92,
        )
        res_t = motor.responder("Bota um timer de 5 minutos pra placa", decisao_jev=dec_timer, face_id="Caio (95%)")
        self.assertEqual(res_t["origem"], "skill:timer")
        self.assertEqual(res_t["skill_executada"], "timer")
        ger_s.obter("timer").cancelar_todos()

        # 2) Troca manual de cérebro por voz funciona para colega e é bloqueada para professor
        res_pin = motor.responder("Ativa o cérebro técnico", face_id="Caio (95%)")
        self.assertEqual(res_pin["origem"], "local:troca_cerebro:tecnico")
        self.assertEqual(motor.cerebro_forcado, "tecnico")

        res_prof_blk = motor.responder("Ativa o cérebro desgracado", face_id="Felipe Sass (96%)")
        self.assertEqual(res_prof_blk["origem"], "local:trava_cerebro_professor")

    def test_studio_expoe_cerebros_e_skills(self) -> None:
        app = StudioPromptApp(perfis_path=self.perfis_copy, cerebros_dir=self.cerebros_copy)
        st = app.obter_estado_inicial()
        self.assertGreaterEqual(len(st["cerebros"]), 5)
        self.assertGreaterEqual(len(st["skills"]), 9)

    def test_sfx_wav_geracao_real_e_sonoplastia(self) -> None:
        # Garante também geração na pasta real `data/sfx/` e em diretório temporário
        gerar_sfx_padrao(DEFAULT_SFX_DIR)
        sfx_tmp = self.tmp_dir / "sfx_tmp"
        SkillSonoplastia(sfx_dir=sfx_tmp, dry_run=True)
        for nome_arq in ("ba_dum_tss.wav", "grilo.wav", "trombone.wav", "aplauso.wav"):
            p = sfx_tmp / nome_arq
            self.assertTrue(p.exists(), f"{nome_arq} não foi gerado")
            with wave.open(str(p), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertGreater(wf.getnframes(), 8000)

    def test_grafo_memoria_local_hibrido_llm_e_skill_listas(self) -> None:
        grafo = GrafoMemoriaSQLite(sqlite_path=self.sqlite_tmp)
        res_loc = grafo.ingerir_texto(
            "Guardei o multímetro no armário 2.",
            pessoa_id="caio",
            pessoa_nome="Caio",
            modo="local",
        )
        self.assertGreaterEqual(res_loc["gravadas"], 1)
        self.assertEqual(res_loc["total_tokens"], 0)

        # Extração LLM mockada + busca de subgrafo
        fake_sess = MagicMock()
        fake_sess.post.return_value = _FakeResponse(
            json_data={
                "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 20, "totalTokenCount": 70},
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        [
                                            {
                                                "sujeito": "Nathan",
                                                "tipo_sujeito": "pessoa",
                                                "relacao": "mora_em",
                                                "objeto": "Icaraí",
                                                "tipo_objeto": "local",
                                            }
                                        ]
                                    )
                                }
                            ]
                        }
                    }
                ],
            }
        )
        res_llm = grafo.ingerir_texto(
            "Na verdade eu moro em Icaraí.",
            pessoa_id="nathan",
            pessoa_nome="Nathan",
            modo="hibrido",
            api_key="fake-key",
            session=fake_sess,
        )
        self.assertEqual(res_llm["gravadas"], 1)
        self.assertEqual(res_llm["total_tokens"], 70)

        sub = grafo.formatar_subgrafo_prompt("caio", "Onde ficou o multímetro?")
        self.assertIn("armário 2", sub)

        # SkillListas (adicionar, ler, remover, limpar)
        sk_list = SkillListas(grafo=grafo)
        self.assertTrue(sk_list.pode_atender_local("Bota resistor 10k na lista de compras"))
        r_add1 = sk_list.executar(ContextoSkill(transcricao="Bota resistor 10k na lista de compras"))
        r_add2 = sk_list.executar(ContextoSkill(transcricao="Adiciona capacitor 100uF na lista de compras"))
        self.assertTrue(r_add1.sucesso and r_add2.sucesso)
        r_ler = sk_list.executar(ContextoSkill(transcricao="O que tem na lista de compras?"))
        self.assertIn("resistor 10k", r_ler.resposta_direta)
        self.assertIn("capacitor 100uF", r_ler.resposta_direta)

        r_rem = sk_list.executar(ContextoSkill(transcricao="Tira resistor 10k da lista de compras"))
        self.assertEqual(r_rem.metadados["removidos"], 1)
        self.assertNotIn("resistor 10k", sk_list.executar(ContextoSkill(transcricao="Lê a lista de compras")).resposta_direta)

        r_lim = sk_list.executar(ContextoSkill(transcricao="Limpa a lista de compras"))
        self.assertEqual(r_lim.metadados["removidos"], 1)

    def test_musica_headless_fila_e_ducking(self) -> None:
        sk = SkillMusica(dry_run=True)
        r_play = sk.executar(ContextoSkill(transcricao="Toca Linkin Park aí"))
        self.assertTrue(r_play.sucesso)
        self.assertGreaterEqual(len(sk.player.fila), 2)

        r_q = sk.executar(ContextoSkill(transcricao="Que música tá tocando?"))
        self.assertIn("Linkin Park", r_q.resposta_direta)

        set_music_ducking(True)
        self.assertTrue(sk.player.ducking_ativo)
        set_music_ducking(False)
        self.assertFalse(sk.player.ducking_ativo)

        sk.executar(ContextoSkill(transcricao="Adiciona Charlie Brown Jr na fila"))
        r_next = sk.executar(ContextoSkill(transcricao="Pula a música"))
        self.assertTrue(r_next.sucesso)
        sk.executar(ContextoSkill(transcricao="Aumenta o volume"))
        self.assertGreater(sk.player.volume, 0.80)
        sk.executar(ContextoSkill(transcricao="Para a música"))
        self.assertFalse(sk.player.tocando)

    def test_skill_rotinas_modo_estudo_e_bom_dia(self) -> None:
        grafo = GrafoMemoriaSQLite(sqlite_path=self.sqlite_tmp)
        ger = criar_gerenciador_skills(sqlite_path=self.sqlite_tmp, grafo=grafo, dry_run_os=True)

        # Saudacao simples "Bom dia!" NÃO deve disparar rotina
        sk_rot = ger.obter("rotinas")
        self.assertFalse(sk_rot.pode_atender_local("Bom dia!"))
        self.assertTrue(sk_rot.pode_atender_local("Bom dia PETeleco"))
        self.assertTrue(sk_rot.pode_atender_local("Ativa o modo estudo"))

        # Adiciona tarefa e recado para validar rotina bom_dia
        ger.obter("listas").executar(ContextoSkill(transcricao="Adiciona revisar PCB na lista de tarefas"))
        ger.obter("recados").salvar_recado(
            de_pessoa_id="heitor",
            de_nome="Heitor",
            para_pessoa_id="caio",
            para_nome="Caio",
            mensagem="Traz o osciloscópio",
        )
        ger.obter("clima")._buscar_open_meteo = lambda: {
            "cidade": "Niterói",
            "temp_c": 29.0,
            "sensacao_c": 32.0,
            "umidade_pct": 65,
            "vento_kmh": 10.0,
            "chuva_mm": 0.0,
            "condicao": "céu limpo",
            "prob_chuva_max_pct": 10,
            "temp_min_c": 22.0,
            "temp_max_c": 31.0,
        }
        ger.obter("noticias")._buscar_manchetes = lambda _cat: ("G1", ["Manchete A", "Manchete B"])

        r_bd = sk_rot.executar(ContextoSkill(transcricao="Bom dia PETeleco", pessoa_id="caio", pessoa_nome="Caio"))
        self.assertTrue(r_bd.sucesso)
        self.assertTrue(r_bd.precisa_llm)
        self.assertIn("revisar PCB", r_bd.resumo_contexto)
        self.assertIn("Traz o osciloscópio", r_bd.resumo_contexto)
        self.assertIn("Niterói", r_bd.resumo_contexto)

        # Modo estudo -> timer 25 min, sonoplastia auto desligado, lo-fi vol 0.35
        r_est = sk_rot.executar(ContextoSkill(transcricao="Ativa o modo estudo", pessoa_id="caio", pessoa_nome="Caio"))
        self.assertTrue(r_est.sucesso)
        self.assertFalse(r_est.precisa_llm)
        self.assertFalse(ger.obter("sonoplastia").auto_ativo)
        self.assertEqual(len(ger.obter("timer").listar_ativos()), 1)
        self.assertAlmostEqual(ger.obter("musica").player.volume, 0.35, places=2)

        r_sair = sk_rot.executar(ContextoSkill(transcricao="Sai do modo estudo"))
        self.assertTrue(r_sair.sucesso)
        self.assertTrue(ger.obter("sonoplastia").auto_ativo)
        self.assertEqual(len(ger.obter("timer").listar_ativos()), 0)
        self.assertFalse(ger.obter("musica").player.tocando)

    def test_voz_biometria_dois_niveis_e_snr(self) -> None:
        voices_tmp = self.tmp_dir / "voices.json"
        vb = IdentificadorVoz(
            voices_path=voices_tmp,
            limiar_confianca=0.82,
            limiar_duvida=0.70,
            margem_minima=0.025,
            preferir_ge2e=True,
            auto_download_ge2e=False,
        )
        sr = 16000
        t = np.linspace(0.0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
        fala = 0.6 * np.sin(2.0 * np.pi * 150.0 * t) + 0.25 * np.sin(2.0 * np.pi * 300.0 * t)
        # Adiciona pausa no inicio/fim (simula fala com VAD pre-roll -> SNR alto)
        sil = np.zeros(int(0.5 * sr), dtype=np.float32)
        sinal_limpo = np.concatenate([sil, fala, sil])
        self.assertGreater(vb.estimar_snr_db(sinal_limpo, sr=sr), 15.0)
        self.assertGreaterEqual(vb.cadastrar_amostra("caio", sinal_limpo, sr=sr), 1)

        # 1) Áudio limpo da mesma voz -> confirmado
        det_ok = vb.identificar_detalhado(sinal_limpo, sr=sr)
        self.assertEqual(det_ok["status"], "confirmado")
        self.assertEqual(det_ok["pessoa_id"], "caio")
        self.assertFalse(det_ok["requer_confirmacao"])

        # 2) Áudio contínuo com SNR baixo (< 7 dB) da mesma voz -> nível 'confirmar'
        rng = np.random.default_rng(7)
        sinal_ruidoso = fala + rng.normal(0.0, 0.015, fala.shape).astype(np.float32)
        det_ruido = vb.identificar_detalhado(sinal_ruidoso, sr=sr)
        self.assertLess(det_ruido["snr_db"], 7.0)
        self.assertGreaterEqual(det_ruido["confianca"], 0.72)
        self.assertEqual(det_ruido["status"], "confirmar")
        self.assertEqual(det_ruido["candidato_id"], "caio")
        self.assertTrue(det_ruido["requer_confirmacao"])

        # 3) Voz sintética diferente -> desconhecido
        outra = np.concatenate([sil, 0.6 * np.sin(2.0 * np.pi * 680.0 * t), sil])
        vb.limiar_duvida = 0.90
        det_desc = vb.identificar_detalhado(outra, sr=sr)
        self.assertEqual(det_desc["status"], "desconhecido")
        self.assertFalse(det_desc["requer_confirmacao"])


if __name__ == "__main__":
    unittest.main()
