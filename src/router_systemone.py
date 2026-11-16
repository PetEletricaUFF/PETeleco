#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 2 — Roteador Cognitivo Jev-based (Ollama /v1/systemone).

Avalia contexto multimodal textual (`state`) contra perguntas tipadas
(`noul`, `choice`) em uma chamada HTTP local, retornando a rota de decisao
e probabilidades calibradas para travas de seguranca motora.

Desacoplado de microfone, TTS, UDP e Gemini para permitir reuso entre camadas
(CLI, servidor HTTP remoto, pipeline de voz e simulacoes).
"""

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Dict, Optional
import unicodedata

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "tev1:0.8b"


def _norm_simples(texto: str) -> str:
    decomp = unicodedata.normalize("NFKD", texto or "")
    sem = "".join(c for c in decomp if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", sem.lower())).strip()

_MAPA_ESTADO_ROVER = {
    "STANDBY": "idle_ready (motors off, waiting)",
    "AGUARDANDO_CONFIRMACAO": "waiting_user_confirmation",
    "RADAR": "scanning_with_camera",
    "CENTRALIZA": "centering_target",
    "ALINHA": "aligning_chassis",
    "APROXIMA": "driving_towards_target",
    "RODEIO": "orbiting_target",
}


@dataclass
class ContextoRover:
    """Snapshot textual dos sensores e estado do robo para o roteador."""

    transcricao: str
    estado_rover: str = "STANDBY"
    pessoa_sface: str = "nenhuma"
    objetos_yolo: str = "nenhum"
    modo_escuta_ativa: bool = False

    def formatar_state(self) -> str:
        """Formata o estado estruturado enviado ao campo `state` do SystemOne."""
        estado_desc = _MAPA_ESTADO_ROVER.get(self.estado_rover, self.estado_rover)
        base = (
            f'USER_SPEECH (pt-BR): "{self.transcricao.strip()}"\n'
            f"CURRENT_ROVER_MODE: {estado_desc}\n"
            f"LOCAL_FACE_ID (SFace): {self.pessoa_sface}\n"
            f"LOCAL_OBJECTS_SEEN (YOLO): {self.objetos_yolo}"
        )
        if self.modo_escuta_ativa:
            base += "\nACTIVE_LISTENING: true (user already woke PETeleco, treat speech as directed at robot)"
        return base


@dataclass
class DecisaoRoteador:
    """Resultado consolidado do roteador hierarquico + telemetria de latencia."""

    rota_final: str
    subtipo: str
    confianca: float
    dirigido_ao_robo: float
    requer_confirmacao: bool = False
    precisa_imagem_camera: bool = False
    prob_imagem_camera: float = 0.0
    latencia_ms: float = 0.0
    respostas_brutas: Dict[str, Any] = field(default_factory=dict)


def construir_perguntas_rover() -> Dict[str, Any]:
    """Define o conjunto paralelo de perguntas Jev-based do Rover Cognitivo."""
    return {
        "dirigido_ao_robo": {
            "type": "noul",
            "instructions": (
                "Is USER_SPEECH directed at the Rover robot (speaking to the robot/assistant), "
                "or is it unrelated speech between humans / TV broadcast?"
            ),
            "criteria": {
                "true": (
                    "User addresses the robot, greets/says goodbye/compliments, asks a question "
                    "(factual, visual, or scientific), confirms an action, or gives a movement/stop command."
                ),
                "false": (
                    "Speaker explicitly addresses another human being by name/vocative (talking to a colleague), "
                    "or it is a TV/radio/news/sports broadcast not meant for the robot."
                ),
            },
        },
        "rota_principal": {
            "type": "choice",
            "instructions": (
                "Based ONLY on USER_SPEECH intent (ignore CURRENT_ROVER_MODE unless confirming), "
                "which single handling layer owns this utterance?"
            ),
            "criteria": {
                "ignorar": (
                    "Speech explicitly directed to another human being by name/vocative, "
                    "or TV/news/sports broadcast audio not meant for the robot."
                ),
                "pool_fala_pronta": (
                    "Short non-informational social expression ONLY (NO factual question about clock time, weather, "
                    "battery, or person identity): greeting or asking how the robot is doing, farewell, "
                    "short confirmation, or thanks/praise."
                ),
                "fala_parametrizada": (
                    "Direct question for live local telemetry/status or face recognition ONLY: current clock time/date, "
                    "weather/rain forecast, robot battery/voltage/temperature, or name/identity of the person seen in front."
                ),
                "acao_rover": (
                    "Command asking the robot to physically move, follow someone, go/move closer to an object, "
                    "stop/cut motors, or rotate camera/scan to search for a target."
                ),
                "cognicao_gemini": (
                    "Request to visually inspect/read/describe what is on a surface/scene, "
                    "or an open-ended 'why/how' scientific, technical, or general-knowledge question."
                ),
            },
        },
        "pool_fala_pronta": {
            "type": "choice",
            "instructions": "If USER_SPEECH is a short social utterance (and NOT a request for a joke, explanation, or command), which canned category fits?",
            "criteria": {
                "saudacao": "Pure greeting ONLY like oi, ola, bom dia, boa tarde, tudo bem. NOT a movement command like vamos para a frente.",
                "despedida": "Farewell like tchau, ate logo, ate amanha.",
                "ok_confirmacao": "Short confirmation like sim, ok, beleza, pode ir, entendi, confirmar.",
                "elogio_agradecimento": "Short thank-you or praise of the robot's action ONLY (e.g. obrigado, valeu, mandou bem). NOT asking for a joke, story, or help.",
                "nenhuma": "Not a simple social phrase (it is a command, request for joke/story, telemetry question, complex question, or side chatter).",
            },
        },
        "fala_parametrizada": {
            "type": "choice",
            "instructions": "Does USER_SPEECH ask for one of these four deterministic local values?",
            "criteria": {
                "hora_atual": "Asking what time of day, weekday, or calendar date it is right now (e.g. que horas sao, que dia e hoje).",
                "previsao_clima": "Asking for the current weather, rain forecast, or whether it is cold/hot/sunny outside right now.",
                "bateria_temp": "Asking about the robot's battery level, voltage, or hardware temperature.",
                "quem_vejo_agora": "Asking who the person in front of the robot is or if the robot recognizes them.",
                "nenhuma": "Not asking for time/date, weather forecast, robot battery/temp, or person identity (e.g. general science why-questions).",
            },
        },
        "acao_rover": {
            "type": "choice",
            "instructions": (
                "Does USER_SPEECH explicitly command the robot to change physical movement or camera scan? "
                "Do NOT choose 'parar' unless the user literally says stop/brake/cut motors."
            ),
            "criteria": {
                "seguir_rosto": "Command to follow the speaker/user or accompany/track a person or face as they move (e.g. seguir, acompanhar).",
                "ir_ate_objeto": "Command to move/drive forward, backward, turn, approach, drive up to, or go closer to a target or location (e.g. ir ate, aproximar, chegar perto, vamos para a frente, anda pra frente, da re).",
                "parar": "Explicit command to stop, halt, brake, or cut motors immediately.",
                "radar": "Command to search, scan, look around, or rotate the camera head to find a target.",
                "nenhuma": "Does not command physical movement, stopping, or camera scanning (greetings, questions, facts, or chatter).",
            },
        },
        "cognicao_gemini": {
            "type": "choice",
            "instructions": "If complex cloud cognition is needed, what kind of request is USER_SPEECH?",
            "criteria": {
                "descrever_cena": "Asking what the robot sees on the table/scene, reading written text, or visual inspection.",
                "conversa_aberta": "Technical explanation, conceptual question, telling jokes, chatting, or open-ended conversation.",
                "nenhuma": "Can be handled locally without cloud LLM/VLM.",
            },
        },
        "precisa_imagem_camera": {
            "type": "noul",
            "instructions": (
                "Does answering USER_SPEECH actually require inspecting a live camera photo "
                "(e.g. describing objects, reading paper/labels), rather than local time/weather/battery/commands?"
            ),
            "criteria": {
                "true": "User asks to describe what is in front/on the table, read text on paper, or visually analyze a scene.",
                "false": "Time, weather, battery, face-ID lookup, greetings, motor commands, or pure conceptual questions.",
            },
        },
    }


class RoteadorSystemOne:
    """Cliente HTTP para o endpoint `/v1/systemone` do Ollama com arbitragem em camadas."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 30.0,
        keep_alive: Any = -1,
        limiar_dirigido: float = 0.55,
        limiar_acao_direta: float = 0.80,
        limiar_acao_confirmacao: float = 0.50,
        limiar_imagem: float = 0.55,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.limiar_dirigido = limiar_dirigido
        self.limiar_acao_direta = limiar_acao_direta
        self.limiar_acao_confirmacao = limiar_acao_confirmacao
        self.limiar_imagem = limiar_imagem
        self._session = requests.Session()
        self._questions = construir_perguntas_rover()

    def verificar_servidor(self) -> str:
        """Consulta a versao do servidor Ollama local e retorna a string de versao."""
        resp = self._session.get(f"{self.base_url}/api/version", timeout=3.0)
        resp.raise_for_status()
        return str(resp.json().get("version", "desconhecida"))

    def consultar_bruto(
        self,
        contexto: ContextoRover,
        timeout_override_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executa uma chamada raw ao `/v1/systemone` e mede a latencia total HTTP."""
        payload = {
            "model": self.model,
            "state": contexto.formatar_state(),
            "questions": self._questions,
            "keep_alive": self.keep_alive,
        }
        t0 = time.perf_counter()
        resp = self._session.post(
            f"{self.base_url}/v1/systemone",
            json=payload,
            timeout=timeout_override_s or self.timeout_s,
        )
        latencia_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        data = resp.json()
        data["_latencia_ms"] = latencia_ms
        return data

    def decidir(
        self,
        contexto: ContextoRover,
        timeout_override_s: Optional[float] = None,
    ) -> DecisaoRoteador:
        """Avalia o contexto via SystemOne e aplica travas deterministicas de seguranca."""
        bruto = self.consultar_bruto(contexto, timeout_override_s=timeout_override_s)
        answers = bruto.get("answers", {})
        latencia_ms = float(bruto.get("_latencia_ms", 0.0))

        p_dirigido = float(answers.get("dirigido_ao_robo", {}).get("noul", 0.0))
        p_imagem = float(answers.get("precisa_imagem_camera", {}).get("noul", 0.0))

        rota_ans = answers.get("rota_principal", {})
        rota_probs = rota_ans.get("probabilities") or {}
        rota_choice = str(rota_ans.get("choice", "cognicao_gemini"))
        rota_prob = self._prob_da_escolha(rota_ans, rota_choice)

        acao_choice, acao_prob = self._melhor_subtipo(answers.get("acao_rover", {}))
        param_choice, param_prob = self._melhor_subtipo(answers.get("fala_parametrizada", {}))
        pool_choice, pool_prob = self._melhor_subtipo(answers.get("pool_fala_pronta", {}))
        gem_choice, gem_prob = self._melhor_subtipo(answers.get("cognicao_gemini", {}))

        norm_speech = _norm_simples(contexto.transcricao)

        # Trava de emergencia: parada explicita com alta confianca vence mesmo em fala rapida
        if acao_choice == "parar" and acao_prob >= 0.90 and (p_dirigido >= 0.50 or contexto.modo_escuta_ativa):
            return DecisaoRoteador(
                rota_final="acao_rover",
                subtipo="parar",
                confianca=acao_prob,
                dirigido_ao_robo=p_dirigido,
                requer_confirmacao=False,
                precisa_imagem_camera=False,
                prob_imagem_camera=p_imagem,
                latencia_ms=latencia_ms,
                respostas_brutas=answers,
            )

        # Filtro combinado de escuta: desativado em Modo Escuta Ativa (pos wake-word)
        p_ignorar = float(rota_probs.get("ignorar", 0.0))
        invocou_robo = any(w in norm_speech for w in ("peteleco", "pe teleco", "teleco", "robo"))
        if not contexto.modo_escuta_ativa and not invocou_robo:
            if (
                p_dirigido < 0.35
                or (p_dirigido < self.limiar_dirigido and p_ignorar >= 0.50)
                or (p_ignorar >= 0.85 and p_dirigido < 0.65)
            ):
                return DecisaoRoteador(
                    rota_final="ignorar",
                    subtipo="conversa_paralela_ou_ruido",
                    confianca=max(1.0 - p_dirigido, p_ignorar),
                    dirigido_ao_robo=p_dirigido,
                    precisa_imagem_camera=False,
                    prob_imagem_camera=p_imagem,
                    latencia_ms=latencia_ms,
                    respostas_brutas=answers,
                )

        # Caso rota_principal tenha empatado abaixo de 0.50 em ignorar, usa a melhor camada ativa
        if rota_choice == "ignorar":
            candidatas = {
                k: float(v)
                for k, v in rota_probs.items()
                if k != "ignorar"
            }
            if candidatas:
                rota_choice = max(candidatas, key=candidatas.get)
                rota_prob = candidatas[rota_choice]

        parece_movimento_livre = any(
            kw in norm_speech
            for kw in (
                "para a frente",
                "pra frente",
                "para frente",
                "vai pra frente",
                "anda pra frente",
                "vamos para frente",
                "vamos para a frente",
                "para tras",
                "pra tras",
                "da re",
                "dar re",
                "vira pra esquerda",
                "vira pra direita",
            )
        )

        if rota_choice == "acao_rover":
            if acao_choice != "nenhuma":
                precisa_confirmar = acao_choice != "parar" and acao_prob < self.limiar_acao_direta
                return DecisaoRoteador(
                    rota_final="acao_rover",
                    subtipo=acao_choice,
                    confianca=acao_prob,
                    dirigido_ao_robo=p_dirigido,
                    requer_confirmacao=precisa_confirmar,
                    precisa_imagem_camera=False,
                    prob_imagem_camera=p_imagem,
                    latencia_ms=latencia_ms,
                    respostas_brutas=answers,
                )
            # Se rota_principal indicou acao fisica e uma subacao clara ou movimento livre aparece,
            # escala em vez de descartar silenciosamente para saudacao falsa.
            acao_probs = answers.get("acao_rover", {}).get("probabilities") or {}
            acoes_validas = {k: float(v) for k, v in acao_probs.items() if k != "nenhuma"}
            if acoes_validas and (
                (rota_prob >= 0.90 and (p_dirigido >= 0.65 or contexto.modo_escuta_ativa))
                or (rota_prob >= 0.65 and parece_movimento_livre)
            ):
                sub_acao = "ir_ate_objeto" if parece_movimento_livre else max(acoes_validas, key=acoes_validas.get)
                sub_prob = max(acoes_validas.get(sub_acao, 0.0), rota_prob * 0.7 if parece_movimento_livre else 0.0)
                if sub_prob >= 0.25 or parece_movimento_livre:
                    return DecisaoRoteador(
                        rota_final="acao_rover",
                        subtipo=sub_acao,
                        confianca=sub_prob,
                        dirigido_ao_robo=p_dirigido,
                        requer_confirmacao=not parece_movimento_livre,
                        precisa_imagem_camera=False,
                        prob_imagem_camera=p_imagem,
                        latencia_ms=latencia_ms,
                        respostas_brutas=answers,
                    )

            # Se acao_rover afirmou 'nenhuma', evita acionar motores por falso positivo em rota_principal
            outras = {
                k: float(v)
                for k, v in rota_probs.items()
                if k != "acao_rover" and (not contexto.modo_escuta_ativa or k != "ignorar")
            }
            if outras:
                rota_choice = max(outras, key=outras.get)
                rota_prob = outras[rota_choice]
                if rota_choice == "ignorar":
                    return DecisaoRoteador(
                        rota_final="ignorar",
                        subtipo="conversa_paralela_ou_ruido",
                        confianca=max(1.0 - p_dirigido, rota_prob),
                        dirigido_ao_robo=p_dirigido,
                        precisa_imagem_camera=False,
                        prob_imagem_camera=p_imagem,
                        latencia_ms=latencia_ms,
                        respostas_brutas=answers,
                    )

        # Fallback fonetico para consulta de data quando STT engole "dia" -> "que de hoje"
        if any(f in norm_speech for f in ("que dia e hoje", "que de hoje", "que dia hoje", "data de hoje")):
            return DecisaoRoteador(
                rota_final="fala_parametrizada",
                subtipo="hora_atual",
                confianca=max(param_prob, 0.90),
                dirigido_ao_robo=max(p_dirigido, 0.90),
                precisa_imagem_camera=False,
                prob_imagem_camera=p_imagem,
                latencia_ms=latencia_ms,
                respostas_brutas=answers,
            )

        # Guardrail para pedidos abertos (ex.: "conta uma piada boa ai") nao cairem em elogio/saudacao pronta
        parece_pedido_aberto = any(
            kw in norm_speech
            for kw in ("piada", "historia", "explica", "como funciona", "por que", "o que e", "zoeira", "conta uma")
        )
        if rota_choice == "pool_fala_pronta" and parece_pedido_aberto:
            rota_choice = "cognicao_gemini"

        # Desempate por especialidade: consulta deterministica forte (hora/clima/bateria/rosto)
        # prevalece sobre confusao de rota_principal com social curto ou Gemini vago
        if (
            param_choice != "nenhuma"
            and param_prob >= 0.85
            and rota_choice in ("pool_fala_pronta", "cognicao_gemini")
            and p_imagem < self.limiar_imagem
        ):
            return DecisaoRoteador(
                rota_final="fala_parametrizada",
                subtipo=param_choice,
                confianca=param_prob,
                dirigido_ao_robo=p_dirigido,
                precisa_imagem_camera=False,
                prob_imagem_camera=p_imagem,
                latencia_ms=latencia_ms,
                respostas_brutas=answers,
            )

        if rota_choice == "fala_parametrizada" and param_choice != "nenhuma":
            return DecisaoRoteador(
                rota_final="fala_parametrizada",
                subtipo=param_choice,
                confianca=max(param_prob, rota_prob * 0.9),
                dirigido_ao_robo=p_dirigido,
                precisa_imagem_camera=False,
                prob_imagem_camera=p_imagem,
                latencia_ms=latencia_ms,
                respostas_brutas=answers,
            )

        if rota_choice == "pool_fala_pronta" and pool_choice != "nenhuma":
            return DecisaoRoteador(
                rota_final="pool_fala_pronta",
                subtipo=pool_choice,
                confianca=max(pool_prob, rota_prob * 0.9),
                dirigido_ao_robo=p_dirigido,
                precisa_imagem_camera=False,
                prob_imagem_camera=p_imagem,
                latencia_ms=latencia_ms,
                respostas_brutas=answers,
            )

        subtipo_gem = gem_choice if gem_choice != "nenhuma" else "conversa_aberta"
        precisa_img = p_imagem >= self.limiar_imagem or subtipo_gem == "descrever_cena"
        return DecisaoRoteador(
            rota_final="cognicao_gemini",
            subtipo=subtipo_gem,
            confianca=max(gem_prob, rota_prob),
            dirigido_ao_robo=p_dirigido,
            precisa_imagem_camera=precisa_img,
            prob_imagem_camera=p_imagem,
            latencia_ms=latencia_ms,
            respostas_brutas=answers,
        )

    @staticmethod
    def _prob_da_escolha(answer: Dict[str, Any], choice: str) -> float:
        """Extrai a probabilidade especifica da classe escolhida (com fallback em confidence)."""
        probs = answer.get("probabilities") or {}
        if choice in probs:
            return float(probs[choice])
        return float(answer.get("confidence", 0.0))

    @classmethod
    def _melhor_subtipo(cls, answer: Dict[str, Any]) -> tuple[str, float]:
        """Retorna o subtipo escolhido e sua probabilidade calibrada."""
        choice = str(answer.get("choice", "nenhuma"))
        return choice, cls._prob_da_escolha(answer, choice)

    def decidir_gemini(
        self,
        contexto: ContextoRover,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        timeout_s: float = 8.0,
    ) -> Optional[DecisaoRoteador]:
        """
        Roteador rapido via Gemini JSON (~500 ms em maquinas sem GPU CUDA),
        mantendo o mesmo contrato `DecisaoRoteador` + travas fisicas e `modo_escuta_ativa`.
        """
        key = (api_key or "").strip()
        if not key:
            return None

        norm_speech = _norm_simples(contexto.transcricao)
        if any(f in norm_speech for f in ("que dia e hoje", "que de hoje", "que dia hoje", "data de hoje")):
            return DecisaoRoteador(
                rota_final="fala_parametrizada",
                subtipo="hora_atual",
                confianca=0.96,
                dirigido_ao_robo=0.98,
                precisa_imagem_camera=False,
                prob_imagem_camera=0.02,
                latencia_ms=1.0,
                respostas_brutas={"router_backend": "fast_rule:data_hoje"},
            )

        rotas_validas = {
            "ignorar": ["conversa_paralela_ou_ruido"],
            "pool_fala_pronta": ["saudacao", "despedida", "ok_confirmacao", "elogio_agradecimento"],
            "fala_parametrizada": ["hora_atual", "previsao_clima", "bateria_temp", "quem_vejo_agora"],
            "acao_rover": ["seguir_rosto", "ir_ate_objeto", "parar", "radar"],
            "cognicao_gemini": ["descrever_cena", "conversa_aberta"],
        }
        prompt = (
            "Voce e o roteador de intencoes do robo PETeleco (em portugues BR). "
            "Classifique USER_SPEECH no formato JSON exato.\n"
            f"ESTADO:\n{contexto.formatar_state()}\n\n"
            "Regras:\n"
            "- Se ACTIVE_LISTENING=true ou mencionar 'PETeleco'/'robo', NUNCA escolha 'ignorar'.\n"
            "- 'vamos para a frente', 'anda', 'chega perto', 'vira', 'da re', 'me segue', 'para tudo', 'procura': rota 'acao_rover'.\n"
            "- Pedidos de hora/data ('que dia e hoje', 'que horas sao'), bateria/temperatura, clima ou 'quem sou eu': rota 'fala_parametrizada'.\n"
            "- Despedida ('tchau', 'pode dormir', 'encerrar escuta') ou confirmacao seca ('sim', 'pode ir', 'confirmo' em AGUARDANDO_CONFIRMACAO): rota 'pool_fala_pronta' (NUNCA classifique apenas 'PETeleco' como despedida).\n"
            "- Cumprimentos com assunto, zoeira, piada, bate-papo, perguntas abertas ou visao ('olha isso', 'o que voce ve'): rota 'cognicao_gemini' "
            "(para que o PETeleco responda de forma organica e bem-humorada ao vivo!).\n"
            "- Escolha 'ignorar' apenas se ACTIVE_LISTENING nao estiver ativo e for conversa clara entre terceiros ou TV.\n\n"
            'Retorne APENAS JSON valido: {"rota": str, "subtipo": str, "confianca": float, '
            '"dirigido_ao_robo": float, "precisa_imagem": bool, "requer_confirmacao": bool}'
        )
        t0 = time.perf_counter()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 180,
                "responseMimeType": "application/json",
            },
        }
        try:
            resp = self._session.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            partes = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
            raw_txt = "".join(str(p.get("text", "")) for p in partes if not p.get("thought")).strip()
            dados = json.loads(raw_txt)
        except Exception:
            return None

        rota = str(dados.get("rota", "cognicao_gemini")).strip()
        if rota not in rotas_validas:
            rota = "cognicao_gemini"
        if rota == "ignorar" and contexto.modo_escuta_ativa:
            rota = "cognicao_gemini"

        subtipo = str(dados.get("subtipo", rotas_validas[rota][0])).strip()
        if subtipo not in rotas_validas[rota]:
            subtipo = rotas_validas[rota][-1] if rota == "cognicao_gemini" else rotas_validas[rota][0]

        conf = min(1.0, max(0.0, float(dados.get("confianca", 0.9))))
        dirig = min(1.0, max(0.0, float(dados.get("dirigido_ao_robo", 0.95))))
        if contexto.modo_escuta_ativa:
            dirig = max(dirig, 0.95)
        precisa_img = bool(dados.get("precisa_imagem", subtipo == "descrever_cena"))
        req_conf = bool(dados.get("requer_confirmacao", False))
        if rota != "acao_rover" or subtipo == "parar":
            req_conf = False

        probs_fake = {k: (conf if k == rota else round((1.0 - conf) / 4.0, 4)) for k in rotas_validas}
        return DecisaoRoteador(
            rota_final=rota,
            subtipo=subtipo,
            confianca=conf,
            dirigido_ao_robo=dirig,
            requer_confirmacao=req_conf,
            precisa_imagem_camera=precisa_img,
            prob_imagem_camera=0.92 if precisa_img else 0.05,
            latencia_ms=lat_ms,
            respostas_brutas={
                "router_backend": f"gemini_router({model})",
                "rota_principal": {"choice": rota, "probabilities": probs_fake, "confidence": conf},
                "dirigido_ao_robo": {"noul": dirig},
                "precisa_imagem_camera": {"noul": 0.92 if precisa_img else 0.05},
                rota: {"choice": subtipo, "probabilities": {subtipo: conf}, "confidence": conf},
            },
        )
