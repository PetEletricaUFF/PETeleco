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
import time
from typing import Any, Dict, Optional

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "tev1:0.8b"

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

    def formatar_state(self) -> str:
        """Formata o estado estruturado enviado ao campo `state` do SystemOne."""
        estado_desc = _MAPA_ESTADO_ROVER.get(self.estado_rover, self.estado_rover)
        return (
            f'USER_SPEECH (pt-BR): "{self.transcricao.strip()}"\n'
            f"CURRENT_ROVER_MODE: {estado_desc}\n"
            f"LOCAL_FACE_ID (SFace): {self.pessoa_sface}\n"
            f"LOCAL_OBJECTS_SEEN (YOLO): {self.objetos_yolo}"
        )


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
            "instructions": "If USER_SPEECH is a short social utterance, which canned audio category fits?",
            "criteria": {
                "saudacao": "Greeting like oi, ola, bom dia, boa tarde, tudo bem.",
                "despedida": "Farewell like tchau, ate logo, ate amanha.",
                "ok_confirmacao": "Short confirmation like sim, ok, beleza, pode ir, entendi, confirmar.",
                "elogio_agradecimento": "Thanks, praise, approval, or enthusiastic positive reaction/compliment about performance (e.g. obrigado, valeu, gostei, excelente).",
                "nenhuma": "Not a simple social phrase (it is a command, telemetry question, complex question, or side chatter).",
            },
        },
        "fala_parametrizada": {
            "type": "choice",
            "instructions": "Does USER_SPEECH ask for one of these four deterministic local values?",
            "criteria": {
                "hora_atual": "Asking what time of day or calendar date it is right now.",
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
                "ir_ate_objeto": "Command to move toward, approach, drive up to, or get closer to a specific object or furniture (e.g. ir ate, aproximar, chegar perto).",
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
                "conversa_aberta": "Technical explanation, conceptual question, or open-ended conversation.",
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

        # Trava de emergencia: parada explicita com alta confianca vence mesmo em fala rapida
        if acao_choice == "parar" and acao_prob >= 0.90 and p_dirigido >= 0.50:
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

        # Filtro combinado de escuta: noul baixo ou rota_principal forte em ignorar
        p_ignorar = float(rota_probs.get("ignorar", 0.0))
        if (
            p_dirigido < 0.35
            or (p_dirigido < self.limiar_dirigido and p_ignorar >= 0.35)
            or (p_ignorar >= 0.65 and p_dirigido < 0.75)
            or (p_ignorar >= 0.55 and p_dirigido < 0.65)
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
            # Se rota_principal tem altissima confianca de acao fisica e uma subacao clara aparece (>= 0.25),
            # escala para confirmacao de seguranca em vez de descartar silenciosamente.
            acao_probs = answers.get("acao_rover", {}).get("probabilities") or {}
            acoes_validas = {k: float(v) for k, v in acao_probs.items() if k != "nenhuma"}
            if acoes_validas and rota_prob >= 0.90 and p_dirigido >= 0.65:
                sub_acao = max(acoes_validas, key=acoes_validas.get)
                sub_prob = acoes_validas[sub_acao]
                if sub_prob >= 0.25:
                    return DecisaoRoteador(
                        rota_final="acao_rover",
                        subtipo=sub_acao,
                        confianca=sub_prob,
                        dirigido_ao_robo=p_dirigido,
                        requer_confirmacao=True,
                        precisa_imagem_camera=False,
                        prob_imagem_camera=p_imagem,
                        latencia_ms=latencia_ms,
                        respostas_brutas=answers,
                    )

            # Se acao_rover afirmou 'nenhuma', evita acionar motores por falso positivo em rota_principal
            outras = {
                k: float(v)
                for k, v in rota_probs.items()
                if k != "acao_rover"
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
