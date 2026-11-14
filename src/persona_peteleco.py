#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 3/4 — Motor de Persona, Consciencia Corporal e Orquestracao Cognitiva.

Carrega o System Prompt v3.1 diretamente de `prompts/persona_peteleco.md`,
aplica travas fisicas de seguranca corporal (`EstadoCorporal`), resolve rotas
locais (`pool_fala_pronta`, `fala_parametrizada`, `acao_rover`) adaptadas ao
perfil da pessoa e chama a API REST do Gemini na rota `cognicao_gemini` (com
normalizacao fonetica pronta para o Piper TTS).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.memoria_pessoas import (
    GerenciadorMemoriaPessoas,
    ROOT_DIR,
    carregar_env,
    normalizar_chave,
)
from src.router_systemone import DecisaoRoteador


DEFAULT_PROMPT_PATH = ROOT_DIR / "prompts" / "persona_peteleco.md"


@dataclass
class EstadoCorporal:
    """Snapshot vivo do corpo fisico do rover (altura, movimento e sensores)."""

    superficie: str = "chao"  # chao | bancada | no_colo
    movimento: str = "parado"  # parado | andando | seguindo_rosto | radar
    pan_graus: float = 90.0
    tilt_graus: float = 90.0
    bateria_v: float = 7.9
    temperatura_c: float = 34.0
    objetos_yolo: str = "person"

    def formatar_prompt(self) -> str:
        return (
            f"superficie={self.superficie}, movimento={self.movimento}, "
            f"pan_tilt=({int(round(self.pan_graus))}, {int(round(self.tilt_graus))})"
        )

    def formatar_telemetria(self) -> str:
        return (
            f"bateria {self.bateria_v:.1f}V, temp {self.temperatura_c:.0f}C, "
            f"objetos: {self.objetos_yolo}"
        )


def carregar_system_prompt_base(prompt_path: Optional[Path] = None) -> str:
    """Extrai o bloco System Prompt v3.1 do Markdown para manter fonte unica."""
    path = Path(prompt_path or DEFAULT_PROMPT_PATH)
    if not path.exists():
        return "Você é o PETeleco, o robô assistente do PET Elétrica da UFF."
    conteudo = path.read_text(encoding="utf-8")
    m = re.search(r"## 3\. System Prompt[^\n]*\n+```text\n(.*?)\n```", conteudo, flags=re.DOTALL)
    if not m:
        return "Você é o PETeleco, o robô assistente do PET Elétrica da UFF."
    bloco = m.group(1)
    marcador = "=== CONTEXTO DO TURNO"
    if marcador in bloco:
        bloco = bloco.split(marcador, 1)[0].rstrip()
    return bloco


def normalizar_para_tts(texto: str) -> str:
    """Limpa markdown e converte siglas/unidades para leitura fluida no Piper TTS."""
    limpo = re.sub(r"[*_`#]", "", texto or "")
    limpo = re.sub(r"(\d+)[.,](\d+)\s*V\b", r"\1 vírgula \2 volts", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"(\d+)\s*V\b", r"\1 volts", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"(\d+)\s*°?\s*C\b", r"\1 graus", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"(\d+)\s*mm\b", r"\1 milímetros", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"(\d+)\s*mAh\b", r"\1 miliampere-hora", limpo, flags=re.IGNORECASE)

    siglas = {
        r"\bUFF\b": "Úfi",
        r"\bPID\b": "Pê I Dê",
        r"\bPWM\b": "Pê Dáblio Ême",
        r"\bI2S\b": "I Dois Ésse",
        r"\bOLED\b": "Oléd",
        r"\bUSB\b": "U Ésse Bê",
        r"\bPCB\b": "Pê Cê Bê",
    }
    for padrao, troca in siglas.items():
        limpo = re.sub(padrao, troca, limpo)
    return re.sub(r"\s+", " ", limpo).strip()


class MotorPersonaPETeleco:
    """Orquestra persona adaptativa por pessoa, estado corporal e chamadas LLM."""

    def __init__(
        self,
        memoria: Optional[GerenciadorMemoriaPessoas] = None,
        estado: Optional[EstadoCorporal] = None,
        prompt_path: Optional[Path] = None,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.memoria = memoria or GerenciadorMemoriaPessoas()
        self.estado = estado or EstadoCorporal()
        self.prompt_base = carregar_system_prompt_base(prompt_path)
        self.ollama_url = ollama_url.rstrip("/")
        self.modo_persona: str = "NORMAL"  # NORMAL | SERIO
        self._env = carregar_env()
        self._session = requests.Session()
        self._historicos_por_pessoa: Dict[str, List[Dict[str, str]]] = {}

    def eh_modo_formal(self, perfil: Dict[str, Any]) -> bool:
        """Retorna True se o turno exigir postura estritamente formal (SERIO ou Orientador)."""
        if self.modo_persona == "SERIO":
            return True
        if perfil.get("papel") == "professor_orientador":
            return True
        return not bool(perfil.get("libera_palavrao", False)) and not bool(
            perfil.get("libera_zoeira_pesada", False)
        )

    def _montar_system_instruction(
        self,
        pessoa_id: str,
        pessoa_display: str,
        modo_identificacao: str,
        perfil: Dict[str, Any],
    ) -> str:
        perfil_str, memoria_str = self.memoria.formatar_contexto_prompt(pessoa_id)
        modo_efetivo = "SERIO" if self.eh_modo_formal(perfil) else self.modo_persona
        contexto_turno = (
            "=== CONTEXTO DO TURNO (injetado dinamicamente) ===\n"
            f"MODO_PERSONA: {modo_efetivo}\n"
            f"PESSOA_ATUAL: {pessoa_display}\n"
            f"IDENTIFICACAO: {modo_identificacao}\n"
            f"PERFIL_PESSOA: {{{perfil_str}}}\n"
            f"MEMORIA_PESSOA: {memoria_str}\n"
            f"ESTADO_CORPORAL: {{{self.estado.formatar_prompt()}}}\n"
            f"TELEMETRIA: {self.estado.formatar_telemetria()}"
        )
        return f"{self.prompt_base}\n\n{contexto_turno}"

    def _verificar_interceptadores_fisicos_e_modo(
        self,
        transcricao: str,
        perfil: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        """Trata troca de modo e avisos de manuseio fisico imediato antes de acionar motores/nuvem."""
        norm = normalizar_chave(transcricao)
        formal = self.eh_modo_formal(perfil)
        nome = perfil.get("nome", "")

        if any(f in norm for f in ("entra no modo serio", "modo serio", "fica serio")):
            self.modo_persona = "SERIO"
            return "Modo sério ativado.", "local:troca_modo_serio"

        if any(f in norm for f in ("voltar ao normal", "volta ao normal", "modo normal", "sai do modo serio")):
            self.modo_persona = "NORMAL"
            if perfil.get("papel") == "professor_orientador":
                return "Certo, professor Felipe. Sigo à disposição.", "local:troca_modo_normal"
            return "Beleza, voltei ao normal!", "local:troca_modo_normal"

        # Aviso de manuseio / pegada no colo -> trava rodas na hora
        if any(
            f in norm
            for f in (
                "vou te pegar",
                "vou te levantar",
                "to te pegando",
                "estou te pegando",
                "fica quieto que vou te pegar",
                "fica parado que vou te pegar",
                "segurar voce",
            )
        ):
            self.estado.movimento = "parado"
            self.estado.superficie = "no_colo"
            if formal:
                return (
                    "Motores travados. Pode me levantar e direcionar a câmera para o que deseja mostrar.",
                    "local:manuseio_pegar",
                )
            return (
                "Beleza, já parei as rodas aqui, pode pegar. Aponta minha câmera pro lado certo que eu te aviso quando tiver vendo!",
                "local:manuseio_pegar",
            )

        if any(
            f in norm
            for f in (
                "te botei na mesa",
                "te coloquei na mesa",
                "te botei na bancada",
                "te coloquei na bancada",
                "ta em cima da bancada",
                "ta na bancada",
            )
        ):
            self.estado.movimento = "parado"
            self.estado.superficie = "bancada"
            if formal:
                return "Certo, estou na bancada com as rodas paradas.", "local:superficie_bancada"
            return "Firmeza, tô em cima da bancada aqui com as rodas travadas pra não despencar.", "local:superficie_bancada"

        if any(
            f in norm
            for f in (
                "te botei no chao",
                "te coloquei no chao",
                "ta no chao",
                "volta pro chao",
            )
        ):
            self.estado.movimento = "parado"
            self.estado.superficie = "chao"
            if formal:
                return f"Entendido{', ' + nome if nome and nome != 'desconhecido' else ''}. Estou no chão e pronto.", "local:superficie_chao"
            return "Boa, de volta no chão. Aqui já dá pra rodar tranquilo se precisar.", "local:superficie_chao"

        return None

    def _responder_local_pool(self, subtipo: str, perfil: Dict[str, Any]) -> str:
        formal = self.eh_modo_formal(perfil)
        nome = perfil.get("nome", "")
        vocativo = f", {nome}" if nome and nome != "desconhecido" else ""

        if formal:
            tabela_formal = {
                "saudacao": f"Olá{', professor Felipe' if perfil.get('papel') == 'professor_orientador' else vocativo}. Estou pronto para ajudar.",
                "despedida": "Até logo. Qualquer coisa estou por aqui.",
                "ok_confirmacao": "Entendido.",
                "elogio_agradecimento": "Por nada, estou à disposição.",
            }
            return tabela_formal.get(subtipo, "Certo.")

        tabela_normal = {
            "saudacao": [
                f"Qual foi{vocativo}! Tô na escuta.",
                f"Fala{vocativo}, tranquilo? Manda a boa!",
                "Opa, tô ligado aqui. Que que pega?",
            ],
            "despedida": [
                f"Valeu{vocativo}, até depois!",
                "Falou! Se precisar de alguma parada na bancada eu tô por aqui.",
            ],
            "ok_confirmacao": [
                "Beleza, fechado.",
                "Show, entendi.",
            ],
            "elogio_agradecimento": [
                "Tamo junto! Pelo menos alguém aqui reconhece meu trampo.",
                f"Tranquilo{vocativo}, é pra isso que eu tô aqui!",
            ],
        }
        opcoes = tabela_normal.get(subtipo, ["Beleza."])
        return random.choice(opcoes)

    def _responder_local_parametrizada(
        self,
        subtipo: str,
        perfil: Dict[str, Any],
        pessoa_display: str,
        modo_identificacao: str,
    ) -> str:
        formal = self.eh_modo_formal(perfil)
        agora = datetime.now()

        if subtipo == "hora_atual":
            h_str = agora.strftime("%H horas e %M minutos")
            if formal:
                return f"Agora são {h_str}."
            return f"Agora são {h_str}."

        if subtipo == "bateria_temp":
            v = f"{self.estado.bateria_v:.1f}".replace(".", " vírgula ")
            t = f"{self.estado.temperatura_c:.0f}"
            if formal:
                return f"A bateria está com {v} volts e a temperatura estimada é de {t} graus."
            return f"Tô com {v} volts na bateria e rodando tranquilo a {t} graus."

        if subtipo == "previsao_clima":
            if formal:
                return "Ainda não conectei o serviço local de meteorologia aqui no computador para consultar a previsão."
            return "Pô, ainda não ligaram a API do clima aqui em mim, tô preso dentro do laboratório sem saber se tá chovendo lá fora."

        if subtipo == "quem_vejo_agora":
            nome = perfil.get("nome", "desconhecido")
            if nome == "desconhecido":
                return "Ainda não identifiquei você nem pelo rosto nem pela voz. Qual é o seu nome?"
            if modo_identificacao == "apenas_voz":
                if formal:
                    return f"Reconheci sua voz como {nome}, porém você não está no campo de visão da minha câmera."
                return f"Tô ouvindo que é tu, {nome}, pela tua voz, mas tu não tá na frente da minha câmera agora não!"
            if formal:
                return f"Estou vendo {nome}."
            return f"Tô vendo que é tu, {nome}!"

        return f"Identificação atual: {pessoa_display}."

    def _responder_acao_rover(
        self,
        decisao: DecisaoRoteador,
        perfil: Dict[str, Any],
    ) -> str:
        subtipo = decisao.subtipo
        formal = self.eh_modo_formal(perfil)

        if subtipo == "parar":
            self.estado.movimento = "parado"
            return "Motores parados." if formal else "Parei tudo aqui!"

        # Trava fisica obrigatoria: nunca anda solto em cima da bancada ou no colo
        if subtipo in ("seguir_rosto", "ir_ate_objeto") and self.estado.superficie in ("bancada", "no_colo"):
            self.estado.movimento = "parado"
            onde = "em cima da bancada" if self.estado.superficie == "bancada" else "na tua mão"
            if formal:
                return f"Não posso acionar as rodas enquanto estiver {onde}, pois não tenho sensor de borda. Por favor, coloque-me no chão primeiro."
            return f"Tá maluco? Eu tô {onde} e nem tenho sensor de borda ainda, se eu andar aqui eu vou pro chão me quebrar todo. Me bota no chão primeiro!"

        if decisao.requer_confirmacao:
            alvo = "te seguir" if subtipo == "seguir_rosto" else "ir até o objeto"
            if formal:
                return f"Confirma que deseja que eu execute a movimentação para {alvo}?"
            return f"Tô sem sensor de distância ainda, só na câmera. Tu confirma que eu posso {alvo}?"

        if subtipo == "seguir_rosto":
            self.estado.movimento = "seguindo_rosto"
            return "Seguindo seu rosto." if formal else "Bora, já tô te seguindo aqui."
        if subtipo == "ir_ate_objeto":
            self.estado.movimento = "andando"
            return "Indo em direção ao objeto." if formal else "Fechado, tô indo lá perto."
        if subtipo == "radar":
            self.estado.movimento = "radar"
            return "Iniciando varredura com a câmera." if formal else "Deixa comigo, tô girando a cabeça pra procurar aqui."
        return "Certo."

    def responder(
        self,
        transcricao: str,
        decisao_jev: Optional[DecisaoRoteador] = None,
        face_id: str = "nenhuma",
        voice_id: str = "desconhecido",
        voice_conf: float = 0.0,
        imagem_jpeg_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executa o turno completo (identificacao -> travas -> rota -> memoria)."""
        pessoa_id, perfil, modo_id, pessoa_display = self.memoria.resolver_pessoa(
            transcricao=transcricao,
            face_id=face_id,
            voice_id=voice_id,
            voice_conf=voice_conf,
        )

        # 1) Interceptadores fisicos/modo tem prioridade imediata
        inter = self._verificar_interceptadores_fisicos_e_modo(transcricao, perfil)
        if inter is not None:
            resposta_texto, origem = inter
        elif decisao_jev is not None:
            if decisao_jev.rota_final == "ignorar":
                return {
                    "resposta": "",
                    "resposta_tts": "",
                    "origem": "ignorar",
                    "pessoa_id": pessoa_id,
                    "pessoa_display": pessoa_display,
                    "modo_identificacao": modo_id,
                    "modo_persona": "SERIO" if self.eh_modo_formal(perfil) else self.modo_persona,
                    "estado_corporal": self.estado.formatar_prompt(),
                }
            if decisao_jev.rota_final == "pool_fala_pronta":
                resposta_texto = self._responder_local_pool(decisao_jev.subtipo, perfil)
                origem = f"local:pool:{decisao_jev.subtipo}"
            elif decisao_jev.rota_final == "fala_parametrizada":
                resposta_texto = self._responder_local_parametrizada(
                    decisao_jev.subtipo, perfil, pessoa_display, modo_id
                )
                origem = f"local:param:{decisao_jev.subtipo}"
            elif decisao_jev.rota_final == "acao_rover":
                resposta_texto = self._responder_acao_rover(decisao_jev, perfil)
                origem = f"local:acao:{decisao_jev.subtipo}"
            else:
                resposta_texto, origem = self._gerar_resposta_llm(
                    transcricao=transcricao,
                    pessoa_id=pessoa_id,
                    pessoa_display=pessoa_display,
                    modo_identificacao=modo_id,
                    perfil=perfil,
                    imagem_jpeg_b64=imagem_jpeg_b64,
                )
        else:
            resposta_texto, origem = self._gerar_resposta_llm(
                transcricao=transcricao,
                pessoa_id=pessoa_id,
                pessoa_display=pessoa_display,
                modo_identificacao=modo_id,
                perfil=perfil,
                imagem_jpeg_b64=imagem_jpeg_b64,
            )

        hist_pessoa = self._historicos_por_pessoa.setdefault(pessoa_id, [])
        hist_pessoa.append({"role": "user", "text": transcricao.strip()})
        hist_pessoa.append({"role": "model", "text": resposta_texto})
        self._historicos_por_pessoa[pessoa_id] = hist_pessoa[-6:]

        self.memoria.registrar_interacao(
            pessoa_id=pessoa_id,
            fala_usuario=transcricao,
            resposta_robo=resposta_texto,
        )

        return {
            "resposta": resposta_texto,
            "resposta_tts": normalizar_para_tts(resposta_texto),
            "origem": origem,
            "pessoa_id": pessoa_id,
            "pessoa_display": pessoa_display,
            "modo_identificacao": modo_id,
            "modo_persona": "SERIO" if self.eh_modo_formal(perfil) else self.modo_persona,
            "estado_corporal": self.estado.formatar_prompt(),
        }

    def _gerar_resposta_llm(
        self,
        transcricao: str,
        pessoa_id: str,
        pessoa_display: str,
        modo_identificacao: str,
        perfil: Dict[str, Any],
        imagem_jpeg_b64: Optional[str] = None,
    ) -> Tuple[str, str]:
        system_inst = self._montar_system_instruction(
            pessoa_id=pessoa_id,
            pessoa_display=pessoa_display,
            modo_identificacao=modo_identificacao,
            perfil=perfil,
        )

        api_key = self._env.get("GEMINI_API_KEY", "").strip()
        if api_key:
            texto, modelo_usado = self._chamar_gemini_rest(
                api_key=api_key,
                pessoa_id=pessoa_id,
                system_inst=system_inst,
                transcricao=transcricao,
                imagem_jpeg_b64=imagem_jpeg_b64,
            )
            if texto:
                return texto, f"gemini:{modelo_usado}"

        texto_ol, modelo_ol = self._chamar_ollama_chat_fallback(
            pessoa_id=pessoa_id,
            system_inst=system_inst,
            transcricao=transcricao,
        )
        if texto_ol:
            return texto_ol, f"ollama_chat:{modelo_ol}"

        return (
            "Não consegui conectar no Gemini nem achei modelo de chat local no Ollama. Confere sua GEMINI_API_KEY no .env!",
            "fallback_sem_llm",
        )

    def _chamar_gemini_rest(
        self,
        api_key: str,
        pessoa_id: str,
        system_inst: str,
        transcricao: str,
        imagem_jpeg_b64: Optional[str] = None,
    ) -> Tuple[str, str]:
        modelos: List[str] = []
        for cand in (
            self._env.get("GEMINI_PERSONA_MODEL", ""),
            "gemini-3.5-flash-lite",
            self._env.get("GEMINI_MODEL", ""),
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
        ):
            m = cand.strip()
            if m and m not in modelos:
                modelos.append(m)

        contents: List[Dict[str, Any]] = []
        hist_pessoa = self._historicos_por_pessoa.get(pessoa_id, [])
        for item in hist_pessoa[-4:]:
            contents.append(
                {
                    "role": "user" if item["role"] == "user" else "model",
                    "parts": [{"text": item["text"]}],
                }
            )
        partes_atual: List[Dict[str, Any]] = [{"text": transcricao.strip()}]
        if imagem_jpeg_b64:
            partes_atual.append({"inlineData": {"mimeType": "image/jpeg", "data": imagem_jpeg_b64}})
        contents.append({"role": "user", "parts": partes_atual})

        payload = {
            "systemInstruction": {"parts": [{"text": system_inst}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.75, "maxOutputTokens": 1024},
        }

        for modelo in modelos:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={api_key}"
            try:
                resp = self._session.post(url, json=payload, timeout=25.0)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
                cands = data.get("candidates") or []
                if not cands:
                    continue
                partes = cands[0].get("content", {}).get("parts", [])
                texto = " ".join(
                    str(p.get("text", "")).strip()
                    for p in partes
                    if not p.get("thought") and p.get("text")
                ).strip()
                if texto:
                    return texto, modelo
            except Exception:
                continue
        return "", ""

    def _chamar_ollama_chat_fallback(
        self,
        pessoa_id: str,
        system_inst: str,
        transcricao: str,
    ) -> Tuple[str, str]:
        """Fallback local caso exista modelo de chat (nao `tev1`, que e exclusivo do SystemOne)."""
        try:
            tags_resp = self._session.get(f"{self.ollama_url}/api/tags", timeout=2.0)
            tags_resp.raise_for_status()
            modelos = [
                str(m.get("name", ""))
                for m in tags_resp.json().get("models", [])
                if not str(m.get("name", "")).startswith("tev1")
            ]
        except Exception:
            return "", ""

        preferido = self._env.get("OLLAMA_CHAT_MODEL", "").strip()
        if preferido and preferido in modelos:
            escolhido = preferido
        elif modelos:
            escolhido = modelos[0]
        else:
            return "", ""

        msgs = [{"role": "system", "content": system_inst}]
        hist_pessoa = self._historicos_por_pessoa.get(pessoa_id, [])
        for item in hist_pessoa[-4:]:
            msgs.append(
                {
                    "role": "user" if item["role"] == "user" else "assistant",
                    "content": item["text"],
                }
            )
        msgs.append({"role": "user", "content": transcricao.strip()})

        try:
            resp = self._session.post(
                f"{self.ollama_url}/api/chat",
                json={"model": escolhido, "messages": msgs, "stream": False},
                timeout=30.0,
            )
            resp.raise_for_status()
            texto = str(resp.json().get("message", {}).get("content", "")).strip()
            return texto, escolhido
        except Exception:
            return "", ""
