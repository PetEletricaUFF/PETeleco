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

import base64
from dataclasses import dataclass
from datetime import datetime
import json
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
COSINE_THRESHOLD_SFACE = 0.363


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
        self._face_init_tentado = False
        self._yunet_detector: Any = None
        self._sface_recognizer: Any = None
        self._known_faces: Dict[str, Any] = {}

    def _inicializar_biometria_facial(self) -> bool:
        """Carrega YuNet + SFace e known_faces sob demanda sem travar quem roda sem OpenCV."""
        if self._face_init_tentado:
            return self._yunet_detector is not None and self._sface_recognizer is not None
        self._face_init_tentado = True

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return False

        yunet_cands = [
            ROOT_DIR / "models" / "face_detection_yunet_2023mar.onnx",
            ROOT_DIR.parent / "rover" / "tools" / "models" / "face_detection_yunet_2023mar.onnx",
        ]
        sface_cands = [
            ROOT_DIR / "models" / "face_recognition_sface_2021dec.onnx",
            ROOT_DIR.parent / "rover" / "tools" / "models" / "face_recognition_sface_2021dec.onnx",
        ]
        db_cands = [
            ROOT_DIR / "data" / "known_faces.json",
            ROOT_DIR.parent / "rover" / "tools" / "known_faces.json",
        ]

        yunet_path = next((p for p in yunet_cands if p.exists()), None)
        sface_path = next((p for p in sface_cands if p.exists()), None)
        db_path = next((p for p in db_cands if p.exists()), None)
        if not (yunet_path and sface_path and db_path):
            return False

        try:
            self._yunet_detector = cv2.FaceDetectorYN.create(
                str(yunet_path), "", (320, 320), score_threshold=0.60, nms_threshold=0.30, top_k=10
            )
            self._sface_recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
            raw = json.loads(db_path.read_text(encoding="utf-8"))
            self._known_faces = {
                str(nome): np.array([emb], dtype=np.float32)
                for nome, emb in raw.items()
                if isinstance(emb, list) and len(emb) == 128
            }
            return bool(self._known_faces)
        except Exception:
            self._yunet_detector = None
            self._sface_recognizer = None
            self._known_faces = {}
            return False

    def reconhecer_rosto_imagem_b64(self, imagem_jpeg_b64: str) -> str:
        """Executa YuNet + SFace em uma imagem JPEG base64 e devolve 'nome (XX%)' ou 'nenhuma'."""
        if not imagem_jpeg_b64 or not self._inicializar_biometria_facial():
            return "nenhuma"
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            raw_b64 = imagem_jpeg_b64.split(",", 1)[-1].strip()
            buf = np.frombuffer(base64.b64decode(raw_b64), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return "nenhuma"

            h, w, _ = img.shape
            max_dim = max(w, h)
            if max_dim > 1024:
                scale = 960.0 / max_dim
                sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                img_small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                self._yunet_detector.setInputSize((sw, sh))
                _, faces = self._yunet_detector.detect(img_small)
                if faces is not None and len(faces) > 0:
                    faces = faces.copy()
                    faces[:, :14] /= scale
            else:
                self._yunet_detector.setInputSize((w, h))
                _, faces = self._yunet_detector.detect(img)

            if faces is None or len(faces) == 0:
                return "nenhuma"

            # Avalia a maior face detectada em primeiro plano
            maior_face = max(faces, key=lambda f: float(f[2] * f[3]))
            aligned = self._sface_recognizer.alignCrop(img, maior_face)
            feat = self._sface_recognizer.feature(aligned)

            melhor_nome = ""
            melhor_score = 0.0
            for nome, feat_db in self._known_faces.items():
                score = float(self._sface_recognizer.match(feat, feat_db, cv2.FaceRecognizerSF_FR_COSINE))
                if score > melhor_score:
                    melhor_score = score
                    if score >= COSINE_THRESHOLD_SFACE:
                        melhor_nome = nome

            if melhor_nome:
                return f"{melhor_nome} ({int(round(melhor_score * 100))}%)"
        except Exception:
            return "nenhuma"
        return "nenhuma"

    def _obter_historico_hidratado(self, pessoa_id: str) -> List[Dict[str, str]]:
        """Garante continuidade multi-turno mesmo entre execucoes/processos separados."""
        hist = self._historicos_por_pessoa.get(pessoa_id)
        if not hist:
            hist = self.memoria.obter_historico_dialogo(pessoa_id, limite_turnos=4)
            if hist:
                self._historicos_por_pessoa[pessoa_id] = hist
        return hist or []

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
        contexto_sala = self.memoria.obter_contexto_sala_recente(limite=3)
        contexto_turno = (
            "=== CONTEXTO DO TURNO (injetado dinamicamente) ===\n"
            f"MODO_PERSONA: {modo_efetivo}\n"
            f"PESSOA_ATUAL: {pessoa_display}\n"
            f"IDENTIFICACAO: {modo_identificacao}\n"
            f"PERFIL_PESSOA: {{{perfil_str}}}\n"
            f"MEMORIA_PESSOA: {memoria_str}\n"
            f"ESTADO_CORPORAL: {{{self.estado.formatar_prompt()}}}\n"
            f"TELEMETRIA: {self.estado.formatar_telemetria()}\n"
            f"ULTIMAS_FALAS_SALA: {contexto_sala}"
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
        palavrao = bool(perfil.get("libera_palavrao", False))

        if formal:
            tabela_formal = {
                "saudacao": f"Olá{', professor Felipe' if perfil.get('papel') == 'professor_orientador' else vocativo}. Estou pronto aqui na bancada, em que posso ajudar?",
                "despedida": "Até logo! Qualquer coisa é só chamar que eu sigo por aqui.",
                "ok_confirmacao": "Entendido, tudo certo por aqui.",
                "elogio_agradecimento": "Por nada! Estou à disposição quando precisar.",
            }
            return tabela_formal.get(subtipo, "Certo.")

        if nome == "desconhecido":
            tabela_desconhecido = {
                "saudacao": [
                    "Qual foi! Tô acordado aqui na bancada. E aí, tu é quem aqui do PET ou tá visitando a sala?",
                    "Opa, fala aí! Eu ainda não tenho tua cara nem tua voz salvas aqui. Qual é o teu nome?",
                ],
                "despedida": [
                    "Valeu, até a próxima! Da próxima vez passa teu nome pra eu já te reconhecer na entrada.",
                    "Falou! Se passar aqui pelo PET de novo dá um salve.",
                ],
            }
            if subtipo in tabela_desconhecido:
                return random.choice(tabela_desconhecido[subtipo])

        tabela_normal = {
            "saudacao": [
                f"Qual foi{vocativo}! Tô na escuta aqui na bancada. Tu tá aprontando o quê hoje?",
                f"Fala{vocativo}, tranquilo? Tô ligado com a câmera aberta aqui, manda a boa!",
                f"Opa{vocativo}, cheguei! Tava só olhando a sala esperando alguém aparecer pra trocar ideia. Que que pega?",
            ],
            "despedida": [
                f"Valeu{vocativo}, até depois! Não vai largar o ferro de solda ligado antes de sair, hein.",
                "Falou! Se precisar de ajuda na bancada ou quiser testar alguma parada minha depois, só chamar.",
            ],
            "ok_confirmacao": [
                "Beleza, fechado! Qualquer outra parada só falar.",
                "Show, entendi certinho. Tô aqui aguardando o próximo BO.",
            ],
            "elogio_agradecimento": [
                f"Tamo junto{vocativo}! Pelo menos alguém nessa sala reconhece quando eu mando bem.",
                f"Tranquilo{vocativo}, é pra isso que me botaram nessa placa aqui! Precisando, só mandar.",
            ],
        }
        if palavrao:
            tabela_normal["saudacao"].append(
                f"Aí sim{vocativo}! Finalmente alguém falou comigo, porra. E aí, qual é a missão de hoje na bancada?"
            )
            tabela_normal["elogio_agradecimento"].append(
                f"Que isso{vocativo}, tô brabo demais hoje, né? Valeu, pô, manda a próxima!"
            )
        opcoes = tabela_normal.get(subtipo, ["Beleza, tô acompanhando aqui!"])
        return random.choice(opcoes)

    def _responder_local_parametrizada(
        self,
        subtipo: str,
        perfil: Dict[str, Any],
        pessoa_display: str,
        modo_identificacao: str,
    ) -> str:
        formal = self.eh_modo_formal(perfil)
        nome = perfil.get("nome", "")
        vocativo = f", {nome}" if nome and nome != "desconhecido" else ""
        agora = datetime.now()

        if subtipo == "hora_atual":
            h_str = agora.strftime("%H horas e %M minutos")
            if formal:
                return f"Agora são {h_str}."
            return f"Agora são {h_str}{vocativo}. Tá correndo contra algum prazo aí ou tá só matando tempo na sala?"

        if subtipo == "bateria_temp":
            v = f"{self.estado.bateria_v:.1f}".replace(".", " vírgula ")
            t = f"{self.estado.temperatura_c:.0f}"
            if formal:
                return f"A bateria está com {v} volts e a temperatura estimada é de {t} graus."
            return f"Tô com {v} volts na bateria e rodando de boa a {t} graus. Por enquanto não vou pegar fogo não!"

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
        """Executa o turno completo (biometria facial -> identificacao -> travas -> rota -> memoria)."""
        face_id_efetivo = (face_id or "nenhuma").strip()
        if imagem_jpeg_b64 and normalizar_chave(face_id_efetivo) in (
            "",
            "nenhuma",
            "nenhum",
            "auto",
            "none",
            "desconhecido",
        ):
            face_detectada = self.reconhecer_rosto_imagem_b64(imagem_jpeg_b64)
            if face_detectada != "nenhuma":
                face_id_efetivo = face_detectada

        pessoa_id, perfil, modo_id, pessoa_display = self.memoria.resolver_pessoa(
            transcricao=transcricao,
            face_id=face_id_efetivo,
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
        hist_pessoa = self._obter_historico_hidratado(pessoa_id)
        for item in hist_pessoa[-6:]:
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
        hist_pessoa = self._obter_historico_hidratado(pessoa_id)
        for item in hist_pessoa[-6:]:
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
