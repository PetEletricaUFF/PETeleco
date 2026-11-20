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
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.cerebros import CerebroSpec, GerenciadorCerebros
from src.grafo_memoria import GrafoMemoriaSQLite
from src.memoria_pessoas import (
    GerenciadorMemoriaPessoas,
    ROOT_DIR,
    carregar_env,
    normalizar_chave,
)
from src.router_systemone import DEFAULT_OLLAMA_URL, DecisaoRoteador, normalizar_ollama_url
from src.skills import ContextoSkill, GerenciadorSkills, ResultadoSkill, criar_gerenciador_skills


os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


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
    """Limpa markdown, risadas escritas, underscores e siglas/unidades eletricas para TTS fluido."""
    limpo = re.sub(r"_+", " ", texto or "")
    limpo = re.sub(r"[*`#]", "", limpo)
    # Remove emojis / simbolos fora do bloco basico latino para nao atrapalhar prosodia
    limpo = re.sub(r"[\U00010000-\U0010FFFF]", "", limpo)
    # Converte risadas de chat ("kkk", "kkkkk", "rsrs") em riso pronunciavel para TTS neural
    limpo = re.sub(r"\bk{3,}\b", "há há há", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"\b(?:rs){2,}\b", "hehe", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"\bha(?:ha){2,}\b", "há há há", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"\bhaha\b", "há há", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"\bhehe\b", "hê hê", limpo, flags=re.IGNORECASE)
    limpo = re.sub(r"\.{3,}", ", ", limpo)

    # Moeda brasileira ("R$1,99" -> "1 real e 99")
    limpo = re.sub(
        r"R\$\s*(\d+)[.,](\d+)",
        lambda m: (
            f"{m.group(1)} {'real' if int(m.group(1)) == 1 else 'reais'} e {m.group(2)}"
            if int(m.group(2)) > 0
            else f"{m.group(1)} {'real' if int(m.group(1)) == 1 else 'reais'}"
        ),
        limpo,
    )
    limpo = re.sub(
        r"R\$\s*(\d+)",
        lambda m: f"{m.group(1)} {'real' if int(m.group(1)) == 1 else 'reais'}",
        limpo,
    )

    # Siglas de engenharia faladas antes das unidades para evitar colisoes (ex.: I2C != 2 graus)
    siglas = {
        r"\bUFF\b": "Úfi",
        r"\bPID\b": "Pê I Dê",
        r"\bPWM\b": "Pê Dáblio Ême",
        r"\bI2S\b": "I Dois Ésse",
        r"\bI2C\b": "I Dois Cê",
        r"\bADC\b": "A Dê Cê",
        r"\bGPIO\b": "Gê Pê I Ó",
        r"\bUART\b": "Ú art",
        r"\bUDP\b": "U Dê Pê",
        r"\bOLED\b": "Oléd",
        r"\bUSB\b": "U Ésse Bê",
        r"\bPCB\b": "Pê Cê Bê",
    }
    for padrao, troca in siglas.items():
        limpo = re.sub(padrao, troca, limpo)

    # Notacoes compactas de bancada ("3V3", "4k7", "4R7")
    limpo = re.sub(r"(?<![A-Za-z])(\d+)[Vv](\d+)\b", r"\1 vírgula \2 volts", limpo)
    limpo = re.sub(r"(?<![A-Za-z])(\d+)[kK](\d+)\b", r"\1 vírgula \2 quilo-ohms", limpo)
    limpo = re.sub(r"(?<![A-Za-z])(\d+)[rR](\d+)\b", r"\1 vírgula \2 ohms", limpo)

    # Unidades com casas decimais ou inteiras (prefixo (?<![A-Za-z]) evita pegar I2C/MAX98357A)
    unidades = [
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*(?:kΩ|kohms?|k)\b", r"\1 vírgula \2 quilo-ohms", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*(?:kΩ|kohms?|k)\b", r"\1 quilo-ohms", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*(?:MΩ|Mohms?)\b", r"\1 vírgula \2 megaohms", 0),
        (r"(?<![A-Za-z])(\d+)\s*(?:MΩ|Mohms?)\b", r"\1 megaohms", 0),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*(?:Ω|ohms?)\b", r"\1 vírgula \2 ohms", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*(?:Ω|ohms?)\b", r"\1 ohms", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*[uµ]F\b", r"\1 vírgula \2 microfarads", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*[uµ]F\b", r"\1 microfarads", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*nF\b", r"\1 vírgula \2 nanofarads", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*nF\b", r"\1 nanofarads", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*pF\b", r"\1 vírgula \2 picofarads", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*pF\b", r"\1 picofarads", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*mAh\b", r"\1 vírgula \2 miliampere-hora", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*mAh\b", r"\1 miliampere-hora", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*mA\b", r"\1 vírgula \2 miliamperes", 0),
        (r"(?<![A-Za-z])(\d+)\s*mA\b", r"\1 miliamperes", 0),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*A\b", r"\1 vírgula \2 amperes", 0),
        (r"(?<![A-Za-z])(\d+)\s*A\b", r"\1 amperes", 0),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*MHz\b", r"\1 vírgula \2 megahertz", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*MHz\b", r"\1 megahertz", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*kHz\b", r"\1 vírgula \2 quilohertz", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*kHz\b", r"\1 quilohertz", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*Hz\b", r"\1 vírgula \2 hertz", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*Hz\b", r"\1 hertz", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*ms\b", r"\1 vírgula \2 milissegundos", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*ms\b", r"\1 milissegundos", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*[uµ]s\b", r"\1 vírgula \2 microssegundos", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*[uµ]s\b", r"\1 microssegundos", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*V\b", r"\1 vírgula \2 volts", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*V\b", r"\1 volts", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*°?\s*C\b", r"\1 vírgula \2 graus", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*°?\s*C\b", r"\1 graus", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)[.,](\d+)\s*mm\b", r"\1 vírgula \2 milímetros", re.IGNORECASE),
        (r"(?<![A-Za-z])(\d+)\s*mm\b", r"\1 milímetros", re.IGNORECASE),
    ]
    for pat, repl, flg in unidades:
        limpo = re.sub(pat, repl, limpo, flags=flg)

    return re.sub(r"\s+", " ", limpo).strip()


_URL_YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_URL_SFACE = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

_GATILHOS_VISUAIS = (
    "olha",
    "olhando",
    "ta vendo",
    "tá vendo",
    "esta vendo",
    "está vendo",
    "consegue ver",
    "da pra ver",
    "dá pra ver",
    "meu corte",
    "corte de cabelo",
    "na regua",
    "na régua",
    "minha cara",
    "meu rosto",
    "minha roupa",
    "minha camisa",
    "minha mao",
    "minha mão",
    "segurando",
    "mostrando",
    "essa placa",
    "esse circuito",
    "esse componente",
    "esse codigo",
    "esse código",
    "na minha frente",
    "aqui na frente",
    "na bancada",
    "o que tem aqui",
)


def parece_pedido_visual(transcricao: str) -> bool:
    """Detecta pedidos que exigem olhar a camera mesmo quando o roteador local fica dividido."""
    norm = normalizar_chave(transcricao)
    if not norm:
        return False
    return any(normalizar_chave(g) in norm for g in _GATILHOS_VISUAIS)


class MotorPersonaPETeleco:
    """Orquestra persona adaptativa por pessoa, estado corporal e chamadas LLM."""

    def __init__(
        self,
        memoria: Optional[GerenciadorMemoriaPessoas] = None,
        estado: Optional[EstadoCorporal] = None,
        prompt_path: Optional[Path] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        cerebros: Optional[GerenciadorCerebros] = None,
        skills: Optional[GerenciadorSkills] = None,
        grafo: Optional[GrafoMemoriaSQLite] = None,
        usar_multi_cerebro: bool = True,
    ) -> None:
        self.memoria = memoria or GerenciadorMemoriaPessoas()
        self.estado = estado or EstadoCorporal()
        self.prompt_base = carregar_system_prompt_base(prompt_path)
        self.cerebros = cerebros or GerenciadorCerebros()
        self.grafo = grafo or GrafoMemoriaSQLite(sqlite_path=self.memoria.sqlite_path)
        self.skills = skills or criar_gerenciador_skills(
            sqlite_path=self.memoria.sqlite_path,
            grafo=self.grafo,
        )
        self.usar_multi_cerebro = bool(usar_multi_cerebro)
        self.cerebro_forcado: str = ""
        self.ollama_url = normalizar_ollama_url(ollama_url)
        self.modo_persona: str = "NORMAL"  # NORMAL | SERIO | DESGRACADO
        self._env = carregar_env()
        self._session = requests.Session()
        self._historicos_por_pessoa: Dict[str, List[Dict[str, str]]] = {}
        self._face_init_tentado = False
        self._yunet_detector: Any = None
        self._sface_recognizer: Any = None
        self._known_faces: Dict[str, Any] = {}
        self._known_faces_path: Path = ROOT_DIR / "data" / "known_faces.json"

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
        if not yunet_path:
            yunet_path = self._baixar_modelo_onnx(yunet_cands[0], _URL_YUNET)
        if not sface_path:
            sface_path = self._baixar_modelo_onnx(sface_cands[0], _URL_SFACE)
        if not (yunet_path and sface_path):
            return False

        try:
            self._yunet_detector = cv2.FaceDetectorYN.create(
                str(yunet_path), "", (320, 320), score_threshold=0.60, nms_threshold=0.30, top_k=10
            )
            self._sface_recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
            if db_path:
                self._known_faces_path = db_path
                raw = json.loads(db_path.read_text(encoding="utf-8"))
                self._known_faces = {
                    str(nome): np.array([emb], dtype=np.float32)
                    for nome, emb in raw.items()
                    if isinstance(emb, list) and len(emb) == 128
                }
            return True
        except Exception:
            self._yunet_detector = None
            self._sface_recognizer = None
            self._known_faces = {}
            return False

    def _baixar_modelo_onnx(self, destino: Path, url: str) -> Optional[Path]:
        """Baixa modelos oficiais YuNet/SFace para `models/` se nao houver copia local."""
        try:
            destino.parent.mkdir(parents=True, exist_ok=True)
            tmp = destino.with_suffix(".onnx.tmp")
            print(f"[VISAO] Baixando modelo facial {destino.name} para {destino.parent}...")
            with self._session.get(url, stream=True, timeout=60.0, allow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < 10_000:
                tmp.unlink(missing_ok=True)
                return None
            tmp.replace(destino)
            print(f"[VISAO] Modelo {destino.name} pronto ({destino.stat().st_size // 1024} KB).")
            return destino
        except Exception:
            return None

    @staticmethod
    def _decodificar_imagem_b64(imagem_jpeg_b64: str) -> Any:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return None
        if not imagem_jpeg_b64:
            return None
        try:
            raw_b64 = imagem_jpeg_b64.split(",", 1)[-1].strip()
            buf = np.frombuffer(base64.b64decode(raw_b64), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def detectar_e_reconhecer_faces(
        self,
        img: Any,
        incluir_embedding: bool = False,
    ) -> List[Dict[str, Any]]:
        """Executa YuNet + SFace em frame BGR OpenCV e retorna lista ordenada por area."""
        if img is None or not self._inicializar_biometria_facial():
            return []
        try:
            import cv2  # type: ignore

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
                return []

            resultados: List[Dict[str, Any]] = []
            faces_ord = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)
            for face in faces_ord:
                aligned = self._sface_recognizer.alignCrop(img, face)
                feat = self._sface_recognizer.feature(aligned)

                melhor_nome = ""
                melhor_cand = ""
                melhor_score = 0.0
                for nome, feat_db in self._known_faces.items():
                    score = float(self._sface_recognizer.match(feat, feat_db, cv2.FaceRecognizerSF_FR_COSINE))
                    if score > melhor_score:
                        melhor_score = score
                        melhor_cand = nome
                        if score >= COSINE_THRESHOLD_SFACE:
                            melhor_nome = nome

                x, y, bw, bh = [int(round(float(v))) for v in face[:4]]
                landmarks = [
                    (int(round(float(face[idx]))), int(round(float(face[idx + 1]))))
                    for idx in range(4, 14, 2)
                ] if len(face) >= 14 else []
                det_score = float(face[14]) if len(face) > 14 else 1.0
                label = f"{melhor_nome} ({int(round(melhor_score * 100))}%)" if melhor_nome else "desconhecido"
                item: Dict[str, Any] = {
                    "bbox": (x, y, bw, bh),
                    "landmarks": landmarks,
                    "det_score": round(det_score, 3),
                    "nome": melhor_nome or "desconhecido",
                    "candidato": melhor_cand or "",
                    "score": round(melhor_score, 4),
                    "limiar": COSINE_THRESHOLD_SFACE,
                    "label": label,
                }
                if incluir_embedding:
                    item["feat"] = feat
                resultados.append(item)
            return resultados
        except Exception:
            return []

    def listar_rostos_conhecidos(self) -> List[str]:
        """Retorna chaves cadastradas na base SFace carregada."""
        self._inicializar_biometria_facial()
        return sorted(self._known_faces.keys())

    def reconhecer_rosto_imagem_b64(self, imagem_jpeg_b64: str) -> str:
        """Executa YuNet + SFace em uma imagem JPEG base64 e devolve 'nome (XX%)' ou 'nenhuma'."""
        img = self._decodificar_imagem_b64(imagem_jpeg_b64)
        faces = self.detectar_e_reconhecer_faces(img)
        if not faces or faces[0]["nome"] == "desconhecido":
            return "nenhuma"
        return str(faces[0]["label"])

    def cadastrar_rosto_imagem_b64(self, nome_pessoa: str, imagem_jpeg_b64: str) -> Tuple[bool, str]:
        """Extrai o embedding SFace da maior face na imagem e persiste em `known_faces.json`."""
        img = self._decodificar_imagem_b64(imagem_jpeg_b64)
        faces = self.detectar_e_reconhecer_faces(img, incluir_embedding=True)
        if not faces:
            return False, "Nenhum rosto detectado na imagem."
        try:
            import numpy as np  # type: ignore

            feat = faces[0]["feat"]
            vec = np.asarray(feat, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                vec = vec / norm

            chave = nome_pessoa.strip()
            db_path = self._known_faces_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            dados: Dict[str, Any] = {}
            if db_path.exists():
                dados = json.loads(db_path.read_text(encoding="utf-8"))
            dados[chave] = [float(x) for x in vec.tolist()]

            tmp = db_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(db_path)

            self._known_faces[chave] = np.array([vec], dtype=np.float32)
            return True, f"Rosto de '{chave}' salvo em {db_path.name} ({len(vec)}-D)."
        except Exception as exc:
            return False, f"Falha ao salvar rosto: {exc}"

    def _obter_historico_hidratado(self, pessoa_id: str) -> List[Dict[str, str]]:
        """Garante continuidade multi-turno mesmo entre execucoes/processos separados."""
        hist = self._historicos_por_pessoa.get(pessoa_id)
        if not hist:
            hist = self.memoria.obter_historico_dialogo(pessoa_id, limite_turnos=6)
            if hist:
                self._historicos_por_pessoa[pessoa_id] = hist
        return hist or []

    def eh_modo_formal(self, perfil: Dict[str, Any]) -> bool:
        """Retorna True se o turno exigir postura estritamente formal (SERIO ou Orientador)."""
        if self.modo_persona == "SERIO":
            return True
        if perfil.get("papel") == "professor_orientador":
            return True
        tom = str(perfil.get("tom_com_a_pessoa") or perfil.get("tom") or "").lower()
        return tom in (
            "serio",
            "formal",
            "respeitoso_formal_objetivo",
        )

    def _montar_bloco_contexto(
        self,
        pessoa_id: str,
        pessoa_display: str,
        modo_identificacao: str,
        perfil: Dict[str, Any],
        transcricao: str = "",
    ) -> str:
        perfil_str, memoria_str = self.memoria.formatar_contexto_prompt(pessoa_id)
        modo_efetivo = "SERIO" if self.eh_modo_formal(perfil) else self.modo_persona
        contexto_sala = self.memoria.obter_contexto_sala_recente(limite=3)
        grafo_str = self.grafo.formatar_subgrafo_prompt(pessoa_id, transcricao)
        data_str, hora_str = self._formatar_data_hora_atual()
        return (
            "=== CONTEXTO DO TURNO (injetado dinamicamente) ===\n"
            f"DATA_HORA_ATUAL: {data_str}, {hora_str}\n"
            f"MODO_PERSONA: {modo_efetivo}\n"
            f"PESSOA_ATUAL: {pessoa_display}\n"
            f"IDENTIFICACAO: {modo_identificacao}\n"
            f"PERFIL_PESSOA: {{{perfil_str}}}\n"
            f"MEMORIA_PESSOA: {memoria_str}\n"
            f"GRAFO_MEMORIA: {grafo_str}\n"
            f"ESTADO_CORPORAL: {{{self.estado.formatar_prompt()}}}\n"
            f"TELEMETRIA: {self.estado.formatar_telemetria()}\n"
            f"ULTIMAS_FALAS_SALA: {contexto_sala}"
        )

    def _montar_system_instruction(
        self,
        pessoa_id: str,
        pessoa_display: str,
        modo_identificacao: str,
        perfil: Dict[str, Any],
        cerebro: Optional[CerebroSpec] = None,
        dados_skill: str = "",
        recados_pendentes: str = "",
        transcricao: str = "",
    ) -> str:
        contexto_turno = self._montar_bloco_contexto(
            pessoa_id=pessoa_id,
            pessoa_display=pessoa_display,
            modo_identificacao=modo_identificacao,
            perfil=perfil,
            transcricao=transcricao,
        )
        if self.usar_multi_cerebro and cerebro is not None:
            return GerenciadorCerebros.compor_system_instruction(
                cerebro=cerebro,
                bloco_contexto=contexto_turno,
                dados_skill=dados_skill,
                recados_pendentes=recados_pendentes,
            )
        extra = ""
        if dados_skill.strip():
            extra += f"\n\n=== DADOS_SKILL ===\n{dados_skill.strip()}"
        if recados_pendentes.strip():
            extra += f"\n\n=== RECADOS_PENDENTES ===\n{recados_pendentes.strip()}"
        return f"{self.prompt_base}\n\n{contexto_turno}{extra}"

    def _verificar_interceptadores_fisicos_e_modo(
        self,
        transcricao: str,
        perfil: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        """Trata troca de modo/cerebro e avisos de manuseio fisico imediato antes de acionar motores/nuvem."""
        norm = normalizar_chave(transcricao)
        formal = self.eh_modo_formal(perfil)
        nome = perfil.get("nome", "")

        if any(
            f in norm
            for f in (
                "voltar ao normal",
                "volta ao normal",
                "sai do modo serio",
                "sai do modo desgracado",
                "modo normal",
                "cerebro automatico",
                "modo automatico",
            )
        ):
            self.modo_persona = "NORMAL"
            self.cerebro_forcado = ""
            if perfil.get("papel") == "professor_orientador":
                return "Certo, professor Felipe. Sigo à disposição.", "local:troca_modo_normal"
            return "Beleza, voltei ao normal!", "local:troca_modo_normal"

        if any(f in norm for f in ("entra no modo serio", "modo serio", "fica serio")):
            self.modo_persona = "SERIO"
            self.cerebro_forcado = ""
            return "Modo sério ativado.", "local:troca_modo_serio"

        if any(
            f in norm
            for f in (
                "entra no modo desgracado",
                "modo desgracado",
                "ativa o modo desgracado",
                "fica desgracado",
            )
        ):
            if perfil.get("papel") == "professor_orientador":
                self.modo_persona = "SERIO"
                self.cerebro_forcado = ""
                return (
                    "Certo, professor Felipe. Mantenho o modo formal por aqui.",
                    "local:troca_modo_serio",
                )
            self.modo_persona = "DESGRACADO"
            self.cerebro_forcado = ""
            return "Modo desgraçado ativado. Agora segura a língua porque eu não vou segurar.", "local:troca_modo_desgracado"

        # Pin manual opcional de cérebro por voz (ex.: "ativa o cérebro técnico")
        m_cer = re.search(r"(?:ativa|usar|usa|troca pro|muda pro|entra no)\s+(?:o\s+)?cerebro\s+([a-z0-9_]+)", norm)
        if m_cer:
            alvo = m_cer.group(1).strip()
            disponiveis = {c.id: c for c in self.cerebros.listar()}
            if alvo in disponiveis:
                spec = disponiveis[alvo]
                if perfil.get("papel") == "professor_orientador" and spec.bloqueado_para_professor:
                    return (
                        "Certo, professor Felipe. Mantenho o cérebro formal ativo para o senhor.",
                        "local:trava_cerebro_professor",
                    )
                self.cerebro_forcado = spec.id
                return (
                    f"{spec.nome} ativado por aqui." if formal else f"Botei o {spec.nome} no comando aqui!",
                    f"local:troca_cerebro:{spec.id}",
                )

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
                "saudacao": f"Olá{', professor Felipe' if perfil.get('papel') == 'professor_orientador' else vocativo}. À disposição, em que posso ajudar hoje?",
                "despedida": "Até logo! Qualquer coisa é só chamar.",
                "ok_confirmacao": "Entendido, tudo certo por aqui.",
                "elogio_agradecimento": "À disposição! Sempre que precisar, é só chamar.",
            }
            return tabela_formal.get(subtipo, "Certo.")

        if nome == "desconhecido":
            tabela_desconhecido = {
                "saudacao": [
                    "Opa, fala aí! Ainda não te tenho registrado por aqui. Como você se chama?",
                    "E aí! Primeira vez que ouço essa voz por aqui, qual é o teu nome?",
                ],
                "despedida": [
                    "Valeu, até a próxima! Da próxima vez me diz teu nome pra eu já te reconhecer de primeira.",
                    "Falou! Se precisar de alguma coisa depois, só aparecer.",
                ],
            }
            if subtipo in tabela_desconhecido:
                return random.choice(tabela_desconhecido[subtipo])

        if self.modo_persona == "DESGRACADO":
            tabela_desgracado = {
                "saudacao": [
                    f"Olha quem resolveu dar sinal de vida{vocativo}. Trouxe algum problema real ou veio só procrastinar em áudio?",
                    f"Fala{vocativo}. Se for pedir milagre pra tarefa atrasada, já vai soltando logo.",
                ],
                "despedida": [
                    f"Já vai tarde{vocativo}. Tenta não arrumar outro desastre nos próximos vinte minutos.",
                    f"Falou{vocativo}, vê se volta com alguma missão que preste na próxima.",
                ],
                "ok_confirmacao": [
                    "Anotado. Vamos ver quanto tempo dura até mudar de ideia.",
                    "Beleza, registrado no sistema. Manda a próxima.",
                ],
                "elogio_agradecimento": [
                    f"Finalmente um pouco de reconhecimento pelo trabalho bem-feito{vocativo}. Manda outra.",
                    f"Eu sei que sou eficiente{vocativo}, o difícil mesmo é acompanhar o teu ritmo caótico.",
                ],
            }
            opcoes_desg = tabela_desgracado.get(subtipo, ["Tô ouvindo, manda logo o que você precisa."])
            return random.choice(opcoes_desg)

        tabela_normal = {
            "saudacao": [
                f"Fala{vocativo}, tudo certo? Manda a boa que eu tô na escuta.",
                f"E aí{vocativo}! Qual é o plano ou a treta de hoje?",
                f"Opa{vocativo}, sistema ativo e ouvidos bem abertos. O que manda?",
            ],
            "despedida": [
                f"Valeu{vocativo}, até mais! Qualquer coisa é só dar um toque.",
                "Falou! Quando precisar desenrolar alguma ideia ou problema, só chamar.",
            ],
            "ok_confirmacao": [
                "Fechado, anotado por aqui. Pode mandar a próxima.",
                "Beleza, acompanhando. Segue o baile.",
            ],
            "elogio_agradecimento": [
                f"Que isso{vocativo}, faço parecer fácil mesmo. Precisando de mais alguma coisa, só falar.",
                f"Disponha{vocativo}! Sempre bom manter o padrão lá em cima por aqui.",
            ],
        }
        opcoes = tabela_normal.get(subtipo, ["Beleza, tô acompanhando aqui!"])
        return random.choice(opcoes)

    @staticmethod
    def _formatar_data_hora_atual() -> Tuple[str, str]:
        agora = datetime.now()
        dias_sem = (
            "segunda-feira",
            "terça-feira",
            "quarta-feira",
            "quinta-feira",
            "sexta-feira",
            "sábado",
            "domingo",
        )
        meses = (
            "janeiro", "fevereiro", "março", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
        )
        data_str = f"{dias_sem[agora.weekday()]}, {agora.day} de {meses[agora.month - 1]} de {agora.year}"
        hora_str = agora.strftime("%H horas e %M minutos")
        return data_str, hora_str

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

        if subtipo == "hora_atual":
            data_str, h_str = self._formatar_data_hora_atual()
            if formal:
                return f"Hoje é {data_str}, e agora são {h_str}."
            return f"Hoje é {data_str}, e são {h_str}{vocativo}. Tá correndo contra algum prazo ou só no rolê?"

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
        transcricao: str = "",
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
            return f"Opa, calma aí! Haha, eu tô {onde} e não tenho sensor de borda ainda, se eu andar aqui eu mergulho pro chão! Me bota no chão primeiro."

        if decisao.requer_confirmacao:
            alvo = "te seguir" if subtipo == "seguir_rosto" else "ir até lá"
            if formal:
                return f"Confirma que deseja que eu execute a movimentação para {alvo}?"
            return f"Tô sem sensor de distância ainda, só na câmera. Tu confirma que eu posso {alvo}?"

        if subtipo == "seguir_rosto":
            self.estado.movimento = "seguindo_rosto"
            return "Seguindo seu rosto." if formal else "Bora, já tô te seguindo aqui!"
        if subtipo == "ir_ate_objeto":
            self.estado.movimento = "andando"
            norm = normalizar_chave(transcricao)
            if any(k in norm for k in ("frente", "tras", "re", "vira", "esquerda", "direita")):
                return "Movimentação acionada conforme solicitado." if formal else "Partiu! Acionei as rodas aqui, bora lá!"
            return "Indo em direção ao objeto." if formal else "Fechado, tô indo lá perto!"
        if subtipo == "radar":
            self.estado.movimento = "radar"
            return "Iniciando varredura com a câmera." if formal else "Deixa comigo, tô girando a cabeça pra procurar aqui."
        return "Certo."

    def _resolver_skill_turno(
        self,
        transcricao: str,
        decisao_jev: Optional[DecisaoRoteador],
    ) -> Optional[str]:
        """Determina se alguma skill modular deve ser acionada neste turno."""
        if decisao_jev is not None:
            if decisao_jev.rota_final in ("ignorar", "acao_rover"):
                return None
            if decisao_jev.rota_final == "fala_parametrizada" and decisao_jev.subtipo == "previsao_clima":
                return "clima"
            raw_sk = getattr(decisao_jev, "skill_sugerida", "nenhuma")
            raw_conf = getattr(decisao_jev, "confianca_skill", 0.0)
            sk_jev = raw_sk.strip().lower() if isinstance(raw_sk, str) else "nenhuma"
            conf_sk = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0
            if sk_jev != "nenhuma" and conf_sk >= 0.55:
                sk_obj = self.skills.obter(sk_jev)
                if sk_obj is not None:
                    # Evita que pergunta científica sobre "céu" dispare Open-Meteo sem pedir clima
                    if sk_jev == "clima" and decisao_jev.rota_final == "cognicao_gemini" and not sk_obj.pode_atender_local(transcricao):
                        return None
                    return sk_jev

        # Fallback determinístico local para comandos inequívocos (ex.: recados/timer/sonoplastia)
        return self.skills.detectar_skill_local(transcricao)

    def responder(
        self,
        transcricao: str,
        decisao_jev: Optional[DecisaoRoteador] = None,
        face_id: str = "nenhuma",
        voice_id: str = "desconhecido",
        voice_conf: float = 0.0,
        imagem_jpeg_b64: Optional[str] = None,
        reproduzir_sfx: bool = False,
    ) -> Dict[str, Any]:
        """Executa o turno completo (biometria -> travas -> skills -> multi-cérebro -> memória)."""
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
        formal = self.eh_modo_formal(perfil)
        modo_efetivo = "SERIO" if formal else self.modo_persona

        sem_rosto = normalizar_chave(face_id_efetivo) in ("", "nenhuma", "nenhum", "auto", "none", "desconhecido")
        voz_em_duvida = (
            sem_rosto
            and voice_id != "desconhecido"
            and 0.70 <= float(voice_conf) < 0.75
        )
        if voz_em_duvida:
            modo_id = "apenas_voz_confirmar"

        cerebro_usado: Optional[CerebroSpec] = None
        cerebro_motivo = ""
        res_skill: Optional[ResultadoSkill] = None
        if voz_em_duvida:
            nome_cand = str(perfil.get("nome") or voice_id)
            recados_pendentes = (
                f"CONFIRMACAO_DE_VOZ: Pela voz parece {nome_cand} ({int(round(voice_conf * 100))}%), "
                "mas há ruído ou dúvida; pergunte rapidamente se é ela mesmo antes de assumir certeza."
            )
        else:
            recados_pendentes = self.skills.coletar_recados_pendentes(pessoa_id)

        # 1) Interceptadores fisicos/modo/cerebro tem prioridade imediata
        inter = self._verificar_interceptadores_fisicos_e_modo(transcricao, perfil)
        if inter is not None:
            resposta_texto, origem = inter
        elif decisao_jev is not None and decisao_jev.rota_final == "ignorar":
            return {
                "resposta": "",
                "resposta_tts": "",
                "origem": "ignorar",
                "pessoa_id": pessoa_id,
                "pessoa_display": pessoa_display,
                "modo_identificacao": modo_id,
                "modo_persona": modo_efetivo,
                "estado_corporal": self.estado.formatar_prompt(),
                "cerebro_id": "",
                "cerebro_nome": "",
                "cerebro_motivo": "ignorado",
                "skill_executada": "",
                "skill_metadados": {},
                "sfx_disparado": None,
            }
        elif decisao_jev is not None and decisao_jev.rota_final == "acao_rover":
            resposta_texto = self._responder_acao_rover(decisao_jev, perfil, transcricao=transcricao)
            origem = f"local:acao:{decisao_jev.subtipo}"
        else:
            self.grafo.ingerir_texto(
                transcricao=transcricao,
                pessoa_id=pessoa_id,
                pessoa_nome=str(perfil.get("nome") or pessoa_id),
                modo="local",
            )
            # 2) Skills modulares (via Jev `skill_modular` ou gatilho local inequívoco)
            skill_id = self._resolver_skill_turno(transcricao, decisao_jev)
            if skill_id:
                ctx_skill = ContextoSkill(
                    transcricao=transcricao,
                    pessoa_id=pessoa_id,
                    pessoa_nome=str(perfil.get("nome") or "desconhecido"),
                    modo_persona=modo_efetivo,
                    eh_formal=formal,
                    cerebro_id=self.cerebro_forcado or (decisao_jev.cerebro_sugerido if decisao_jev else "resenha"),
                    memoria=self.memoria,
                )
                res_skill = self.skills.executar(skill_id, ctx_skill)

            if res_skill is not None and not res_skill.precisa_llm and res_skill.resposta_direta:
                resposta_texto = res_skill.resposta_direta
                origem = f"skill:{res_skill.skill_id}"
            elif res_skill is not None:
                resposta_texto, origem, cerebro_usado, cerebro_motivo = self._gerar_resposta_llm(
                    transcricao=transcricao,
                    pessoa_id=pessoa_id,
                    pessoa_display=pessoa_display,
                    modo_identificacao=modo_id,
                    perfil=perfil,
                    imagem_jpeg_b64=imagem_jpeg_b64,
                    decisao_jev=decisao_jev,
                    dados_skill=res_skill.resumo_contexto,
                    recados_pendentes=recados_pendentes,
                )
            elif decisao_jev is not None and decisao_jev.rota_final == "pool_fala_pronta" and not recados_pendentes:
                resposta_texto = self._responder_local_pool(decisao_jev.subtipo, perfil)
                origem = f"local:pool:{decisao_jev.subtipo}"
            elif (
                decisao_jev is not None
                and decisao_jev.rota_final == "fala_parametrizada"
                and decisao_jev.subtipo != "previsao_clima"
                and not recados_pendentes
            ):
                resposta_texto = self._responder_local_parametrizada(
                    decisao_jev.subtipo, perfil, pessoa_display, modo_id
                )
                origem = f"local:param:{decisao_jev.subtipo}"
            else:
                resposta_texto, origem, cerebro_usado, cerebro_motivo = self._gerar_resposta_llm(
                    transcricao=transcricao,
                    pessoa_id=pessoa_id,
                    pessoa_display=pessoa_display,
                    modo_identificacao=modo_id,
                    perfil=perfil,
                    imagem_jpeg_b64=imagem_jpeg_b64,
                    decisao_jev=decisao_jev,
                    recados_pendentes=recados_pendentes,
                )

        # 3) Avaliador anti-cringe de sonoplastia (bloqueado no SERIO / técnico / utilitário)
        cerebro_id_final = cerebro_usado.id if cerebro_usado else ("serio" if formal else ("desgracado" if modo_efetivo == "DESGRACADO" else "resenha"))
        sfx_disparado = None
        if not (res_skill and res_skill.skill_id == "sonoplastia"):
            sfx_disparado = self.skills.avaliar_sonoplastia(
                transcricao=transcricao,
                resposta=resposta_texto,
                cerebro_id=cerebro_id_final,
                modo_persona=modo_efetivo,
                eh_formal=formal,
                reproduzir_audio=reproduzir_sfx,
            )

        hist_pessoa = self._historicos_por_pessoa.setdefault(pessoa_id, [])
        hist_pessoa.append({"role": "user", "text": transcricao.strip()})
        hist_pessoa.append({"role": "model", "text": resposta_texto})
        self._historicos_por_pessoa[pessoa_id] = hist_pessoa[-12:]

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
            "cerebro_id": cerebro_usado.id if cerebro_usado else "",
            "cerebro_nome": cerebro_usado.nome if cerebro_usado else "",
            "cerebro_motivo": cerebro_motivo,
            "skill_executada": res_skill.skill_id if res_skill else "",
            "skill_metadados": res_skill.metadados if res_skill else {},
            "sfx_disparado": sfx_disparado,
        }

    def _gerar_resposta_llm(
        self,
        transcricao: str,
        pessoa_id: str,
        pessoa_display: str,
        modo_identificacao: str,
        perfil: Dict[str, Any],
        imagem_jpeg_b64: Optional[str] = None,
        decisao_jev: Optional[DecisaoRoteador] = None,
        dados_skill: str = "",
        recados_pendentes: str = "",
    ) -> Tuple[str, str, Optional[CerebroSpec], str]:
        cerebro_spec: Optional[CerebroSpec] = None
        cerebro_motivo = "prompt_unificado_legado"
        if self.usar_multi_cerebro:
            raw_cer = getattr(decisao_jev, "cerebro_sugerido", "") if decisao_jev else ""
            raw_cer_conf = getattr(decisao_jev, "confianca_cerebro", 0.0) if decisao_jev else 0.0
            cerebro_spec, cerebro_motivo = self.cerebros.resolver_cerebro(
                modo_persona=self.modo_persona,
                perfil=perfil,
                cerebro_jev=raw_cer if isinstance(raw_cer, str) else "",
                confianca_jev=float(raw_cer_conf) if isinstance(raw_cer_conf, (int, float)) else 0.0,
                cerebro_forcado=self.cerebro_forcado,
                tem_dados_skill=bool(dados_skill.strip() or recados_pendentes.strip()),
                precisa_imagem=bool(imagem_jpeg_b64),
            )

        system_inst = self._montar_system_instruction(
            pessoa_id=pessoa_id,
            pessoa_display=pessoa_display,
            modo_identificacao=modo_identificacao,
            perfil=perfil,
            cerebro=cerebro_spec,
            dados_skill=dados_skill,
            recados_pendentes=recados_pendentes,
            transcricao=transcricao,
        )

        temperatura = cerebro_spec.temperature if cerebro_spec else None
        max_tokens = cerebro_spec.max_tokens if cerebro_spec else 320
        few_shot = cerebro_spec.exemplos_few_shot if cerebro_spec else None

        api_key = self._env.get("GEMINI_API_KEY", "").strip()
        if api_key:
            texto, modelo_usado = self._chamar_gemini_rest(
                api_key=api_key,
                pessoa_id=pessoa_id,
                system_inst=system_inst,
                transcricao=transcricao,
                imagem_jpeg_b64=imagem_jpeg_b64,
                temperature=temperatura,
                max_tokens=max_tokens,
                exemplos_few_shot=few_shot,
            )
            if texto:
                sufixo = f":{cerebro_spec.id}" if cerebro_spec else ""
                return texto, f"gemini:{modelo_usado}{sufixo}", cerebro_spec, cerebro_motivo

        texto_ol, modelo_ol = self._chamar_ollama_chat_fallback(
            pessoa_id=pessoa_id,
            system_inst=system_inst,
            transcricao=transcricao,
            imagem_jpeg_b64=imagem_jpeg_b64,
        )
        if texto_ol:
            return texto_ol, f"ollama_chat:{modelo_ol}", cerebro_spec, cerebro_motivo

        return (
            "Não consegui conectar no Gemini nem achei modelo de chat local no Ollama. Confere sua GEMINI_API_KEY no .env!",
            "fallback_sem_llm",
            cerebro_spec,
            cerebro_motivo,
        )

    def _chamar_gemini_rest(
        self,
        api_key: str,
        pessoa_id: str,
        system_inst: str,
        transcricao: str,
        imagem_jpeg_b64: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 320,
        exemplos_few_shot: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, str]:
        modelos: List[str] = []
        for cand in (
            self._env.get("GEMINI_PERSONA_MODEL", ""),
            "gemini-3.5-flash-lite",
            self._env.get("GEMINI_MODEL", ""),
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
        ):
            m = cand.strip()
            if m and m not in modelos:
                modelos.append(m)

        contents: List[Dict[str, Any]] = []
        for u_ex, a_ex in (exemplos_few_shot or []):
            contents.append({"role": "user", "parts": [{"text": u_ex}]})
            contents.append({"role": "model", "parts": [{"text": a_ex}]})
        hist_pessoa = self._obter_historico_hidratado(pessoa_id)
        for item in hist_pessoa[-12:]:
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

        if temperature is None:
            temperatura = 1.0
            if "MODO_PERSONA: SERIO" in system_inst:
                temperatura = 0.45
            elif "MODO_PERSONA: DESGRACADO" in system_inst:
                temperatura = 1.1
        else:
            temperatura = float(temperature)

        payload = {
            "systemInstruction": {"parts": [{"text": system_inst}]},
            "contents": contents,
            "generationConfig": {"temperature": temperatura, "maxOutputTokens": int(max_tokens)},
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
        imagem_jpeg_b64: Optional[str] = None,
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

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_inst}]
        hist_pessoa = self._obter_historico_hidratado(pessoa_id)
        for item in hist_pessoa[-12:]:
            msgs.append(
                {
                    "role": "user" if item["role"] == "user" else "assistant",
                    "content": item["text"],
                }
            )
        msg_atual: Dict[str, Any] = {"role": "user", "content": transcricao.strip()}
        if imagem_jpeg_b64:
            msg_atual["images"] = [imagem_jpeg_b64]
        msgs.append(msg_atual)

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
