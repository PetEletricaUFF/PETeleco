#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 1 — Captura de Microfone (VAD simples por energia) + STT PT-BR Multi-Backend.

Ordem de transcricao:
1. `faster-whisper` local (CUDA `float16` se disponivel, com fallback CPU `int8`),
   calibrado com `initial_prompt` contendo os nomes e apelidos do PET Eletrica UFF.
2. Fallback automatico via Gemini Multimodal Audio REST (`GEMINI_API_KEY`),
   permitindo testar fala e transcricao mesmo antes de instalar `faster-whisper`.
"""

import base64
from collections import deque
import io
import time
from typing import Any, List, Optional, Tuple
import wave

import numpy as np
import requests
import sounddevice as sd
import torch

from src.memoria_pessoas import carregar_env


try:
    from faster_whisper import WhisperModel  # type: ignore

    _HAS_FASTER_WHISPER = True
except Exception:
    WhisperModel = None
    _HAS_FASTER_WHISPER = False


class EscutaSTT:
    """Gravacao com deteccao automatica de fim de fala e transcricao PT-BR."""

    def __init__(
        self,
        sample_rate: int = 16000,
        whisper_model_size: str = "small",
        aliases_stt: Optional[List[str]] = None,
        device_index: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.whisper_model_size = whisper_model_size
        self.aliases_stt = aliases_stt or []
        self.device_index = device_index
        self._env = carregar_env()
        self._whisper_model: Any = None
        self._whisper_device_label: str = ""
        self._whisper_falhou: bool = False
        self._session = requests.Session()

    @staticmethod
    def listar_microfones() -> List[Tuple[int, str]]:
        """Retorna lista `(indice, nome)` dos dispositivos de entrada disponiveis."""
        dispositivos = sd.query_devices()
        return [
            (idx, str(d["name"]))
            for idx, d in enumerate(dispositivos)
            if int(d.get("max_input_channels", 0)) > 0
        ]

    def gravar_duracao_fixa(self, segundos: float = 4.0) -> np.ndarray:
        """Grava janela fixa do microfone padrao (util para cadastro de voz `/voz`)."""
        n_amostras = max(1, int(self.sample_rate * segundos))
        gravacao = sd.rec(
            n_amostras,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device_index,
        )
        sd.wait()
        return gravacao.reshape(-1)

    def gravar_microfone(
        self,
        duracao_max_s: float = 10.0,
        silencio_fim_s: float = 1.2,
        timeout_espera_s: float = 7.0,
        limiar_rms: float = 0.015,
    ) -> np.ndarray:
        """
        Escuta o microfone, espera o inicio da fala e para no silencio pos-fala.

        Mantem um pre-roll de 350 ms para nao cortar a primeira silaba.
        """
        bloco_ms = 50
        bloco_amostras = int(self.sample_rate * (bloco_ms / 1000.0))
        pre_roll = deque(maxlen=max(1, int(350 / bloco_ms)))
        blocos_gravados: List[np.ndarray] = []

        falando = False
        t_inicio = time.monotonic()
        t_ultima_fala = t_inicio

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=bloco_amostras,
            device=self.device_index,
        ) as stream:
            while True:
                bloco, _ = stream.read(bloco_amostras)
                bloco_1d = bloco.reshape(-1).copy()
                rms = float(np.sqrt(np.mean(bloco_1d * bloco_1d) + 1e-9))
                agora = time.monotonic()

                if not falando:
                    pre_roll.append(bloco_1d)
                    if rms >= limiar_rms:
                        falando = True
                        t_ultima_fala = agora
                        blocos_gravados.extend(pre_roll)
                    elif (agora - t_inicio) >= timeout_espera_s:
                        return np.zeros(0, dtype=np.float32)
                else:
                    blocos_gravados.append(bloco_1d)
                    if rms >= limiar_rms * 0.8:
                        t_ultima_fala = agora
                    if (agora - t_ultima_fala) >= silencio_fim_s:
                        break
                    if (agora - t_inicio) >= duracao_max_s:
                        break

        if not blocos_gravados:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocos_gravados).astype(np.float32)

    def construir_initial_prompt(self) -> str:
        base = [
            "PETeleco",
            "PET Elétrica UFF",
            "Niterói",
            "Praia Vermelha",
            "Professor Felipe Sass",
        ]
        todos = base + [a for a in self.aliases_stt if a not in base]
        return "Vocabulário do laboratório: " + ", ".join(todos) + "."

    def _carregar_whisper(self) -> Optional[Any]:
        if not _HAS_FASTER_WHISPER or self._whisper_falhou:
            return None
        if self._whisper_model is not None:
            return self._whisper_model

        tentativas = [("cuda", "float16"), ("cpu", "int8")] if torch.cuda.is_available() else [("cpu", "int8")]
        for device, compute_type in tentativas:
            try:
                self._whisper_model = WhisperModel(
                    self.whisper_model_size,
                    device=device,
                    compute_type=compute_type,
                )
                self._whisper_device_label = f"faster-whisper({self.whisper_model_size}/{device})"
                return self._whisper_model
            except Exception:
                self._whisper_model = None
        self._whisper_falhou = True
        return None

    def transcrever(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        """
        Transcreve `audio_np` para PT-BR e retorna `(texto, backend, latencia_ms)`.
        """
        arr = np.asarray(audio_np, dtype=np.float32).reshape(-1)
        if arr.size < int(sr * 0.25):
            return "", "audio_vazio", 0.0

        prompt_stt = initial_prompt or self.construir_initial_prompt()
        t0 = time.perf_counter()

        modelo = self._carregar_whisper()
        if modelo is not None:
            try:
                segmentos, _ = modelo.transcribe(
                    arr,
                    language="pt",
                    vad_filter=True,
                    initial_prompt=prompt_stt,
                )
                texto = " ".join(seg.text.strip() for seg in segmentos if seg.text.strip()).strip()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return texto, self._whisper_device_label, lat_ms
            except Exception:
                # Se CUDA falhar em runtime (ex.: cuDNN ausente), cai para Gemini REST
                self._whisper_falhou = True

        texto_gem, backend_gem = self._transcrever_gemini_rest(arr, sr, prompt_stt)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return texto_gem, backend_gem, lat_ms

    @staticmethod
    def _audio_para_wav_bytes(audio_np: np.ndarray, sr: int) -> bytes:
        clip = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (clip * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def _transcrever_gemini_rest(
        self,
        audio_np: np.ndarray,
        sr: int,
        prompt_stt: str,
    ) -> Tuple[str, str]:
        api_key = self._env.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return "", "sem_stt_e_sem_gemini_key"

        wav_b64 = base64.b64encode(self._audio_para_wav_bytes(audio_np, sr)).decode("ascii")
        modelos: List[str] = []
        for cand in (
            self._env.get("GEMINI_STT_MODEL", ""),
            "gemini-3.5-flash-lite",
            self._env.get("GEMINI_MODEL", ""),
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
        ):
            m = cand.strip()
            if m and m not in modelos:
                modelos.append(m)

        instrucao = (
            "Transcreva fielmente em português brasileiro apenas o que foi falado no áudio, "
            "sem explicações, sem aspas e mantendo nomes/apelidos quando ouvidos. "
            + prompt_stt
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": instrucao},
                        {"inlineData": {"mimeType": "audio/wav", "data": wav_b64}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
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
                    return "", f"gemini_stt({modelo})"
                partes = cands[0].get("content", {}).get("parts", [])
                texto = " ".join(
                    str(p.get("text", "")).strip()
                    for p in partes
                    if not p.get("thought") and p.get("text")
                ).strip()
                return texto, f"gemini_stt({modelo})"
            except Exception:
                continue

        return "", "erro_gemini_stt"
