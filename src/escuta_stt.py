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
import os
import threading
import time
from typing import Any, Callable, List, Optional, Tuple
import wave

import numpy as np
import requests
import sounddevice as sd
import torch
import torchaudio

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
        stt_modo: Optional[str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.whisper_model_size = whisper_model_size
        self.aliases_stt = aliases_stt or []
        self.device_index = device_index
        self._env = carregar_env()
        modo_raw = (stt_modo or self._env.get("PETELECO_STT_MODE") or "auto").strip().lower()
        self.stt_modo = modo_raw if modo_raw in ("auto", "gemini", "whisper") else "auto"
        self._whisper_model: Any = None
        self._whisper_device_label: str = ""
        self._whisper_falhou: bool = False
        self._session = requests.Session()

    def configurar(
        self,
        *,
        stt_modo: Optional[str] = None,
        whisper_model_size: Optional[str] = None,
    ) -> None:
        """Permite alternar backend ou tamanho do Whisper em runtime (UI/CLI)."""
        if stt_modo and stt_modo in ("auto", "gemini", "whisper"):
            self.stt_modo = stt_modo
        if whisper_model_size and whisper_model_size != self.whisper_model_size:
            self.whisper_model_size = whisper_model_size
            self._whisper_model = None
            self._whisper_falhou = False

    def _gemini_disponivel(self) -> bool:
        return bool(self._env.get("GEMINI_API_KEY", "").strip())

    def _usar_gemini_primeiro(self) -> bool:
        if self.stt_modo == "gemini":
            return self._gemini_disponivel()
        if self.stt_modo == "whisper":
            return False
        # Modo auto: na CUDA, faster-whisper local (~250 ms) vence; na CPU, Gemini audio (~1.5 s) e muito mais rapido e preciso
        return (not torch.cuda.is_available()) and self._gemini_disponivel()

    def descrever_backend(self) -> str:
        hw = "CUDA" if torch.cuda.is_available() else "CPU"
        if self._usar_gemini_primeiro():
            gem_mod = (self._env.get("GEMINI_STT_MODEL") or "gemini-3.5-flash-lite").strip()
            return f"gemini-stt:{gem_mod} (fallback whisper {self.whisper_model_size}/{hw})"
        if _HAS_FASTER_WHISPER:
            return f"faster-whisper ({self.whisper_model_size}/{hw})"
        if self._gemini_disponivel():
            return "gemini-stt:gemini-3.5-flash-lite"
        return "indisponivel"

    @staticmethod
    def listar_microfones() -> List[Tuple[int, str]]:
        """Retorna lista `(indice, nome)` dos dispositivos de entrada disponiveis."""
        dispositivos = sd.query_devices()
        return [
            (idx, str(d["name"]))
            for idx, d in enumerate(dispositivos)
            if int(d.get("max_input_channels", 0)) > 0
        ]

    def _resolver_sample_rate_entrada(self) -> int:
        """Usa 16 kHz se o dispositivo aceitar; caso contrario, usa o default_samplerate do hardware."""
        try:
            sd.check_input_settings(device=self.device_index, samplerate=self.sample_rate, channels=1)
            return self.sample_rate
        except Exception:
            try:
                info = sd.query_devices(self.device_index, kind="input")
                return max(8000, int(round(float(info.get("default_samplerate", 48000)))))
            except Exception:
                return self.sample_rate

    def _reamostrar_para_alvo(self, audio_np: np.ndarray, sr_origem: int) -> np.ndarray:
        if sr_origem == self.sample_rate or audio_np.size == 0:
            return audio_np.astype(np.float32)
        wav = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
        wav_rs = torchaudio.functional.resample(wav, sr_origem, self.sample_rate)
        return wav_rs.squeeze(0).numpy().astype(np.float32)

    def gravar_duracao_fixa(self, segundos: float = 4.0) -> np.ndarray:
        """Grava janela fixa do microfone padrao (util para cadastro de voz `/voz`)."""
        sr_cap = self._resolver_sample_rate_entrada()
        n_amostras = max(1, int(sr_cap * segundos))
        gravacao = sd.rec(
            n_amostras,
            samplerate=sr_cap,
            channels=1,
            dtype="float32",
            device=self.device_index,
        )
        sd.wait()
        return self._reamostrar_para_alvo(gravacao.reshape(-1), sr_cap)

    def gravar_microfone(
        self,
        duracao_max_s: float = 12.0,
        silencio_fim_s: float = 1.0,
        timeout_espera_s: float = 7.0,
        limiar_rms: float = 0.012,
        stop_event: Optional[threading.Event] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        """
        Escuta o microfone, espera o inicio da fala e para no silencio pos-fala.

        Mantem um pre-roll de 450 ms para nao cortar a primeira silaba (ex.: 'Que dia')
        e ignora estalos curtissimos (< 2 blocos consecutivos acima do limiar).
        """
        sr_cap = self._resolver_sample_rate_entrada()
        bloco_ms = 50
        bloco_amostras = int(sr_cap * (bloco_ms / 1000.0))
        pre_roll = deque(maxlen=max(1, int(450 / bloco_ms)))
        blocos_gravados: List[np.ndarray] = []

        falando = False
        blocos_voz = 0
        t_inicio = time.monotonic()
        t_ultima_fala = t_inicio

        with sd.InputStream(
            samplerate=sr_cap,
            channels=1,
            dtype="float32",
            blocksize=bloco_amostras,
            device=self.device_index,
        ) as stream:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                bloco, _ = stream.read(bloco_amostras)
                bloco_1d = bloco.reshape(-1).copy()
                rms = float(np.sqrt(np.mean(bloco_1d * bloco_1d) + 1e-9))
                agora = time.monotonic()

                if not falando:
                    pre_roll.append(bloco_1d)
                    if rms >= limiar_rms:
                        falando = True
                        blocos_voz = 1
                        t_ultima_fala = agora
                        blocos_gravados.extend(pre_roll)
                        if on_speech_start is not None:
                            on_speech_start()
                    elif (agora - t_inicio) >= timeout_espera_s:
                        return np.zeros(0, dtype=np.float32)
                else:
                    blocos_gravados.append(bloco_1d)
                    if rms >= limiar_rms * 0.75:
                        blocos_voz += 1
                        t_ultima_fala = agora
                    if (agora - t_ultima_fala) >= silencio_fim_s:
                        break
                    if (agora - t_inicio) >= duracao_max_s:
                        break

        if not blocos_gravados or blocos_voz < 2:
            return np.zeros(0, dtype=np.float32)
        return self._reamostrar_para_alvo(np.concatenate(blocos_gravados), sr_cap)

    def construir_initial_prompt(self) -> str:
        base = [
            "PETeleco",
            "PET Elétrica UFF",
            "Niterói",
            "Praia Vermelha",
            "Professor Felipe Sass",
            "Santisteban",
            "ESP32",
            "GPIO",
            "I2C",
            "UART",
            "OLED",
            "MOSFET",
            "DS18B20",
            "KiCad",
            "TB6612FNG",
            "Mini 560",
            "OneWire",
            "PWM",
            "ADC",
            "D34",
            "microfarads",
            "nanofarads",
            "quilo-ohms",
            "multímetro",
            "osciloscópio",
            "Que dia é hoje?",
            "Vamos para a frente, pai.",
        ]
        todos = base + [a for a in self.aliases_stt if a not in base]
        return "Vocabulário do laboratório: " + ", ".join(todos) + "."

    def _carregar_whisper(self) -> Optional[Any]:
        if not _HAS_FASTER_WHISPER or self._whisper_falhou:
            return None
        if self._whisper_model is not None:
            return self._whisper_model

        tentativas = [("cuda", "float16"), ("cpu", "int8")] if torch.cuda.is_available() else [("cpu", "int8")]
        threads_cpu = min(8, max(4, int(os.cpu_count() or 4)))
        ultimo_erro = ""
        for device, compute_type in tentativas:
            try:
                kwargs = {"device": device, "compute_type": compute_type}
                if device == "cpu":
                    kwargs["cpu_threads"] = threads_cpu
                self._whisper_model = WhisperModel(
                    self.whisper_model_size,
                    **kwargs,
                )
                self._whisper_device_label = f"faster-whisper({self.whisper_model_size}/{device})"
                return self._whisper_model
            except Exception as exc:
                ultimo_erro = str(exc)
                self._whisper_model = None
        if ultimo_erro:
            print(f"[AVISO STT] Falha ao carregar faster-whisper ({ultimo_erro}); usando fallback.")
        self._whisper_falhou = True
        return None

    def _transcrever_whisper(self, arr: np.ndarray, prompt_stt: str) -> Tuple[str, str]:
        modelo = self._carregar_whisper()
        if modelo is None:
            return "", ""
        try:
            # Desativa segundo vad_filter interno porque o trecho ja vem recortado por VAD + pre-roll
            segmentos, _ = modelo.transcribe(
                arr,
                language="pt",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.65,
                initial_prompt=prompt_stt,
            )
            texto = " ".join(seg.text.strip() for seg in segmentos if seg.text.strip()).strip()
            return texto, self._whisper_device_label
        except Exception:
            self._whisper_falhou = True
            return "", ""

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

        if self._usar_gemini_primeiro():
            texto_gem, backend_gem = self._transcrever_gemini_rest(arr, sr, prompt_stt)
            if texto_gem or not backend_gem.startswith("erro_"):
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return texto_gem, backend_gem, lat_ms

        texto_wh, backend_wh = self._transcrever_whisper(arr, prompt_stt)
        if backend_wh:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return texto_wh, backend_wh, lat_ms

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
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ):
            m = cand.strip()
            if m and m not in modelos:
                modelos.append(m)

        instrucao = (
            "Transcreva fielmente em portugues brasileiro exatamente o que foi falado no audio, "
            "sem explicacoes, sem aspas, mantendo pontuacao natural e nomes/apelidos quando ouvidos. "
            "Se houver apenas ruido, batida ou silencio sem fala inteligivel, responda exatamente VAZIO. "
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
                if texto.strip(" .!?").upper() == "VAZIO":
                    texto = ""
                return texto, f"gemini_stt({modelo})"
            except Exception:
                continue

        return "", "erro_gemini_stt"
