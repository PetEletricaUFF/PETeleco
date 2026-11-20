#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 3 — Sintese de Voz Multi-Backend PT-BR (Edge Neural + Gemini TTS + Piper Offline).

Ordem no modo `auto` (padrao):
1. `edge-tts` (`pt-BR-AntonioNeural` neural em portugues brasileiro, ~1.2s, decodificado
   em memoria com PyAV sem depender de `ffmpeg` externo) para fala natural com ritmo e prosodia.
2. Fallback offline automatico para `piper-tts` (`pt_BR-faber-medium`, `length_scale=1.14`
   para cadencia mais calma e legivel), Windows SAPI ou `espeak-ng` caso esteja sem internet.
Modo `gemini` opcional usa `gemini-2.5-flash-preview-tts` para entonacao generativa.
"""

import asyncio
import base64
import concurrent.futures
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional, Tuple
import wave

import numpy as np
import requests

from src.memoria_pessoas import ROOT_DIR, carregar_env


DEFAULT_PIPER_DIR = ROOT_DIR / "models" / "piper"
DEFAULT_PIPER_VOICE = "pt_BR-faber-medium"
DEFAULT_EDGE_VOICE = "pt-BR-AntonioNeural"
DEFAULT_EDGE_RATE = "+4%"
DEFAULT_GEMINI_TTS_VOICE = "Puck"

try:
    import edge_tts  # type: ignore

    _HAS_EDGE_TTS = True
except Exception:
    edge_tts = None
    _HAS_EDGE_TTS = False

try:
    import av  # type: ignore

    _HAS_PYAV = True
except Exception:
    av = None
    _HAS_PYAV = False

try:
    from piper import PiperVoice  # type: ignore
    from piper.config import SynthesisConfig  # type: ignore

    _HAS_PIPER = True
except Exception:
    PiperVoice = None
    SynthesisConfig = None
    _HAS_PIPER = False

try:
    import sounddevice as sd  # type: ignore

    _HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    _HAS_SOUNDDEVICE = False


def _urls_voz_piper(voice_name: str) -> Optional[Tuple[str, str]]:
    """Monta URLs oficiais no HuggingFace (`rhasspy/piper-voices`) para vozes `<locale>-<nome>-<qualidade>`."""
    partes = voice_name.strip().split("-")
    if len(partes) < 3 or "_" not in partes[0]:
        return None
    locale = partes[0]
    lang = locale.split("_", 1)[0]
    quality = partes[-1]
    speaker = "-".join(partes[1:-1])
    base = (
        f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{lang}/{locale}/{speaker}/{quality}/{voice_name}"
    )
    return f"{base}.onnx", f"{base}.onnx.json"


def _definir_ducking_musica(ativo: bool) -> None:
    try:
        from src.skills.musica import set_music_ducking

        set_music_ducking(ativo)
    except Exception:
        pass


class SintetizadorTTS:
    """Sintetizador TTS multi-backend (Edge Neural -> Piper Offline -> SO)."""

    def __init__(
        self,
        voice_name: Optional[str] = None,
        models_dir: Optional[Path] = None,
        length_scale: float = 1.14,
        auto_download: bool = True,
        tts_modo: Optional[str] = None,
        edge_voice: Optional[str] = None,
        edge_rate: Optional[str] = None,
        gemini_voice: Optional[str] = None,
    ) -> None:
        self._env = carregar_env()
        self.voice_name = (voice_name or self._env.get("PIPER_VOICE") or DEFAULT_PIPER_VOICE).strip()
        self.models_dir = Path(models_dir or DEFAULT_PIPER_DIR)
        self.length_scale = float(length_scale)
        self.auto_download = auto_download
        modo_raw = (tts_modo or self._env.get("PETELECO_TTS_MODE") or "auto").strip().lower()
        self.tts_modo = modo_raw if modo_raw in ("auto", "edge", "gemini", "piper") else "auto"
        self.edge_voice = (edge_voice or self._env.get("PETELECO_EDGE_VOICE") or DEFAULT_EDGE_VOICE).strip()
        self.edge_rate = (edge_rate or self._env.get("PETELECO_EDGE_RATE") or DEFAULT_EDGE_RATE).strip()
        self.gemini_voice = (gemini_voice or self._env.get("PETELECO_GEMINI_TTS_VOICE") or DEFAULT_GEMINI_TTS_VOICE).strip()
        self._voice: Any = None
        self._piper_tentado: bool = False
        self._piper_falhou: bool = False
        self._session = requests.Session()
        self.ultimo_backend: str = self.descrever_backend()

    def configurar(
        self,
        *,
        tts_modo: Optional[str] = None,
        edge_voice: Optional[str] = None,
        edge_rate: Optional[str] = None,
        gemini_voice: Optional[str] = None,
        length_scale: Optional[float] = None,
    ) -> None:
        """Atualiza configuracao de TTS em tempo real (UI/CLI)."""
        if tts_modo and tts_modo in ("auto", "edge", "gemini", "piper"):
            self.tts_modo = tts_modo
        if edge_voice:
            self.edge_voice = edge_voice.strip()
        if edge_rate:
            self.edge_rate = edge_rate.strip()
        if gemini_voice:
            self.gemini_voice = gemini_voice.strip()
        if length_scale is not None:
            self.length_scale = max(0.75, min(1.5, float(length_scale)))
        self.ultimo_backend = self.descrever_backend()

    def _edge_disponivel(self) -> bool:
        return bool(_HAS_EDGE_TTS and _HAS_PYAV)

    def _gemini_disponivel(self) -> bool:
        return bool(self._env.get("GEMINI_API_KEY", "").strip())

    def descrever_backend(self) -> str:
        """Retorna descricao amigavel do backend principal configurado."""
        if self.tts_modo == "gemini" and self._gemini_disponivel():
            return f"gemini-tts:{self.gemini_voice}"
        if self.tts_modo in ("auto", "edge") and self._edge_disponivel():
            if _HAS_PIPER:
                return f"edge:{self.edge_voice} (fallback piper)"
            return f"edge:{self.edge_voice}"
        if _HAS_PIPER:
            onnx = self.models_dir / f"{self.voice_name}.onnx"
            if onnx.exists() or self.auto_download:
                return f"piper:{self.voice_name}"
        if sys.platform.startswith("win"):
            return "windows_sapi"
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            return "espeak-ng"
        if shutil.which("spd-say"):
            return "spd-say"
        return "indisponivel"

    def _baixar_arquivo(self, url: str, destino: Path) -> bool:
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(destino.suffix + ".tmp")
        try:
            with self._session.get(url, stream=True, timeout=120.0, allow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < 100:
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(destino)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            return False

    def _carregar_piper(self) -> Optional[Any]:
        if not _HAS_PIPER or self._piper_falhou:
            return None
        if self._voice is not None:
            return self._voice
        if self._piper_tentado:
            return None
        self._piper_tentado = True

        onnx_path = self.models_dir / f"{self.voice_name}.onnx"
        cfg_path = self.models_dir / f"{self.voice_name}.onnx.json"

        if not (onnx_path.exists() and cfg_path.exists()):
            if not self.auto_download:
                self._piper_falhou = True
                return None
            urls = _urls_voz_piper(self.voice_name)
            if not urls:
                self._piper_falhou = True
                return None
            print(f"[TTS] Baixando voz Piper PT-BR '{self.voice_name}' para {self.models_dir}...")
            ok_onnx = onnx_path.exists() or self._baixar_arquivo(urls[0], onnx_path)
            ok_cfg = cfg_path.exists() or self._baixar_arquivo(urls[1], cfg_path)
            if not (ok_onnx and ok_cfg):
                print("[AVISO TTS] Falha ao baixar modelo Piper; usando fallback do sistema.")
                self._piper_falhou = True
                return None
            print(f"[TTS] Voz Piper '{self.voice_name}' pronta ({onnx_path.stat().st_size // (1024 * 1024)} MB).")

        try:
            self._voice = PiperVoice.load(str(onnx_path), config_path=str(cfg_path), use_cuda=False)
            self.ultimo_backend = f"piper:{self.voice_name}"
            return self._voice
        except Exception as exc:
            print(f"[AVISO TTS] Falha ao iniciar Piper ({exc}); usando fallback do sistema.")
            self._piper_falhou = True
            self._voice = None
            return None

    @staticmethod
    def _trim_silencio(audio: np.ndarray, sr: int, pad_ini_ms: int = 35, pad_fim_ms: int = 70) -> np.ndarray:
        """Remove silencio excessivo no inicio/fim preservando ataque natural da voz."""
        if audio.size == 0:
            return audio
        idx = np.where(np.abs(audio) > 0.008)[0]
        if idx.size == 0:
            return audio
        ini = max(0, int(idx[0]) - int(sr * pad_ini_ms / 1000))
        fim = min(int(audio.size), int(idx[-1]) + int(sr * pad_fim_ms / 1000))
        return audio[ini:fim].astype(np.float32)

    def sintetizar_edge_numpy(self, texto: str) -> Tuple[Optional[np.ndarray], int]:
        """Sintetiza `texto` usando Edge Neural TTS PT-BR e decodifica MP3 em memoria com PyAV."""
        if not self._edge_disponivel():
            return None, 24000
        limpo = (texto or "").strip()
        if not limpo:
            return None, 24000

        async def _fetch_mp3() -> bytes:
            comm = edge_tts.Communicate(limpo, self.edge_voice, rate=self.edge_rate)
            buf = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        def _executar_async() -> bytes:
            return asyncio.run(asyncio.wait_for(_fetch_mp3(), timeout=12.0))

        try:
            try:
                asyncio.get_running_loop()
                # Caso ja exista event loop nesta thread, roda em worker isolado
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    mp3_bytes = pool.submit(_executar_async).result(timeout=14.0)
            except RuntimeError:
                mp3_bytes = _executar_async()

            if len(mp3_bytes) < 256:
                return None, 24000

            sr_alvo = 24000
            container = av.open(io.BytesIO(mp3_bytes), format="mp3")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sr_alvo)
            chunks = []
            for frame in container.decode(audio=0):
                for rframe in resampler.resample(frame):
                    chunks.append(rframe.to_ndarray().reshape(-1))
            if not chunks:
                return None, sr_alvo
            audio = np.concatenate(chunks).astype(np.float32)
            return self._trim_silencio(audio, sr_alvo), sr_alvo
        except Exception:
            return None, 24000

    def sintetizar_gemini_numpy(self, texto: str) -> Tuple[Optional[np.ndarray], int]:
        """Sintetiza `texto` via `gemini-2.5-flash-preview-tts` (audio generativo expressivo)."""
        api_key = self._env.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None, 24000
        limpo = (texto or "").strip()
        if not limpo:
            return None, 24000
        modelo = (self._env.get("GEMINI_TTS_MODEL") or "gemini-2.5-flash-preview-tts").strip()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Leia em portugues brasileiro natural de conversa ao vivo, "
                                "com entonacao expressiva e ritmo fluido: " + limpo
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.gemini_voice}}
                },
            },
        }
        try:
            resp = self._session.post(url, json=payload, timeout=18.0)
            resp.raise_for_status()
            data = resp.json()
            partes = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for p in partes:
                inline = p.get("inlineData") or {}
                b64 = inline.get("data")
                if not b64:
                    continue
                bruto = base64.b64decode(b64)
                mime = str(inline.get("mimeType", "")).lower()
                if "wav" in mime:
                    with wave.open(io.BytesIO(bruto), "rb") as wf:
                        sr = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return self._trim_silencio(audio, sr), sr
                # PCM L16 24kHz padrao da API Gemini Audio
                sr = 24000
                if "rate=" in mime:
                    try:
                        sr = int(mime.split("rate=", 1)[1].split(";", 1)[0])
                    except Exception:
                        sr = 24000
                audio = np.frombuffer(bruto, dtype=np.int16).astype(np.float32) / 32768.0
                return self._trim_silencio(audio, sr), sr
        except Exception:
            pass
        return None, 24000

    def sintetizar_piper_numpy(self, texto: str) -> Tuple[Optional[np.ndarray], int]:
        """Sintetiza `texto` em float32 mono `[-1, 1]` usando Piper TTS offline."""
        limpo = (texto or "").strip()
        if not limpo:
            return None, 22050
        voice = self._carregar_piper()
        if voice is None:
            return None, 22050

        syn_cfg = SynthesisConfig(length_scale=self.length_scale) if SynthesisConfig is not None else None
        blocos = []
        sr = 22050
        for chunk in voice.synthesize(limpo, syn_config=syn_cfg):
            sr = int(getattr(chunk, "sample_rate", sr))
            arr = np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)
            if arr.size:
                blocos.append(arr)
        if not blocos:
            return None, sr
        return self._trim_silencio(np.concatenate(blocos).astype(np.float32), sr), sr

    def sintetizar_numpy(self, texto: str) -> Tuple[Optional[np.ndarray], int]:
        """Sintetiza usando o backend principal configurado (Edge -> Gemini -> Piper)."""
        if self.tts_modo == "gemini":
            aud, sr = self.sintetizar_gemini_numpy(texto)
            if aud is not None:
                return aud, sr
        if self.tts_modo in ("auto", "edge"):
            aud, sr = self.sintetizar_edge_numpy(texto)
            if aud is not None:
                return aud, sr
        return self.sintetizar_piper_numpy(texto)

    @staticmethod
    def _reproduzir_numpy(audio: np.ndarray, sr: int) -> bool:
        if audio.size == 0:
            return False
        if _HAS_SOUNDDEVICE:
            try:
                sd.play(audio, samplerate=sr, blocking=True)
                return True
            except Exception:
                pass

        # Fallback por player CLI com WAV temporario
        clip = np.clip(audio, -1.0, 1.0)
        pcm16 = (clip * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        wav_bytes = buf.getvalue()

        for player in ("paplay", "pw-play", "aplay"):
            exe = shutil.which(player)
            if not exe:
                continue
            tmp_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(wav_bytes)
                    tmp_path = tf.name
                subprocess.run(
                    [exe, tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=True,
                )
                return True
            except Exception:
                pass
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)
        return False

    @staticmethod
    def _falar_windows_sapi(texto: str) -> bool:
        if not sys.platform.startswith("win"):
            return False
        env = os.environ.copy()
        env["PETELECO_TTS_TEXT"] = texto
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'pt*' } | Select-Object -First 1; "
            "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
            "$s.Rate = 1; "
            "$s.Speak($env:PETELECO_TTS_TEXT)"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=25,
                check=True,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _falar_linux_cli(texto: str) -> str:
        for cmd in (
            ["espeak-ng", "-v", "pt-br", "-s", "150", texto],
            ["espeak", "-v", "pt-br", "-s", "150", texto],
            ["spd-say", "-l", "pt-BR", "-w", texto],
        ):
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25, check=True)
                    return cmd[0]
                except Exception:
                    continue
        return ""

    def falar(self, texto: str) -> str:
        """Sintetiza e reproduz `texto` no alto-falante, retornando o backend utilizado."""
        limpo = (texto or "").strip()
        if not limpo:
            return ""

        if self.tts_modo == "gemini":
            ordem = [
                (f"gemini-tts:{self.gemini_voice}", self.sintetizar_gemini_numpy),
                (f"edge:{self.edge_voice}", self.sintetizar_edge_numpy),
                (f"piper:{self.voice_name}", self.sintetizar_piper_numpy),
            ]
        elif self.tts_modo in ("auto", "edge"):
            ordem = [
                (f"edge:{self.edge_voice}", self.sintetizar_edge_numpy),
                (f"piper:{self.voice_name}", self.sintetizar_piper_numpy),
            ]
        else:
            ordem = [(f"piper:{self.voice_name}", self.sintetizar_piper_numpy)]

        _definir_ducking_musica(True)
        try:
            for nome_backend, fn in ordem:
                audio, sr = fn(limpo)
                if audio is not None and self._reproduzir_numpy(audio, sr):
                    self.ultimo_backend = nome_backend
                    return self.ultimo_backend

            if self._falar_windows_sapi(limpo):
                self.ultimo_backend = "windows_sapi"
                return self.ultimo_backend

            cli_used = self._falar_linux_cli(limpo)
            if cli_used:
                self.ultimo_backend = cli_used
                return cli_used

            self.ultimo_backend = "indisponivel"
            return ""
        finally:
            _definir_ducking_musica(False)
