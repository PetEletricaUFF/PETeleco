#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 3 — Sintese de Voz Offline Multiplataforma (Piper TTS + Fallbacks).

Prioriza o `piper-tts` na CPU com voz PT-BR (`pt_BR-faber-medium` por padrao),
reproduzindo diretamente via `sounddevice` (ou `paplay`/`pw-play`/`aplay` no Linux).
Mantem fallback automatico para Windows SAPI (`System.Speech`) no Windows e
`espeak-ng` / `spd-say` no Linux caso o Piper nao esteja disponivel offline.
"""

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


class SintetizadorTTS:
    """Sintetizador TTS offline reutilizavel em Linux e Windows (0 MB de VRAM)."""

    def __init__(
        self,
        voice_name: Optional[str] = None,
        models_dir: Optional[Path] = None,
        length_scale: float = 0.95,
        auto_download: bool = True,
    ) -> None:
        env = carregar_env()
        self.voice_name = (voice_name or env.get("PIPER_VOICE") or DEFAULT_PIPER_VOICE).strip()
        self.models_dir = Path(models_dir or DEFAULT_PIPER_DIR)
        self.length_scale = float(length_scale)
        self.auto_download = auto_download
        self._voice: Any = None
        self._piper_tentado: bool = False
        self._piper_falhou: bool = False
        self._session = requests.Session()
        self.ultimo_backend: str = self.descrever_backend()

    def descrever_backend(self) -> str:
        """Retorna descricao amigavel do backend principal disponivel neste SO."""
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

    def sintetizar_numpy(self, texto: str) -> Tuple[Optional[np.ndarray], int]:
        """Sintetiza `texto` em float32 mono `[-1, 1]` usando Piper TTS."""
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
        return np.concatenate(blocos).astype(np.float32), sr

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
            ["espeak-ng", "-v", "pt-br", "-s", "165", texto],
            ["espeak", "-v", "pt-br", "-s", "165", texto],
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

        audio, sr = self.sintetizar_numpy(limpo)
        if audio is not None and self._reproduzir_numpy(audio, sr):
            self.ultimo_backend = f"piper:{self.voice_name}"
            return self.ultimo_backend

        if self._falar_windows_sapi(limpo):
            self.ultimo_backend = "windows_sapi"
            return self.ultimo_backend

        cli_used = self._falar_linux_cli(limpo)
        if cli_used:
            self.ultimo_backend = cli_used
            return cli_used

        return ""
