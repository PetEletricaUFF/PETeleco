#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 1 — Biometria de Voz (Speaker Verification na CPU, 0 MB de VRAM).

Identifica quem esta falando mesmo sem a pessoa estar no campo de visao da
camera, reaproveitando o audio recortado pelo microfone/VAD:
- Usa `resemblyzer` automaticamente se estiver instalado no ambiente.
- Caso contrario, usa extrator nativo em `torch` + `torchaudio` + `numpy`
  (`torchaudio_192d`: 40 MFCCs sem c0 + deltas + 12 bandas Log-Mel + F0/pitch
  + centroide espectral), rodando na CPU sem baixar pesos externos.

Persiste os embeddings em `data/known_voices.json` (media L2 incremental por
pessoa) e suporta auto-cadastro cruzado quando o `SFace` confirma um rosto.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICES_PATH = ROOT_DIR / "data" / "known_voices.json"

try:
    from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore

    _HAS_RESEMBLYZER = True
except Exception:
    VoiceEncoder = None
    preprocess_wav = None
    _HAS_RESEMBLYZER = False


class IdentificadorVoz:
    """Extrator de embedding vocal e classificador por similaridade cosseno."""

    def __init__(
        self,
        voices_path: Optional[Path] = None,
        sample_rate: int = 16000,
        limiar_confianca: float = 0.82,
        margem_minima: float = 0.025,
        preferir_resemblyzer: bool = True,
    ) -> None:
        self.voices_path = Path(voices_path or DEFAULT_VOICES_PATH)
        self.sample_rate = sample_rate
        self.limiar_confianca = limiar_confianca
        self.margem_minima = margem_minima

        self._encoder: Any = None
        if preferir_resemblyzer and _HAS_RESEMBLYZER:
            self.backend = "resemblyzer_256d"
            self._encoder = VoiceEncoder(device="cpu")
        else:
            self.backend = "torchaudio_192d"
            self._mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=41,  # indice 0 e energia; usamos 1..40 para invariancia de ganho
                melkwargs={
                    "n_fft": 512,
                    "hop_length": 160,
                    "n_mels": 64,
                    "f_min": 60.0,
                    "f_max": 7600.0,
                },
            )
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                hop_length=160,
                n_mels=64,
                f_min=60.0,
                f_max=7600.0,
            )

        self.vozes: Dict[str, Dict[str, Any]] = {}
        self.carregar()

    def carregar(self) -> None:
        if not self.voices_path.exists():
            self.voices_path.parent.mkdir(parents=True, exist_ok=True)
            self.vozes = {}
            self.salvar()
            return
        with self.voices_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.vozes = data if isinstance(data, dict) else {}

    def salvar(self) -> None:
        self.voices_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.voices_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.vozes, f, ensure_ascii=False, indent=2)
            f.write("\n")
        tmp_path.replace(self.voices_path)

    def _preparar_audio(self, audio_np: np.ndarray, sr: int) -> torch.Tensor:
        arr = np.asarray(audio_np, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        arr = arr - float(np.mean(arr)) if arr.size else arr
        pico = float(np.max(np.abs(arr))) if arr.size else 0.0
        if pico > 1e-6:
            arr = arr / pico
        wav = torch.from_numpy(arr).unsqueeze(0)
        if sr != self.sample_rate and wav.numel() > 0:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    @staticmethod
    def _mascara_frames_ativos(wav_1d: torch.Tensor, frame_len: int = 400, hop: int = 160) -> Tuple[torch.Tensor, np.ndarray]:
        if wav_1d.numel() < frame_len:
            return torch.zeros(0, dtype=torch.bool), np.zeros((0, frame_len), dtype=np.float32)
        frames = wav_1d.unfold(0, frame_len, hop)
        rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-9)
        if rms.numel() == 0:
            return torch.zeros(0, dtype=torch.bool), np.zeros((0, frame_len), dtype=np.float32)
        p90 = float(torch.quantile(rms, 0.90).item())
        limiar = max(0.012, 0.16 * p90)
        mascara = rms >= limiar
        return mascara, frames.cpu().numpy()

    def extrair_embedding(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
        min_segundos_ativos: float = 0.55,
    ) -> Optional[np.ndarray]:
        """Extrai embedding L2-normalizado se houver fala ativa suficiente."""
        if audio_np is None or np.asarray(audio_np).size < int(sr * 0.4):
            return None

        if self.backend == "resemblyzer_256d" and self._encoder is not None:
            arr = np.asarray(audio_np, dtype=np.float32)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
            wav_proc = preprocess_wav(arr, source_sr=sr)
            if wav_proc.size < int(self.sample_rate * min_segundos_ativos):
                return None
            emb = self._encoder.embed_utterance(wav_proc).astype(np.float32)
            norm = float(np.linalg.norm(emb))
            return emb / norm if norm > 1e-9 else None

        wav = self._preparar_audio(audio_np, sr)
        wav_1d = wav.squeeze(0)
        mascara, frames_np = self._mascara_frames_ativos(wav_1d)
        min_frames = max(25, int((min_segundos_ativos * self.sample_rate) / 160))
        n_ativos = int(mascara.sum().item()) if mascara.numel() else 0
        if n_ativos < min_frames:
            return None

        with torch.no_grad():
            mfcc_full = self._mfcc_transform(wav).squeeze(0)  # (41, T)
            mfcc = mfcc_full[1:41, :]  # (40, T) sem c0
            delta1 = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta1)

            mel = self._mel_transform(wav).squeeze(0)  # (64, T)
            log_mel = torch.log(mel + 1e-6)

        t_min = min(mfcc.shape[1], log_mel.shape[1], mascara.shape[0])
        m = mascara[:t_min]
        mfcc_a = mfcc[:, :t_min][:, m]
        d1_a = delta1[:, :t_min][:, m]
        d2_a = delta2[:, :t_min][:, m]
        mel_a = log_mel[:, :t_min][:, m]  # (64, N)

        # 1) Bloco MFCC (160-D): media estatica (trato vocal) + variabilidade + dinamicas
        bloco_mfcc = torch.cat(
            [
                mfcc_a.mean(dim=1),
                mfcc_a.std(dim=1, unbiased=False),
                d1_a.std(dim=1, unbiased=False),
                d2_a.std(dim=1, unbiased=False),
            ],
            dim=0,
        ).cpu().numpy()

        # 2) Bloco Log-Mel agrupado em 12 bandas (24-D)
        mel_bandas = mel_a[:60, :].reshape(12, 5, -1).mean(dim=1)  # (12, N)
        bloco_mel = torch.cat(
            [
                mel_bandas.mean(dim=1),
                mel_bandas.std(dim=1, unbiased=False),
            ],
            dim=0,
        ).cpu().numpy()

        # 3) Bloco Prosodia F0 / Espectro (8-D)
        frames_ativos = frames_np[:t_min][m.cpu().numpy()]
        bloco_prosodia = self._extrair_prosodia_espectro(frames_ativos)

        # Normaliza cada bloco antes da ponderacao para evitar dominancia de escala
        b1 = self._norm_l2(bloco_mfcc) * 0.65
        b2 = self._norm_l2(bloco_mel) * 0.20
        b3 = self._norm_l2(bloco_prosodia) * 0.15

        emb_final = np.concatenate([b1, b2, b3]).astype(np.float32)
        return self._norm_l2(emb_final)

    def _extrair_prosodia_espectro(self, frames: np.ndarray) -> np.ndarray:
        """Calcula estatisticas de F0 (pitch), centroide espectral e ZCR (8-D)."""
        if frames.shape[0] == 0:
            return np.zeros(8, dtype=np.float32)

        sr = float(self.sample_rate)
        lag_min = int(sr / 360.0)  # ~44 amostras
        lag_max = int(sr / 70.0)   # ~228 amostras
        janela = np.hanning(frames.shape[1]).astype(np.float32)

        f0_vals: List[float] = []
        centroides: List[float] = []
        zcrs: List[float] = []
        freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)

        for frame in frames:
            f = frame - float(np.mean(frame))
            energia = float(np.dot(f, f))
            if energia <= 1e-8:
                continue

            # ZCR
            zcrs.append(float(np.mean(np.abs(np.diff(np.signbit(f))))))

            # Centroide espectral
            mag = np.abs(np.fft.rfft(f * janela))
            soma_mag = float(np.sum(mag))
            if soma_mag > 1e-8:
                centroides.append(float(np.sum(freqs * mag) / soma_mag))

            # F0 via autocorrelacao normalizada
            ac = np.correlate(f, f, mode="full")[len(f) - 1 :]
            if ac[0] > 1e-8 and lag_max < len(ac):
                trecho = ac[lag_min:lag_max] / ac[0]
                idx = int(np.argmax(trecho))
                if float(trecho[idx]) >= 0.42:
                    f0_vals.append(sr / float(lag_min + idx))

        taxa_vozeada = float(len(f0_vals)) / float(max(1, frames.shape[0]))
        if f0_vals:
            f0_arr = np.asarray(f0_vals, dtype=np.float32)
            f0_med = float(np.median(f0_arr)) / 350.0
            f0_p25 = float(np.percentile(f0_arr, 25)) / 350.0
            f0_p75 = float(np.percentile(f0_arr, 75)) / 350.0
            f0_std = float(np.std(f0_arr)) / 100.0
        else:
            f0_med = f0_p25 = f0_p75 = f0_std = 0.0

        c_arr = np.asarray(centroides, dtype=np.float32) if centroides else np.zeros(1, dtype=np.float32)
        z_arr = np.asarray(zcrs, dtype=np.float32) if zcrs else np.zeros(1, dtype=np.float32)

        return np.asarray(
            [
                f0_med,
                f0_p25,
                f0_p75,
                f0_std,
                taxa_vozeada,
                float(np.mean(c_arr)) / 4000.0,
                float(np.std(c_arr)) / 2000.0,
                float(np.mean(z_arr)) * 4.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _norm_l2(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    def cadastrar_amostra(self, pessoa_id: str, audio_np: np.ndarray, sr: int = 16000) -> int:
        """Extrai o embedding do audio e atualiza a media L2 da pessoa no banco."""
        emb = self.extrair_embedding(audio_np, sr=sr)
        if emb is None:
            return 0

        pid = pessoa_id.strip().lower()
        atual = self.vozes.get(pid)
        if atual and atual.get("backend") == self.backend and isinstance(atual.get("embedding"), list):
            n_prev = int(atual.get("amostras", 1))
            vec_prev = np.asarray(atual["embedding"], dtype=np.float32)
            if vec_prev.shape == emb.shape:
                combinado = self._norm_l2((vec_prev * float(n_prev) + emb) / float(n_prev + 1))
                n_total = n_prev + 1
            else:
                combinado = emb
                n_total = 1
        else:
            combinado = emb
            n_total = 1

        self.vozes[pid] = {
            "backend": self.backend,
            "amostras": n_total,
            "atualizado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "embedding": [round(float(x), 6) for x in combinado.tolist()],
        }
        self.salvar()
        return n_total

    def identificar(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Compara a fala com as vozes salvas e retorna `(pessoa_id, confianca, scores)`.

        Retorna `'desconhecido'` se o audio for curto demais, se nenhuma voz atingir
        `limiar_confianca` ou se houver empate ambiguo abaixo de `margem_minima`.
        """
        emb = self.extrair_embedding(audio_np, sr=sr)
        if emb is None:
            return "desconhecido", 0.0, {}

        scores: Dict[str, float] = {}
        for pid, dados in self.vozes.items():
            if not isinstance(dados, dict) or dados.get("backend") != self.backend:
                continue
            ref_list = dados.get("embedding")
            if not isinstance(ref_list, list):
                continue
            ref = self._norm_l2(np.asarray(ref_list, dtype=np.float32))
            if ref.shape != emb.shape:
                continue
            scores[pid] = float(np.dot(emb, ref))

        if not scores:
            return "desconhecido", 0.0, {}

        ordenado = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        melhor_id, melhor_score = ordenado[0]
        segundo_score = ordenado[1][1] if len(ordenado) > 1 else -1.0

        if melhor_score < self.limiar_confianca:
            return "desconhecido", melhor_score, scores
        if len(ordenado) > 1 and (melhor_score - segundo_score) < self.margem_minima:
            return "desconhecido", melhor_score, scores

        return melhor_id, melhor_score, scores
