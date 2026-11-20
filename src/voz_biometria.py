#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 1 — Biometria de Voz (Speaker Verification na CPU, 0 MB de VRAM).

Identifica quem está falando mesmo sem a pessoa estar no campo de visão da câmera:
1. `ge2e_256d` (padrão neural em PyTorch puro, 0 MB VRAM na CPU): 3-layer LSTM `40 -> 256`
   + linear `256` L2-normalizado carregando os pesos pré-treinados do Resemblyzer
   (`models/voice_encoder_ge2e.pt`), sem depender de `webrtcvad` no Windows.
2. `torchaudio_192d` (fallback determinístico 0-download): 40 MFCCs sem c0 + deltas
   + 12 bandas Log-Mel + F0/pitch + centroide espectral. Mantido sempre disponível para
   comparação e compatibilidade com perfis já salvos em `data/known_voices.json`.

Suporta identificação em **2 níveis de confiança** com estimativa de SNR (`estimar_snr_db`):
- `confirmado`: confiança alta + margem segura + SNR limpo.
- `confirmar`: faixa intermediária (`0.70..0.82`) ou áudio ruidoso/ambíguo -> sugere candidato.
- `desconhecido`: abaixo de `0.70`.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torchaudio


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICES_PATH = ROOT_DIR / "data" / "known_voices.json"
DEFAULT_GE2E_WEIGHTS_PATH = ROOT_DIR / "models" / "voice_encoder_ge2e.pt"
GE2E_WEIGHTS_URL = "https://github.com/resemble-ai/Resemblyzer/raw/master/resemblyzer/pretrained.pt"


class _GE2EEncoder(nn.Module):
    """Encoder neural GE2E (LSTM 3x256 -> Linear 256) compatível com `pretrained.pt`."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(40, 256, 3, batch_first=True)
        self.linear = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-9)


class IdentificadorVoz:
    """Extrator de embedding vocal com encoder neural GE2E + fallback 192-D e 2 níveis."""

    def __init__(
        self,
        voices_path: Optional[Path] = None,
        sample_rate: int = 16000,
        limiar_confianca: float = 0.82,
        limiar_duvida: float = 0.70,
        margem_minima: float = 0.025,
        preferir_ge2e: bool = True,
        auto_download_ge2e: bool = True,
        weights_path: Optional[Path] = None,
        preferir_resemblyzer: Optional[bool] = None,
    ) -> None:
        self.voices_path = Path(voices_path or DEFAULT_VOICES_PATH)
        self.weights_path = Path(weights_path or DEFAULT_GE2E_WEIGHTS_PATH)
        self.sample_rate = sample_rate
        self.limiar_confianca = float(limiar_confianca)
        self.limiar_duvida = float(limiar_duvida)
        self.margem_minima = float(margem_minima)
        self.auto_download_ge2e = bool(auto_download_ge2e)

        usar_neural = bool(preferir_ge2e if preferir_resemblyzer is None else preferir_resemblyzer)

        # Transformações base sempre prontas (usadas no fallback e compatibilidade retroativa)
        self._mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=41,
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
        self._ge2e_mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=40,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        )

        self._ge2e_encoder: Optional[_GE2EEncoder] = None
        if usar_neural and self._carregar_ge2e():
            self.backend = "ge2e_256d"
        else:
            self.backend = "torchaudio_192d"

        self.vozes: Dict[str, Dict[str, Any]] = {}
        self.carregar()

    def _carregar_ge2e(self) -> bool:
        if not self.weights_path.exists() and self.auto_download_ge2e:
            try:
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.weights_path.with_suffix(".pt.tmp")
                resp = requests.get(GE2E_WEIGHTS_URL, timeout=20.0, stream=True)
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                if tmp.stat().st_size > 1_000_000:
                    tmp.replace(self.weights_path)
                else:
                    tmp.unlink(missing_ok=True)
            except Exception:
                return False

        if not self.weights_path.exists():
            return False

        try:
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            enc = _GE2EEncoder()
            enc.load_state_dict(state, strict=True)
            enc.eval()
            self._ge2e_encoder = enc
            return True
        except Exception:
            self._ge2e_encoder = None
            return False

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

    def estimar_snr_db(self, audio_np: np.ndarray, sr: int = 16000) -> float:
        """Estima SNR aproximado em dB pela razão entre percentil 90 e 15 de RMS dos frames."""
        if audio_np is None or np.asarray(audio_np).size < 400:
            return 0.0
        wav_1d = self._preparar_audio(audio_np, sr).squeeze(0)
        if wav_1d.numel() < 400:
            return 0.0
        frames = wav_1d.unfold(0, 400, 160)
        rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-10).cpu().numpy()
        if rms.size == 0:
            return 0.0
        p90 = float(np.percentile(rms, 90))
        p15 = float(np.percentile(rms, 15))
        snr = 20.0 * np.log10((p90 + 1e-6) / (p15 + 1e-6))
        return round(float(max(0.0, snr)), 2)

    def _extrair_ge2e_256d(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
        min_segundos_ativos: float = 0.55,
    ) -> Optional[np.ndarray]:
        if self._ge2e_encoder is None:
            return None
        wav = self._preparar_audio(audio_np, sr)
        wav_1d = wav.squeeze(0)
        mascara, _ = self._mascara_frames_ativos(wav_1d)
        min_frames = max(25, int((min_segundos_ativos * self.sample_rate) / 160))
        n_ativos = int(mascara.sum().item()) if mascara.numel() else 0
        if n_ativos < min_frames:
            return None

        with torch.no_grad():
            mel = self._ge2e_mel_transform(wav).squeeze(0).transpose(0, 1)  # (T, 40)
            t_min = min(mel.shape[0], mascara.shape[0])
            mel_ativo = mel[:t_min][mascara[:t_min]]
            if mel_ativo.shape[0] < 30:
                return None

            janela = 160
            passo = 80
            if mel_ativo.shape[0] <= janela:
                janelas = mel_ativo.unsqueeze(0)
            else:
                fatias = []
                for ini in range(0, mel_ativo.shape[0] - janela + 1, passo):
                    fatias.append(mel_ativo[ini : ini + janela])
                if not fatias or (mel_ativo.shape[0] - janela) % passo != 0:
                    fatias.append(mel_ativo[-janela:])
                janelas = torch.stack(fatias, dim=0)

            embeds = self._ge2e_encoder(janelas)  # (N_janelas, 256)
            emb_medio = embeds.mean(dim=0).cpu().numpy().astype(np.float32)
        return self._norm_l2(emb_medio)

    def _extrair_torchaudio_192d(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
        min_segundos_ativos: float = 0.55,
    ) -> Optional[np.ndarray]:
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

        bloco_mfcc = torch.cat(
            [
                mfcc_a.mean(dim=1),
                mfcc_a.std(dim=1, unbiased=False),
                d1_a.std(dim=1, unbiased=False),
                d2_a.std(dim=1, unbiased=False),
            ],
            dim=0,
        ).cpu().numpy()

        mel_bandas = mel_a[:60, :].reshape(12, 5, -1).mean(dim=1)  # (12, N)
        bloco_mel = torch.cat(
            [
                mel_bandas.mean(dim=1),
                mel_bandas.std(dim=1, unbiased=False),
            ],
            dim=0,
        ).cpu().numpy()

        frames_ativos = frames_np[:t_min][m.cpu().numpy()]
        bloco_prosodia = self._extrair_prosodia_espectro(frames_ativos)

        b1 = self._norm_l2(bloco_mfcc) * 0.65
        b2 = self._norm_l2(bloco_mel) * 0.20
        b3 = self._norm_l2(bloco_prosodia) * 0.15

        emb_final = np.concatenate([b1, b2, b3]).astype(np.float32)
        return self._norm_l2(emb_final)

    def extrair_embedding(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
        min_segundos_ativos: float = 0.55,
    ) -> Optional[np.ndarray]:
        """Extrai embedding L2-normalizado usando o backend ativo."""
        if audio_np is None or np.asarray(audio_np).size < int(sr * 0.4):
            return None
        if self.backend == "ge2e_256d":
            emb = self._extrair_ge2e_256d(audio_np, sr=sr, min_segundos_ativos=min_segundos_ativos)
            if emb is not None:
                return emb
        return self._extrair_torchaudio_192d(audio_np, sr=sr, min_segundos_ativos=min_segundos_ativos)

    def _extrair_prosodia_espectro(self, frames: np.ndarray) -> np.ndarray:
        """Calcula estatísticas de F0 (pitch), centroide espectral e ZCR (8-D)."""
        if frames.shape[0] == 0:
            return np.zeros(8, dtype=np.float32)

        sr = float(self.sample_rate)
        lag_min = int(sr / 360.0)
        lag_max = int(sr / 70.0)
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

            zcrs.append(float(np.mean(np.abs(np.diff(np.signbit(f))))))

            mag = np.abs(np.fft.rfft(f * janela))
            soma_mag = float(np.sum(mag))
            if soma_mag > 1e-8:
                centroides.append(float(np.sum(freqs * mag) / soma_mag))

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
        """Extrai o embedding do áudio e atualiza a média L2 da pessoa no banco."""
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

    def identificar_detalhado(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
    ) -> Dict[str, Any]:
        """
        Identifica locutor em 2 níveis de confiança combinando score cosseno, margem e SNR.

        Status retornados:
        - `"confirmado"`: `score >= limiar_confianca`, `margem >= margem_minima` e `snr_db >= 7.0`.
        - `"confirmar"`: faixa de dúvida (`0.70..0.82`) ou confiança afetada por ruído/empate.
        - `"desconhecido"`: abaixo do limiar de dúvida.
        """
        snr_db = self.estimar_snr_db(audio_np, sr=sr)
        emb_prim = self.extrair_embedding(audio_np, sr=sr)
        if emb_prim is None:
            return {
                "status": "desconhecido",
                "pessoa_id": "desconhecido",
                "candidato_id": "",
                "confianca": 0.0,
                "margem": 0.0,
                "snr_db": snr_db,
                "requer_confirmacao": False,
                "backend": self.backend,
                "scores": {},
            }

        emb_192: Optional[np.ndarray] = emb_prim if self.backend == "torchaudio_192d" else None
        scores: Dict[str, float] = {}

        for pid, dados in self.vozes.items():
            if not isinstance(dados, dict):
                continue
            ref_list = dados.get("embedding")
            if not isinstance(ref_list, list):
                continue
            bk_ref = str(dados.get("backend") or self.backend)
            if bk_ref == self.backend:
                emb_alvo = emb_prim
            elif bk_ref == "torchaudio_192d":
                if emb_192 is None:
                    emb_192 = self._extrair_torchaudio_192d(audio_np, sr=sr)
                if emb_192 is None:
                    continue
                emb_alvo = emb_192
            else:
                continue

            ref = self._norm_l2(np.asarray(ref_list, dtype=np.float32))
            if ref.shape != emb_alvo.shape:
                continue
            scores[pid] = float(np.dot(emb_alvo, ref))

        if not scores:
            return {
                "status": "desconhecido",
                "pessoa_id": "desconhecido",
                "candidato_id": "",
                "confianca": 0.0,
                "margem": 0.0,
                "snr_db": snr_db,
                "requer_confirmacao": False,
                "backend": self.backend,
                "scores": {},
            }

        ordenado = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        melhor_id, melhor_score = ordenado[0]
        segundo_score = ordenado[1][1] if len(ordenado) > 1 else -1.0
        margem = (melhor_score - segundo_score) if len(ordenado) > 1 else 1.0

        if melhor_score >= self.limiar_confianca and margem >= self.margem_minima and snr_db >= 7.0:
            status = "confirmado"
            pessoa_id = melhor_id
            candidato_id = melhor_id
            requer_confirmacao = False
        elif (self.limiar_duvida <= melhor_score < self.limiar_confianca) or (
            melhor_score >= 0.72 and (snr_db < 7.0 or margem < self.margem_minima)
        ):
            status = "confirmar"
            pessoa_id = "desconhecido"
            candidato_id = melhor_id
            requer_confirmacao = True
        else:
            status = "desconhecido"
            pessoa_id = "desconhecido"
            candidato_id = ""
            requer_confirmacao = False

        return {
            "status": status,
            "pessoa_id": pessoa_id,
            "candidato_id": candidato_id,
            "confianca": round(float(melhor_score), 4),
            "margem": round(float(margem), 4),
            "snr_db": snr_db,
            "requer_confirmacao": requer_confirmacao,
            "backend": self.backend,
            "scores": scores,
        }

    def identificar(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Compara a fala com as vozes salvas e retorna `(pessoa_id, confianca, scores)`.

        Mantém o contrato clássico usando `identificar_detalhado`.
        """
        det = self.identificar_detalhado(audio_np, sr=sr)
        return str(det["pessoa_id"]), float(det["confianca"]), dict(det["scores"])
