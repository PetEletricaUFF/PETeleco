#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 7 — Sonoplastia de Reação com Trava Anti-Cringe.

Pode ser hilária se usada com timing cirúrgico — e muito cringe se disparar
toda hora. Por isso o módulo impõe 4 travas rígidas por padrão:
1. Bloqueio absoluto em modo `SERIO`, para `professor_orientador` ou cérebros
   técnicos/utilitários (`tecnico`, `serio`, `utilitario`).
2. Cooldown temporal rígido (padrão 90 segundos) + espaçamento mínimo de turnos
   (padrão 3 turnos entre efeitos automáticos).
3. Probabilidade baixa no automático (`prob_auto=0.30`) e apenas quando há sinal
   claro de punchline/zoeira/fail.
4. Suporte a arquivos reais `.wav` em `data/sfx/` (`ba_dum_tss.wav`, `grilo.wav`,
   `trombone.wav`, `aplauso.wav`) com reprodução assíncrona não-bloqueante no
   Windows (`winsound`), além de modo comando explícito ("manda um ba dum tss").
"""

from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional
import wave

import numpy as np

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


DEFAULT_SFX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sfx"
SFX_SAMPLE_RATE = 16000


def _salvar_wav_pcm16(caminho: Path, sinal: np.ndarray, sr: int = SFX_SAMPLE_RATE) -> None:
    """Normaliza com folga segura (-1..1) e grava WAV PCM 16-bit mono."""
    pico = float(np.max(np.abs(sinal))) if sinal.size else 0.0
    if pico > 1e-6:
        sinal = (sinal / pico) * 0.85
    pcm = np.clip(sinal * 32767.0, -32768.0, 32767.0).astype(np.int16)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(caminho), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def gerar_sfx_padrao(sfx_dir: Optional[Path] = None, sobrescrever: bool = False) -> List[Path]:
    """
    Gera de forma determinística os 4 arquivos `.wav` padrão em `data/sfx/`
    (`ba_dum_tss.wav`, `grilo.wav`, `trombone.wav`, `aplauso.wav`) caso não existam.
    """
    alvo_dir = Path(sfx_dir) if sfx_dir else DEFAULT_SFX_DIR
    alvo_dir.mkdir(parents=True, exist_ok=True)
    sr = SFX_SAMPLE_RATE
    rng = np.random.default_rng(2026)
    criados: List[Path] = []

    def _precisa(nome_arq: str) -> Path:
        return alvo_dir / nome_arq

    # 1) ba_dum_tss.wav (~0.9 s: bumbo 85 Hz -> caixa -> prato)
    p_bdt = _precisa("ba_dum_tss.wav")
    if sobrescrever or not p_bdt.exists():
        n = int(0.90 * sr)
        y = np.zeros(n, dtype=np.float32)
        # Bumbo em t=0.00..0.16s (sweep 120 -> 85 Hz)
        n1 = int(0.16 * sr)
        t1 = np.arange(n1, dtype=np.float32) / sr
        freq1 = 85.0 + 45.0 * np.exp(-t1 * 28.0)
        fase1 = 2.0 * np.pi * np.cumsum(freq1) / sr
        y[:n1] += 0.9 * np.sin(fase1) * np.exp(-t1 * 18.0)
        # Caixa em t=0.16..0.34s (tonal + ruido curto)
        i2, n2 = int(0.16 * sr), int(0.18 * sr)
        t2 = np.arange(n2, dtype=np.float32) / sr
        y[i2 : i2 + n2] += (
            0.55 * np.sin(2.0 * np.pi * 185.0 * t2) * np.exp(-t2 * 22.0)
            + 0.45 * rng.normal(0.0, 1.0, n2) * np.exp(-t2 * 26.0)
        )
        # Prato (tss) em t=0.33..0.90s
        i3 = int(0.33 * sr)
        n3 = n - i3
        t3 = np.arange(n3, dtype=np.float32) / sr
        ruido3 = rng.normal(0.0, 1.0, n3).astype(np.float32)
        hp3 = np.diff(ruido3, prepend=0.0)
        y[i3:] += 0.65 * hp3 * np.exp(-t3 * 5.2)
        _salvar_wav_pcm16(p_bdt, y, sr)
        criados.append(p_bdt)

    # 2) grilo.wav (~1.1 s: pulsos modulados em 4.2 kHz estilo cricri)
    p_grilo = _precisa("grilo.wav")
    if sobrescrever or not p_grilo.exists():
        n = int(1.10 * sr)
        t = np.arange(n, dtype=np.float32) / sr
        portadora = 0.7 * np.sin(2.0 * np.pi * 4200.0 * t) + 0.3 * np.sin(2.0 * np.pi * 4350.0 * t)
        env = np.zeros(n, dtype=np.float32)
        for inicio_rajada in (0.05, 0.56):
            for k in range(3):
                ini = inicio_rajada + k * 0.085
                fim = ini + 0.055
                mask = (t >= ini) & (t < fim)
                tau = (t[mask] - ini) / 0.055
                env[mask] = np.sin(np.pi * tau) * (0.65 + 0.35 * np.sin(2.0 * np.pi * 55.0 * (t[mask] - ini)))
        _salvar_wav_pcm16(p_grilo, portadora * env, sr)
        criados.append(p_grilo)

    # 3) trombone.wav (~1.3 s: Bb3 -> A3 -> Ab3 -> G3 com harmonicos e vibrato)
    p_tr = _precisa("trombone.wav")
    if sobrescrever or not p_tr.exists():
        notas = [
            (233.08, 0.24, False),  # Bb3
            (220.00, 0.24, False),  # A3
            (207.65, 0.24, False),  # Ab3
            (196.00, 0.58, True),   # G3 com vibrato/queda triste
        ]
        partes = []
        for freq_base, dur, vibrato in notas:
            ni = int(dur * sr)
            ti = np.arange(ni, dtype=np.float32) / sr
            if vibrato:
                f_inst = freq_base * (1.0 - 0.09 * (ti / dur)) + 3.5 * np.sin(2.0 * np.pi * 6.0 * ti)
            else:
                f_inst = np.full(ni, freq_base, dtype=np.float32)
            fase = 2.0 * np.pi * np.cumsum(f_inst) / sr
            onda = (
                0.55 * np.sin(fase)
                + 0.30 * np.sin(2.0 * fase)
                + 0.15 * np.sin(3.0 * fase)
                + 0.07 * np.sin(4.0 * fase)
            )
            env = np.minimum(1.0, ti / 0.02) * np.exp(-ti * (1.6 if vibrato else 2.2))
            partes.append((onda * env).astype(np.float32))
        y = np.concatenate(partes)
        _salvar_wav_pcm16(p_tr, y, sr)
        criados.append(p_tr)

    # 4) aplauso.wav (~1.1 s: 4 palmas lentas ironicas com ruido filtrado)
    p_ap = _precisa("aplauso.wav")
    if sobrescrever or not p_ap.exists():
        n = int(1.10 * sr)
        y = np.zeros(n, dtype=np.float32)
        for t_palma in (0.04, 0.31, 0.58, 0.85):
            idx = int(t_palma * sr)
            npalma = min(int(0.14 * sr), n - idx)
            if npalma <= 0:
                continue
            tp = np.arange(npalma, dtype=np.float32) / sr
            ruido = rng.normal(0.0, 1.0, npalma).astype(np.float32)
            # Bandpass simples via diferenca + media movel curta
            bp = np.diff(ruido, prepend=0.0)
            bp = np.convolve(bp, np.ones(5, dtype=np.float32) / 5.0, mode="same")
            corpo = bp * np.exp(-tp * 34.0) + 0.18 * np.sin(2.0 * np.pi * 420.0 * tp) * np.exp(-tp * 45.0)
            y[idx : idx + npalma] += corpo
        _salvar_wav_pcm16(p_ap, y, sr)
        criados.append(p_ap)

    return criados

EFEITOS_SUPORTADOS = {
    "ba_dum_tss": {
        "arquivo": "ba_dum_tss.wav",
        "descricao": "Batida seca de bateria pra piada ruim ou trocadilho.",
    },
    "grilo": {
        "arquivo": "grilo.wav",
        "descricao": "Som de grilo pra vácuo, silêncio constrangedor ou piada sem graça.",
    },
    "trombone": {
        "arquivo": "trombone.wav",
        "descricao": "Trombone triste (womp womp) pra derrota, DP, erro bobo ou reprovação.",
    },
    "aplauso": {
        "arquivo": "aplauso.wav",
        "descricao": "Aplauso irônico curto pra feito duvidoso.",
    },
}


class SkillSonoplastia(SkillBase):
    id = "sonoplastia"
    nome = "Sonoplastia Anti-Cringe"
    descricao = "Dispara efeitos sonoros curtos (.wav locais) com cooldown rígido e bloqueio no modo sério/técnico."
    criterio_jev = (
        "Explicitly asking the robot to play a sound effect like ba-dum-tss, crickets, sad trombone, or ironic applause."
    )

    def __init__(
        self,
        sfx_dir: Optional[Path] = None,
        cooldown_s: float = 90.0,
        min_turnos_entre_auto: int = 3,
        prob_auto: float = 0.30,
        auto_ativo: bool = True,
        dry_run: bool = False,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.sfx_dir = Path(sfx_dir) if sfx_dir else DEFAULT_SFX_DIR
        self.cooldown_s = float(cooldown_s)
        self.min_turnos_entre_auto = int(min_turnos_entre_auto)
        self.prob_auto = min(1.0, max(0.0, float(prob_auto)))
        self.auto_ativo = bool(auto_ativo)
        self.dry_run = bool(dry_run)
        self._rng = rng or random.Random()
        self._ultimo_disparo_ts: float = 0.0
        self._turnos_desde_ultimo: int = min_turnos_entre_auto
        gerar_sfx_padrao(self.sfx_dir)

    def pode_atender_local(self, transcricao: str) -> bool:
        return self._detectar_pedido_explicito(transcricao) is not None

    @staticmethod
    def _detectar_pedido_explicito(transcricao: str) -> Optional[str]:
        norm = normalizar_texto_skill(transcricao)
        if any(k in norm for k in ("ba dum tss", "badumtss", "toca bateria", "efeito de bateria")):
            return "ba_dum_tss"
        if any(k in norm for k in ("toca o grilo", "som de grilo", "manda um grilo", "efeito de grilo")):
            return "grilo"
        if any(k in norm for k in ("trombone", "womp womp", "som de derrota", "som triste")):
            return "trombone"
        if any(k in norm for k in ("aplauso ironico", "manda palmas", "toca aplauso")):
            return "aplauso"
        return None

    @staticmethod
    def _detectar_efeito_por_punchline(transcricao: str, resposta: str) -> Optional[str]:
        norm_u = normalizar_texto_skill(transcricao)
        norm_r = normalizar_texto_skill(resposta)
        if any(k in norm_r for k in ("reprov", "dp ", " foi de arrasta", "vexame", "reprovar", "queimou")):
            return "trombone"
        if any(k in norm_u for k in ("ninguem riu", "vacuo", "cri cri", "silencio")):
            return "grilo"
        if any(k in norm_r for k in ("parabens, campeao", "vai la, campeao", "gabaritar sim")):
            return "aplauso"
        if re.search(r"\bk{3,}\b", norm_r):
            return "ba_dum_tss"
        return None

    def _tocar_wav_assincrono(self, efeito_id: str) -> Dict[str, Any]:
        info = EFEITOS_SUPORTADOS.get(efeito_id)
        if not info:
            return {"tocado": False, "motivo": "efeito_desconhecido"}
        wav_path = self.sfx_dir / info["arquivo"]
        if not wav_path.exists():
            return {"tocado": False, "motivo": "wav_ausente", "arquivo_esperado": str(wav_path)}
        if self.dry_run:
            return {"tocado": True, "motivo": "dry_run", "arquivo": str(wav_path)}
        if sys.platform.startswith("win"):
            try:
                import winsound

                winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                return {"tocado": True, "motivo": "winsound_async", "arquivo": str(wav_path)}
            except Exception as exc:
                return {"tocado": False, "motivo": f"erro_winsound:{exc}", "arquivo": str(wav_path)}
        return {"tocado": False, "motivo": "so_nao_suportado", "arquivo": str(wav_path)}

    def avaliar_e_tocar(
        self,
        *,
        transcricao: str,
        resposta: str,
        cerebro_id: str,
        modo_persona: str,
        eh_formal: bool,
        reproduzir_audio: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Avalia se um efeito sonoro faz sentido neste turno sem ficar cringe.
        Retorna dict com metadados do disparo, ou `None` quando bloqueado/ignorado.
        """
        self._turnos_desde_ultimo += 1

        # Trava 1: Nunca toca em modo formal, professor ou cérebros sérios/técnicos
        if eh_formal or modo_persona.upper() == "SERIO":
            return None
        if cerebro_id not in ("resenha", "desgracado"):
            return None

        explicito = self._detectar_pedido_explicito(transcricao)
        if explicito is not None:
            self._ultimo_disparo_ts = time.monotonic()
            self._turnos_desde_ultimo = 0
            audio_res = self._tocar_wav_assincrono(explicito) if reproduzir_audio else {"tocado": False, "motivo": "audio_desligado"}
            return {"efeito": explicito, "origem": "explicito", **audio_res}

        if not self.auto_ativo:
            return None

        # Trava 2: Cooldown temporal e espaçamento mínimo de turnos
        agora = time.monotonic()
        if self._ultimo_disparo_ts > 0.0 and (agora - self._ultimo_disparo_ts) < self.cooldown_s:
            return None
        if self._turnos_desde_ultimo < self.min_turnos_entre_auto:
            return None

        # Trava 3: Heurística de punchline + sorteio conservador (anti-spam)
        candidato = self._detectar_efeito_por_punchline(transcricao, resposta)
        if candidato is None:
            return None
        if self._rng.random() > self.prob_auto:
            return None

        self._ultimo_disparo_ts = agora
        self._turnos_desde_ultimo = 0
        audio_res = self._tocar_wav_assincrono(candidato) if reproduzir_audio else {"tocado": False, "motivo": "audio_desligado"}
        return {"efeito": candidato, "origem": "auto_punchline", **audio_res}

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        if ctx.eh_formal or ctx.modo_persona.upper() == "SERIO":
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=False,
                resposta_direta="Os efeitos sonoros ficam desativados no modo sério.",
                resumo_contexto="Sonoplastia bloqueada no modo sério.",
                precisa_llm=False,
                metadados={"bloqueado": True},
            )
        efeito = self._detectar_pedido_explicito(ctx.transcricao) or "ba_dum_tss"
        self._ultimo_disparo_ts = time.monotonic()
        self._turnos_desde_ultimo = 0
        audio_res = self._tocar_wav_assincrono(efeito)
        nomes_falados = {
            "ba_dum_tss": "Bá dum tss!",
            "grilo": "Cri, cri, cri...",
            "trombone": "Uóm, uóm, uóóóm...",
            "aplauso": "Palmas lentas pra essa obra de arte.",
        }
        return ResultadoSkill(
            skill_id=self.id,
            sucesso=True,
            resumo_contexto=f"Efeito sonoro '{efeito}' acionado.",
            resposta_direta=nomes_falados.get(efeito, "Efeito sonoro disparado!"),
            precisa_llm=False,
            metadados={"efeito": efeito, **audio_res},
        )
