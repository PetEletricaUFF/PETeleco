#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 2 — Player de Música Headless sem Anúncios + Rádio/Fila + Audio Ducking.

Combina:
- Rádios Icecast diretas instantâneas (`SomaFM` Lo-Fi/Chill, DEF CON, Metal, DroneZone).
- Busca oficial de metadados + Rádio automática (`ytmusicapi`) + resolução de stream
  sem anúncios via SoundCloud (`yt-dlp scsearch1:...`).
- Decodificação e reprodução 100% em Python (`PyAV` + `sounddevice`) com **Audio Ducking**
  em tempo real (`18%` de ganho enquanto o PETeleco ouve no mic ou fala no TTS).
- Fallback para teclas de mídia nativas do Windows quando o player headless não está ativo.
"""

from collections import deque
import ctypes
import re
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import weakref

import numpy as np

from .base import ContextoSkill, ResultadoSkill, SkillBase, normalizar_texto_skill


try:
    import av  # type: ignore

    _HAS_PYAV = True
except Exception:
    av = None
    _HAS_PYAV = False

try:
    import sounddevice as sd  # type: ignore

    _HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    _HAS_SOUNDDEVICE = False

try:
    import yt_dlp  # type: ignore

    _HAS_YTDLP = True
except Exception:
    yt_dlp = None
    _HAS_YTDLP = False

try:
    from ytmusicapi import YTMusic  # type: ignore

    _HAS_YTMUSIC = True
except Exception:
    YTMusic = None
    _HAS_YTMUSIC = False


_RADIOS_DIRETAS: Dict[str, Dict[str, str]] = {
    "lofi": {
        "titulo": "SomaFM Groove Salad (Chill & Lo-Fi)",
        "artista": "SomaFM Rádio",
        "stream_url": "https://ice2.somafm.com/groovesalad-128-mp3",
    },
    "eletronica": {
        "titulo": "SomaFM DEF CON Radio (Synth & Eletrônica)",
        "artista": "SomaFM Rádio",
        "stream_url": "https://ice2.somafm.com/defcon-128-mp3",
    },
    "rock": {
        "titulo": "SomaFM Metal & Rock",
        "artista": "SomaFM Rádio",
        "stream_url": "https://ice2.somafm.com/metal-128-mp3",
    },
    "ambiente": {
        "titulo": "SomaFM Drone Zone (Ambiente & Foco)",
        "artista": "SomaFM Rádio",
        "stream_url": "https://ice2.somafm.com/dronezone-128-mp3",
    },
}

_VK_CODES: Dict[str, int] = {
    "play_pause": 0xB3,
    "proxima": 0xB0,
    "anterior": 0xB1,
    "parar": 0xB2,
    "volume_mais": 0xAF,
    "volume_menos": 0xAE,
    "mudo": 0xAD,
}

_PLAYERS_REGISTRADOS: "weakref.WeakSet[PlayerMusicaHeadless]" = weakref.WeakSet()
_PLAYER_SINGLETON: Optional["PlayerMusicaHeadless"] = None
_PLAYER_LOCK = threading.Lock()
_DUCKING_GLOBAL = False


def _enviar_tecla_midia_windows(acao: str, repeticoes: int = 1) -> bool:
    if not sys.platform.startswith("win"):
        return False
    vk = _VK_CODES.get(acao)
    if vk is None:
        return False
    user32 = ctypes.windll.user32
    keyeventf_keyup = 0x0002
    for _ in range(max(1, repeticoes)):
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, keyeventf_keyup, 0)
    return True


def _radio_para_estilo(consulta: str) -> Optional[Dict[str, str]]:
    norm = normalizar_texto_skill(consulta)
    if any(k in norm for k in ("eletronica", "synth", "hacker", "techno")):
        return dict(_RADIOS_DIRETAS["eletronica"])
    if any(k in norm for k in ("radio rock", "radio metal", "heavy metal")):
        return dict(_RADIOS_DIRETAS["rock"])
    if any(k in norm for k in ("ambiente", "drone", "relax", "chuva", "dormir")):
        return dict(_RADIOS_DIRETAS["ambiente"])
    if any(
        k in norm
        for k in (
            "lofi",
            "lo-fi",
            "lo fi",
            "chill",
            "estudo",
            "study beats",
            "foco",
            "radio",
            "alguma musica",
            "uma musica",
        )
    ):
        return dict(_RADIOS_DIRETAS["lofi"])
    return None


class PlayerMusicaHeadless:
    """Player headless em Python puro (`PyAV` + `sounddevice`) com fila, rádio e Audio Ducking."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        self.tocando: bool = False
        self.pausado: bool = False
        self.mudo: bool = False
        self.volume: float = 0.80
        self.ducking_ativo: bool = _DUCKING_GLOBAL
        self.faixa_atual: Optional[Dict[str, Any]] = None
        self.fila: Deque[Dict[str, Any]] = deque()
        self.historico: List[Dict[str, Any]] = []

        self._lock = threading.Lock()
        self._stop_track = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ytm: Any = None
        _PLAYERS_REGISTRADOS.add(self)

    def _obter_ytmusic(self) -> Any:
        if not _HAS_YTMUSIC or YTMusic is None:
            return None
        if self._ytm is None:
            try:
                self._ytm = YTMusic()
            except Exception:
                self._ytm = None
        return self._ytm

    def _carregar_radio_ytmusic_async(self, video_id: str, seed_titulo: str) -> None:
        if not video_id or self.dry_run:
            return
        ytm = self._obter_ytmusic()
        if ytm is None:
            return
        try:
            watch = ytm.get_watch_playlist(videoId=video_id, limit=7)
            tracks = watch.get("tracks") or []
            novas: List[Dict[str, Any]] = []
            for tr in tracks[1:6]:
                if not isinstance(tr, dict):
                    continue
                t_nome = str(tr.get("title") or "").strip()
                artistas = ", ".join(
                    str(a.get("name", "")).strip()
                    for a in (tr.get("artists") or [])
                    if isinstance(a, dict) and a.get("name")
                )
                if not t_nome or t_nome.lower() == seed_titulo.lower():
                    continue
                novas.append(
                    {
                        "titulo": t_nome,
                        "artista": artistas or "Rádio Relacionada",
                        "video_id": str(tr.get("videoId") or ""),
                        "consulta": f"{t_nome} {artistas}".strip(),
                        "stream_url": "",
                        "fonte": "ytmusic_radio",
                    }
                )
            if novas:
                with self._lock:
                    for item in novas:
                        self.fila.append(item)
        except Exception:
            pass

    @staticmethod
    def _buscar_stream_soundcloud(consulta: str) -> Optional[Dict[str, Any]]:
        if not _HAS_YTDLP or yt_dlp is None:
            return None
        opts = {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 8,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                res = ydl.extract_info(f"scsearch1:{consulta}", download=False)
                entries = res.get("entries") if isinstance(res, dict) else None
                item = entries[0] if entries else res
                if isinstance(item, dict) and item.get("url"):
                    return {
                        "titulo": str(item.get("title") or consulta).strip(),
                        "artista": str(item.get("uploader") or "SoundCloud").strip(),
                        "stream_url": str(item["url"]),
                        "fonte": "soundcloud",
                    }
        except Exception:
            return None
        return None

    def resolver_faixa_e_radio(self, consulta: str, carregar_radio: bool = True) -> Dict[str, Any]:
        """Resolve a faixa (título/artista + `stream_url`) e popula a fila de rádio."""
        q = (consulta or "").strip()
        if self.dry_run:
            faixa = {
                "titulo": q or "Lo-Fi Teste",
                "artista": "Simulado",
                "stream_url": "dry://stream",
                "fonte": "dry_run",
            }
            if carregar_radio:
                with self._lock:
                    self.fila.append(
                        {"titulo": "Rádio relacionada 1", "artista": "Simulado", "stream_url": "dry://q1", "fonte": "dry_run"}
                    )
                    self.fila.append(
                        {"titulo": "Rádio relacionada 2", "artista": "Simulado", "stream_url": "dry://q2", "fonte": "dry_run"}
                    )
            return faixa

        # Fast-path instantâneo (0s) para pedidos de estilo genérico / lo-fi / rádio
        radio = _radio_para_estilo(q)
        if radio is not None:
            return {
                "titulo": radio["titulo"],
                "artista": radio["artista"],
                "stream_url": radio["stream_url"],
                "fonte": "somafm",
            }

        titulo_meta = q
        artista_meta = ""
        video_id = ""
        ytm = self._obter_ytmusic()
        if ytm is not None:
            try:
                hits = ytm.search(q, filter="songs", limit=1)
                if hits and isinstance(hits[0], dict):
                    titulo_meta = str(hits[0].get("title") or q).strip()
                    artista_meta = ", ".join(
                        str(a.get("name", "")).strip()
                        for a in (hits[0].get("artists") or [])
                        if isinstance(a, dict) and a.get("name")
                    )
                    video_id = str(hits[0].get("videoId") or "")
            except Exception:
                pass

        if carregar_radio and video_id:
            threading.Thread(
                target=self._carregar_radio_ytmusic_async,
                args=(video_id, titulo_meta),
                name="peteleco-music-radio",
                daemon=True,
            ).start()

        busca_sc = f"{titulo_meta} {artista_meta}".strip() or q
        sc = self._buscar_stream_soundcloud(busca_sc)
        if sc is not None:
            return {
                "titulo": titulo_meta if titulo_meta != q or not sc.get("titulo") else sc["titulo"],
                "artista": artista_meta or sc["artista"],
                "stream_url": sc["stream_url"],
                "fonte": "soundcloud",
            }

        radio_fb = dict(_RADIOS_DIRETAS["lofi"])
        return {
            "titulo": radio_fb["titulo"],
            "artista": radio_fb["artista"],
            "stream_url": radio_fb["stream_url"],
            "fonte": "somafm_fallback",
        }

    def _decodificar_e_tocar_faixa(self, faixa: Dict[str, Any]) -> bool:
        """Decodifica stream via `PyAV` e envia ao `sounddevice` aplicando volume + ducking."""
        stream_url = str(faixa.get("stream_url") or "")
        if not stream_url and faixa.get("consulta"):
            res = self.resolver_faixa_e_radio(str(faixa["consulta"]), carregar_radio=False)
            faixa.update(res)
            stream_url = str(faixa.get("stream_url") or "")

        if self.dry_run or not (_HAS_PYAV and _HAS_SOUNDDEVICE and av is not None and sd is not None):
            while not self._stop_track.is_set():
                time.sleep(0.05)
            return False

        if not stream_url or stream_url.startswith("dry://"):
            return False

        sr_alvo = 44100
        try:
            with av.open(stream_url, timeout=10.0) as container:
                astream = next((s for s in container.streams if s.type == "audio"), None)
                if astream is None:
                    return False
                resampler = av.AudioResampler(format="flt", layout="stereo", rate=sr_alvo)
                with sd.OutputStream(samplerate=sr_alvo, channels=2, dtype="float32") as out:
                    for pkt in container.demux(astream):
                        if self._stop_track.is_set():
                            return False
                        for frame in pkt.decode():
                            if self._stop_track.is_set():
                                return False
                            while self.pausado and not self._stop_track.is_set():
                                time.sleep(0.05)
                            if self._stop_track.is_set():
                                return False

                            for f_out in resampler.resample(frame):
                                arr = f_out.to_ndarray()
                                if arr.ndim == 2 and arr.shape[0] == 1:
                                    pcm = arr.reshape(-1, 2)
                                elif arr.ndim == 2 and arr.shape[0] == 2:
                                    pcm = arr.T
                                elif arr.ndim == 2 and arr.shape[1] == 2:
                                    pcm = arr
                                else:
                                    pcm = arr.reshape(-1, 2)

                                ganho = 0.0 if self.mudo else (self.volume * (0.18 if self.ducking_ativo else 1.0))
                                bloco = np.ascontiguousarray(pcm * np.float32(ganho), dtype=np.float32)
                                out.write(bloco)
            return not self._stop_track.is_set()
        except Exception:
            return False

    def _iniciar_thread_playback(self) -> None:
        self._stop_track.clear()
        if self.dry_run:
            return
        t = threading.Thread(target=self._loop_playback, name="peteleco-music-player", daemon=True)
        self._thread = t
        t.start()

    def _loop_playback(self) -> None:
        while True:
            with self._lock:
                faixa = self.faixa_atual
            if faixa is None:
                return
            terminou_natural = self._decodificar_e_tocar_faixa(faixa)
            if not terminou_natural:
                # Pode ter sido interrompido para trocar de faixa ou parar
                with self._lock:
                    if not self.tocando or self.faixa_atual is not faixa:
                        return
                # Se a faixa atual falhou no decode e ha fila, tenta proxima
            with self._lock:
                if self.faixa_atual is not None:
                    self.historico.append(self.faixa_atual)
                    self.historico = self.historico[-20:]
                if self.fila and self.tocando:
                    self.faixa_atual = self.fila.popleft()
                else:
                    self.faixa_atual = None
                    self.tocando = False
                    return

    def _parar_thread_atual(self) -> None:
        self._stop_track.set()
        th = self._thread
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=1.2)
        self._thread = None

    def tocar(self, consulta: str, volume: Optional[float] = None) -> Dict[str, Any]:
        """Resolve a consulta e inicia reprodução headless (carregando fila automática)."""
        self._parar_thread_atual()
        with self._lock:
            self.fila.clear()
        faixa = self.resolver_faixa_e_radio(consulta, carregar_radio=True)
        with self._lock:
            if self.faixa_atual is not None:
                self.historico.append(self.faixa_atual)
                self.historico = self.historico[-20:]
            if volume is not None:
                self.volume = min(1.0, max(0.05, float(volume)))
            self.faixa_atual = faixa
            self.tocando = True
            self.pausado = False
            self.mudo = False
        self._iniciar_thread_playback()
        return dict(faixa)

    def adicionar_fila(self, consulta: str) -> Dict[str, Any]:
        faixa = self.resolver_faixa_e_radio(consulta, carregar_radio=False)
        with self._lock:
            self.fila.append(faixa)
            tam_fila = len(self.fila)
        if not self.tocando:
            return self.tocar(consulta)
        return {**faixa, "tamanho_fila": tam_fila}

    def alternar_pause(self) -> bool:
        with self._lock:
            if self.faixa_atual is None:
                return False
            self.pausado = not self.pausado
            self.tocando = not self.pausado
            return True

    def parar(self) -> None:
        with self._lock:
            self.tocando = False
            self.pausado = False
            self.faixa_atual = None
            self.fila.clear()
        self._parar_thread_atual()

    def proxima(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self.fila:
                return None
            if self.faixa_atual is not None:
                self.historico.append(self.faixa_atual)
                self.historico = self.historico[-20:]
            prox = self.fila.popleft()
            self.faixa_atual = prox
            self.tocando = True
            self.pausado = False
        self._parar_thread_atual()
        self._iniciar_thread_playback()
        return dict(prox)

    def anterior(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self.historico:
                return None
            if self.faixa_atual is not None:
                self.fila.appendleft(self.faixa_atual)
            ant = self.historico.pop()
            self.faixa_atual = ant
            self.tocando = True
            self.pausado = False
        self._parar_thread_atual()
        self._iniciar_thread_playback()
        return dict(ant)

    def ajustar_volume(self, delta: float) -> float:
        with self._lock:
            self.mudo = False
            self.volume = round(min(1.0, max(0.05, self.volume + float(delta))), 2)
            return self.volume

    def definir_mudo(self, mudo: Optional[bool] = None) -> bool:
        with self._lock:
            self.mudo = (not self.mudo) if mudo is None else bool(mudo)
            return self.mudo

    def definir_ducking(self, ativo: bool) -> None:
        self.ducking_ativo = bool(ativo)


def obter_player_musica(dry_run: bool = False) -> PlayerMusicaHeadless:
    global _PLAYER_SINGLETON
    with _PLAYER_LOCK:
        if _PLAYER_SINGLETON is None:
            _PLAYER_SINGLETON = PlayerMusicaHeadless(dry_run=dry_run)
        return _PLAYER_SINGLETON


def set_music_ducking(ativo: bool) -> None:
    """Ativa/desativa Audio Ducking (abaixa música para ~18% durante fala do usuário ou TTS)."""
    global _DUCKING_GLOBAL
    _DUCKING_GLOBAL = bool(ativo)
    for player in list(_PLAYERS_REGISTRADOS):
        try:
            player.definir_ducking(_DUCKING_GLOBAL)
        except Exception:
            pass


def interpretar_comando_musica(transcricao: str) -> Tuple[str, str]:
    """Retorna `(acao, consulta)` para comandos de música/mídia."""
    bruto = (transcricao or "").strip()
    norm = normalizar_texto_skill(bruto)

    if any(
        k in norm
        for k in (
            "que musica ta tocando",
            "que musica esta tocando",
            "qual musica ta tocando",
            "qual e essa musica",
            "qual eh essa musica",
            "que som e esse",
            "o que ta tocando",
            "o que esta tocando",
        )
    ):
        return "qual_musica", ""

    m_fila = re.search(
        r"(?:adiciona|coloca|bota|poe|põe)\s+(.+?)\s+na\s+fila",
        bruto,
        flags=re.IGNORECASE,
    )
    if m_fila:
        consulta_fila = m_fila.group(1).strip(" .,!?-")
        if consulta_fila:
            return "adicionar_fila", consulta_fila

    if any(k in norm for k in ("pausa", "pause", "pausar", "continua a musica", "retoma a musica", "despausa")):
        return "play_pause", ""
    if any(k in norm for k in ("proxima musica", "pula a musica", "pula essa", "passa a musica", "proxima faixa", "pula")):
        return "proxima", ""
    if any(k in norm for k in ("musica anterior", "volta a musica", "faixa anterior")):
        return "anterior", ""
    if any(k in norm for k in ("aumenta o volume", "sobe o volume", "volume mais alto")):
        return "volume_mais", ""
    if any(k in norm for k in ("abaixa o volume", "diminui o volume", "volume mais baixo")):
        return "volume_menos", ""
    if any(k in norm for k in ("muta", "silencia o som", "tira o som", "sem som")):
        return "mudo", ""
    if any(k in norm for k in ("para a musica", "desliga a musica", "para de tocar", "parar musica")):
        return "parar", ""

    # Busca de faixa/artista: "toca ...", "bota ... pra tocar", "coloca musica ..."
    m = re.search(
        r"(?:toca(?:r)?|bota(?:r)?|coloca(?:r)?|poe|põe|reproduz(?:ir)?)\s+(?:uma\s+m[uú]sica\s+(?:de|do|da)?\s*|um\s+|uma\s+|a\s+m[uú]sica\s+)?(.+)",
        bruto,
        flags=re.IGNORECASE,
    )
    if m:
        consulta = re.sub(
            r"\b(?:pra\s+tocar|ai|a[ií]|por\s+favor|peteleco|no\s+youtube|no\s+spotify)\b",
            "",
            m.group(1),
            flags=re.IGNORECASE,
        ).strip(" .,!?-")
        if consulta:
            return "tocar_busca", consulta

    return "play_pause", ""


class SkillMusica(SkillBase):
    id = "musica"
    nome = "Música & Mídia Headless"
    descricao = "Toca música/rádio sem anúncios em background (`PyAV` + `sounddevice`) com fila e ducking."
    criterio_jev = (
        "Asking to play a song/music/artist/genre/lofi, add a song to queue, ask what song is playing, "
        "pause, resume, skip to next/previous track, stop music, or turn volume up/down/mute."
    )

    def __init__(
        self,
        dry_run: bool = False,
        player: Optional[PlayerMusicaHeadless] = None,
    ) -> None:
        self.dry_run = bool(dry_run)
        if player is not None:
            self.player = player
        elif self.dry_run:
            self.player = PlayerMusicaHeadless(dry_run=True)
        else:
            self.player = obter_player_musica(dry_run=False)

    def pode_atender_local(self, transcricao: str) -> bool:
        norm = normalizar_texto_skill(transcricao)
        # Não sequestrar pedidos explícitos de efeito sonoro (sonoplastia)
        if any(s in norm for s in ("ba dum tss", "badumtss", "grilo", "trombone", "aplauso", "bateria")):
            return False
        padroes = (
            r"\b(que|qual)\s+(musica|som)\s+(ta|esta|e|eh)\s*(tocando|esse)?\b",
            r"\b(adiciona|coloca|bota|poe)\s+.+\s+na\s+fila\b",
            r"\b(pausa|pause|pausar|pula|proxima|anterior)\s+(a\s+)?(musica|faixa|som|essa)\b",
            r"\b(aumenta|abaixa|diminui|sobe)\s+o\s+volume\b",
            r"\b(para|desliga|parar)\s+(a\s+)?musica\b",
            r"\b(toca|bota|coloca)\s+.*\b(musica|lo-fi|lofi|rock|funk|pagode|sertanejo|eletronica|playlist|radio|ambiente)\b",
            r"^(toca|tocar)\s+[a-z0-9]",
        )
        return any(re.search(p, norm) for p in padroes)

    def executar(self, ctx: ContextoSkill) -> ResultadoSkill:
        acao, consulta = interpretar_comando_musica(ctx.transcricao)

        if acao == "qual_musica":
            atual = self.player.faixa_atual
            if atual is None:
                direta = (
                    "Não há nenhuma música em reprodução no momento."
                    if ctx.eh_formal
                    else "Não tá tocando nenhuma música aqui agora não."
                )
            else:
                titulo = atual.get("titulo", "faixa desconhecida")
                artista = atual.get("artista", "")
                quem = f" de {artista}" if artista else ""
                direta = (
                    f"Está tocando {titulo}{quem}."
                    if ctx.eh_formal
                    else f"Tá tocando {titulo}{quem}."
                )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=direta,
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": acao, "faixa_atual": atual},
            )

        if acao == "adicionar_fila" and consulta:
            item = self.player.adicionar_fila(consulta)
            direta = (
                f"Adicionei {item['titulo']} à fila de reprodução."
                if ctx.eh_formal
                else f"Botei {item['titulo']} na fila!"
            )
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=direta,
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": acao, "faixa": item, "tamanho_fila": len(self.player.fila)},
            )

        if acao == "tocar_busca" and consulta:
            faixa = self.player.tocar(consulta)
            titulo = faixa.get("titulo", consulta)
            artista = faixa.get("artista", "")
            quem = f" de {artista}" if artista and artista != "Simulado" else ""
            if ctx.eh_formal:
                direta = f"Iniciando a reprodução de {titulo}{quem}."
            else:
                direta = f"Soltei {titulo}{quem} aqui sem anúncio, bora!"
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=f"Reproduzindo headless: {titulo}{quem} (fonte={faixa.get('fonte')}).",
                resposta_direta=direta,
                precisa_llm=False,
                metadados={
                    "acao": acao,
                    "consulta": consulta,
                    "faixa_atual": faixa,
                    "tamanho_fila": len(self.player.fila),
                    "dry_run": self.dry_run,
                },
            )

        # Se o player headless estiver ativo (ou dry_run), controla ele diretamente
        headless_ativo = self.dry_run or self.player.faixa_atual is not None or self.player.tocando
        if headless_ativo:
            if acao == "play_pause":
                self.player.alternar_pause()
                direta = (
                    ("Reprodução pausada." if self.player.pausado else "Reprodução retomada.")
                    if ctx.eh_formal
                    else ("Pausei a música aqui." if self.player.pausado else "Voltei a tocar a música!")
                )
            elif acao == "parar":
                self.player.parar()
                direta = "Reprodução interrompida." if ctx.eh_formal else "Parei a música e limpei a fila."
            elif acao == "proxima":
                prox = self.player.proxima()
                if prox:
                    direta = f"Avançando para {prox['titulo']}." if ctx.eh_formal else f"Pulei pra {prox['titulo']}!"
                else:
                    direta = "A fila de músicas acabou." if ctx.eh_formal else "Não tem mais nada na fila aqui."
            elif acao == "anterior":
                ant = self.player.anterior()
                if ant:
                    direta = f"Retornando para {ant['titulo']}." if ctx.eh_formal else f"Voltei pra {ant['titulo']}."
                else:
                    direta = "Não há faixa anterior no histórico." if ctx.eh_formal else "Não tem música anterior pra voltar não."
            elif acao == "volume_mais":
                vol = self.player.ajustar_volume(+0.15)
                direta = f"Volume em {int(round(vol * 100))}%." if ctx.eh_formal else f"Subi o som pra {int(round(vol * 100))}%!"
            elif acao == "volume_menos":
                vol = self.player.ajustar_volume(-0.15)
                direta = f"Volume em {int(round(vol * 100))}%." if ctx.eh_formal else f"Abaixei o som pra {int(round(vol * 100))}%."
            elif acao == "mudo":
                mu = self.player.definir_mudo()
                direta = (
                    ("Áudio silenciado." if mu else "Áudio restaurado.")
                    if ctx.eh_formal
                    else ("Mutei a música aqui." if mu else "Tirei do mudo!")
                )
            else:
                direta = "Comando aplicado no player."
            return ResultadoSkill(
                skill_id=self.id,
                sucesso=True,
                resumo_contexto=direta,
                resposta_direta=direta,
                precisa_llm=False,
                metadados={"acao": acao, "headless": True, "volume": self.player.volume, "dry_run": self.dry_run},
            )

        # Fallback para teclas nativas do Windows se nenhum stream headless estiver tocando
        reps = 4 if acao in ("volume_mais", "volume_menos") else 1
        ok = _enviar_tecla_midia_windows(acao, repeticoes=reps)
        mensagens_formal = {
            "play_pause": "Reprodução alternada.",
            "proxima": "Avançando para a próxima faixa.",
            "anterior": "Retornando para a faixa anterior.",
            "parar": "Reprodução interrompida.",
            "volume_mais": "Volume aumentado.",
            "volume_menos": "Volume reduzido.",
            "mudo": "Áudio silenciado.",
        }
        mensagens_normal = {
            "play_pause": "Bati no play pause aqui!",
            "proxima": "Pulei pra próxima faixa!",
            "anterior": "Voltei pra música anterior.",
            "parar": "Parei a música aqui.",
            "volume_mais": "Subi o volume aqui!",
            "volume_menos": "Dei uma abaixada no volume.",
            "mudo": "Mudei o som pro mudo aqui.",
        }
        tabela = mensagens_formal if ctx.eh_formal else mensagens_normal
        resp = tabela.get(acao, "Comando de mídia enviado.")
        if not ok:
            resp = "Não consegui acionar as teclas de mídia nesse sistema operacional."

        return ResultadoSkill(
            skill_id=self.id,
            sucesso=ok,
            resumo_contexto=f"Controle de mídia executado: acao={acao}, sucesso={ok}.",
            resposta_direta=resp,
            precisa_llm=False,
            metadados={"acao": acao, "headless": False, "dry_run": self.dry_run},
        )
