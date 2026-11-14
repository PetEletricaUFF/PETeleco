#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 0/3 — Captura Continua de Webcam + HUD Ao Vivo (YuNet + SFace).

Mantem a webcam aberta em thread dedicada para evitar atraso de abertura/exposicao
por turno, fornece o frame JPEG mais recente sob demanda e exibe opcionalmente uma
janela HUD com bounding boxes faciais, estado corporal e legendas da conversa.
"""

import base64
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import unicodedata


if sys.platform.startswith("linux") and "QT_QPA_FONTDIR" not in os.environ:
    if Path("/usr/share/fonts").exists():
        os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"


def _texto_hud(texto: str) -> str:
    """Converte texto UTF-8 para ASCII legivel nas fontes Hershey do OpenCV."""
    decomp = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in decomp if not unicodedata.combining(c) and 32 <= ord(c) < 127)


def _quebrar_linhas(texto: str, largura: int = 68, max_linhas: int = 2) -> List[str]:
    palavras = _texto_hud(texto).split()
    if not palavras:
        return []
    linhas: List[str] = []
    atual = ""
    for p in palavras:
        cand = f"{atual} {p}".strip() if atual else p
        if len(cand) <= largura:
            atual = cand
        else:
            linhas.append(atual)
            atual = p
            if len(linhas) >= max_linhas:
                break
    if atual and len(linhas) < max_linhas:
        linhas.append(atual)
    if len(linhas) == max_linhas and len(" ".join(palavras)) > sum(len(x) for x in linhas) + max_linhas - 1:
        if len(linhas[-1]) > 3:
            linhas[-1] = linhas[-1][:-3].rstrip() + "..."
    return linhas


def codificar_frame_b64(frame: Any, max_dim: int = 640, qualidade: int = 82) -> Optional[str]:
    """Redimensiona frame BGR ate `max_dim` e codifica em JPEG base64."""
    if frame is None:
        return None
    try:
        import cv2  # type: ignore

        h, w = frame.shape[:2]
        maior = max(h, w)
        img = frame
        if maior > max_dim:
            escala = float(max_dim) / float(maior)
            img = cv2.resize(
                frame,
                (max(1, int(round(w * escala))), max(1, int(round(h * escala)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(qualidade)])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


class WebcamLive:
    """Gerencia webcam local em thread dedicada com preview HUD opcional."""

    JANELA_TITULO = "PETeleco — Simulador Local (Webcam + Mic)"

    def __init__(
        self,
        camera_idx: int = 0,
        detector_faces: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
        preview: bool = False,
        intervalo_face_s: float = 0.40,
    ) -> None:
        self.camera_idx = int(camera_idx)
        self._detector_faces = detector_faces
        self._intervalo_face_s = max(0.15, float(intervalo_face_s))

        self._lock = threading.Lock()
        self._face_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._disponivel = False
        self._frame_atual: Any = None
        self._faces_atuais: List[Dict[str, Any]] = []
        self._preview_ativo = bool(preview)
        self._janela_aberta = False
        self._fechado_pelo_usuario = False

        self._hud: Dict[str, str] = {
            "status": "PRONTO",
            "pessoa": "auto",
            "modo": "NORMAL",
            "corpo": "chao/parado",
            "jev": "ON",
            "tts": "OFF",
            "fala_user": "",
            "fala_robo": "",
        }
        self._thread: Optional[threading.Thread] = None

    @property
    def disponivel(self) -> bool:
        return self._disponivel

    @property
    def fechado_pelo_usuario(self) -> bool:
        return self._fechado_pelo_usuario

    @property
    def preview_ativo(self) -> bool:
        with self._lock:
            return self._preview_ativo

    def iniciar(self, timeout_s: float = 4.0) -> bool:
        """Abre a webcam e inicia captura continua em background."""
        if self._thread is not None and self._thread.is_alive():
            return self._disponivel
        self._stop_event.clear()
        self._ready_event.clear()
        self._fechado_pelo_usuario = False
        self._thread = threading.Thread(target=self._loop_captura, name="peteleco-webcam", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=timeout_s)
        return self._disponivel

    def ativar_preview(self, ativo: bool) -> None:
        with self._lock:
            self._preview_ativo = bool(ativo)
            if ativo:
                self._fechado_pelo_usuario = False

    def atualizar_hud(self, **campos: str) -> None:
        with self._lock:
            for k, v in campos.items():
                if v is not None:
                    self._hud[k] = str(v)

    def capturar_frame(self) -> Any:
        with self._lock:
            if self._frame_atual is None:
                return None
            return self._frame_atual.copy()

    def capturar_b64(self, max_dim: int = 640, qualidade: int = 82) -> Optional[str]:
        frame = self.capturar_frame()
        return codificar_frame_b64(frame, max_dim=max_dim, qualidade=qualidade)

    def obter_faces_atuais(self) -> List[Dict[str, Any]]:
        """Executa deteccao imediata sobre o frame atual (thread-safe)."""
        frame = self.capturar_frame()
        if frame is None or self._detector_faces is None:
            return []
        with self._face_lock:
            faces = self._detector_faces(frame)
        with self._lock:
            self._faces_atuais = list(faces)
        return faces

    def fechar(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._disponivel = False

    def _abrir_capture(self) -> Any:
        import cv2  # type: ignore

        cap = None
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(self.camera_idx)
        return cap

    def _desenhar_hud(self, frame: Any, faces: List[Dict[str, Any]], hud: Dict[str, str]) -> Any:
        import cv2  # type: ignore

        vis = frame.copy()
        h, w = vis.shape[:2]

        for f in faces:
            x, y, bw, bh = f.get("bbox", (0, 0, 0, 0))
            nome = str(f.get("nome", "desconhecido"))
            label = _texto_hud(str(f.get("label", nome)))
            cor = (80, 230, 120) if nome != "desconhecido" else (40, 190, 255)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), cor, 2)
            y_txt = max(24, y - 8)
            cv2.rectangle(vis, (x, y_txt - 18), (min(w, x + max(130, len(label) * 9 + 12)), y_txt + 4), cor, -1)
            cv2.putText(vis, label, (x + 5, y_txt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 2)

        # Barra superior de telemetria
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (18, 22, 28), -1)
        tem_rodape = bool(hud.get("fala_user") or hud.get("fala_robo"))
        if tem_rodape:
            cv2.rectangle(overlay, (0, max(0, h - 84)), (w, h), (18, 22, 28), -1)
        vis = cv2.addWeighted(overlay, 0.72, vis, 0.28, 0)

        status = _texto_hud(hud.get("status", "PRONTO"))
        cor_status = (80, 230, 120)
        if "OUVINDO" in status:
            cor_status = (60, 200, 255)
        elif "PENSANDO" in status:
            cor_status = (80, 165, 255)
        elif "FALANDO" in status:
            cor_status = (240, 180, 60)

        linha1 = _texto_hud(f"PETeleco [{status}] | ID: {hud.get('pessoa', 'auto')}")
        linha2 = _texto_hud(
            f"Corpo: {hud.get('corpo', '')} | Modo: {hud.get('modo', '')} | "
            f"Jev: {hud.get('jev', '')} | TTS: {hud.get('tts', '')} | [Q] fechar"
        )
        cv2.putText(vis, linha1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cor_status, 1, cv2.LINE_AA)
        cv2.putText(vis, linha2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 225, 235), 1, cv2.LINE_AA)

        if tem_rodape:
            y_base = h - 62
            if hud.get("fala_user"):
                for ln in _quebrar_linhas(f"Voce: {hud['fala_user']}", largura=66, max_linhas=1):
                    cv2.putText(vis, ln, (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 235, 255), 1, cv2.LINE_AA)
                    y_base += 20
            if hud.get("fala_robo"):
                for ln in _quebrar_linhas(f"PETeleco: {hud['fala_robo']}", largura=66, max_linhas=2):
                    cv2.putText(vis, ln, (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (165, 255, 190), 1, cv2.LINE_AA)
                    y_base += 19

        return vis

    def _loop_captura(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            self._disponivel = False
            self._ready_event.set()
            return

        cap = None
        try:
            cap = self._abrir_capture()
            if cap is None or not cap.isOpened():
                self._disponivel = False
                self._ready_event.set()
                return

            # Descarta frames iniciais para estabilizar exposicao automatica
            primeiro_ok = False
            for _ in range(5):
                ok, frm = cap.read()
                if ok and frm is not None:
                    with self._lock:
                        self._frame_atual = frm
                    primeiro_ok = True
                time.sleep(0.03)

            self._disponivel = primeiro_ok
            self._ready_event.set()
            if not primeiro_ok:
                return

            t_ultimo_face = 0.0
            while not self._stop_event.is_set():
                ok, frm = cap.read()
                if ok and frm is not None:
                    with self._lock:
                        self._frame_atual = frm
                else:
                    time.sleep(0.03)
                    continue

                with self._lock:
                    quer_preview = self._preview_ativo
                    hud_snap = dict(self._hud)

                if quer_preview:
                    agora = time.monotonic()
                    if self._detector_faces is not None and (agora - t_ultimo_face) >= self._intervalo_face_s:
                        t_ultimo_face = agora
                        with self._face_lock:
                            faces = self._detector_faces(frm)
                        with self._lock:
                            self._faces_atuais = list(faces)
                    else:
                        with self._lock:
                            faces = list(self._faces_atuais)

                    vis = self._desenhar_hud(frm, faces, hud_snap)
                    if not self._janela_aberta:
                        cv2.namedWindow(self.JANELA_TITULO, cv2.WINDOW_NORMAL)
                        self._janela_aberta = True
                    cv2.imshow(self.JANELA_TITULO, vis)
                    tecla = cv2.waitKey(25) & 0xFF
                    if tecla in (ord("q"), 27):
                        with self._lock:
                            self._preview_ativo = False
                            self._fechado_pelo_usuario = True
                else:
                    if self._janela_aberta:
                        cv2.destroyWindow(self.JANELA_TITULO)
                        cv2.waitKey(1)
                        self._janela_aberta = False
                    time.sleep(0.04)
        except Exception:
            self._disponivel = False
            self._ready_event.set()
        finally:
            if self._janela_aberta:
                try:
                    import cv2  # type: ignore

                    cv2.destroyWindow(self.JANELA_TITULO)
                    cv2.waitKey(1)
                except Exception:
                    pass
                self._janela_aberta = False
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
