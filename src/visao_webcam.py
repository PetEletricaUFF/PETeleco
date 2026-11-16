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
        deteccao_continua: bool = False,
        intervalo_face_s: float = 0.35,
    ) -> None:
        self.camera_idx = int(camera_idx)
        self._detector_faces = detector_faces
        self._intervalo_face_s = max(0.15, float(intervalo_face_s))

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._face_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._disponivel = False
        self._frame_atual: Any = None
        self._frame_seq: int = 0
        self._fps_atual: float = 0.0
        self._face_ms: float = 0.0
        self._resolucao: tuple[int, int] = (0, 0)
        self._faces_atuais: List[Dict[str, Any]] = []
        self._preview_ativo = bool(preview)
        self._deteccao_continua = bool(deteccao_continua)
        self._janela_aberta = False
        self._fechado_pelo_usuario = False
        self._mjpeg_cache_seq: int = -1
        self._mjpeg_cache_bytes: Optional[bytes] = None
        self._erro_camera: str = ""

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

    @property
    def deteccao_continua_ativa(self) -> bool:
        with self._lock:
            return self._deteccao_continua

    def iniciar(self, timeout_s: float = 4.0) -> bool:
        """Abre a webcam e inicia captura continua em background."""
        if self._thread is not None and self._thread.is_alive():
            return self._disponivel
        self._stop_event.clear()
        self._ready_event.clear()
        self._fechado_pelo_usuario = False
        self._erro_camera = ""
        self._thread = threading.Thread(target=self._loop_captura, name="peteleco-webcam", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=timeout_s)
        return self._disponivel

    def ativar_preview(self, ativo: bool) -> None:
        with self._lock:
            self._preview_ativo = bool(ativo)
            if ativo:
                self._fechado_pelo_usuario = False

    def ativar_deteccao_continua(self, ativo: bool) -> None:
        with self._lock:
            self._deteccao_continua = bool(ativo)
            if not ativo and not self._preview_ativo:
                self._faces_atuais = []
                self._face_ms = 0.0

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
        t0 = time.perf_counter()
        with self._face_lock:
            faces = self._detector_faces(frame)
        face_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._faces_atuais = list(faces)
            self._face_ms = face_ms
        return faces

    def aguardar_novo_frame(self, ultimo_seq: int, timeout_s: float = 0.5) -> int:
        """Bloqueia ate chegar um novo frame da webcam ou atingir timeout."""
        with self._frame_cond:
            self._frame_cond.wait_for(
                lambda: self._stop_event.is_set() or self._frame_seq != ultimo_seq,
                timeout=timeout_s,
            )
            return self._frame_seq

    def obter_telemetria(self) -> Dict[str, Any]:
        """Retorna snapshot estruturado da webcam e deteccoes YuNet+SFace."""
        with self._lock:
            faces_serial = []
            for f in self._faces_atuais:
                faces_serial.append(
                    {
                        "bbox": [int(v) for v in f.get("bbox", (0, 0, 0, 0))],
                        "landmarks": [[int(p[0]), int(p[1])] for p in f.get("landmarks", [])],
                        "det_score": float(f.get("det_score", 1.0)),
                        "nome": str(f.get("nome", "desconhecido")),
                        "candidato": str(f.get("candidato", "")),
                        "score": float(f.get("score", 0.0)),
                        "limiar": float(f.get("limiar", 0.363)),
                        "label": str(f.get("label", "desconhecido")),
                    }
                )
            return {
                "disponivel": self._disponivel,
                "erro": self._erro_camera,
                "resolucao": [int(self._resolucao[0]), int(self._resolucao[1])],
                "fps": round(self._fps_atual, 1),
                "face_ms": round(self._face_ms, 1),
                "frame_seq": self._frame_seq,
                "deteccao_continua": self._deteccao_continua,
                "faces": faces_serial,
            }

    def obter_frame_mjpeg_bytes(self, qualidade: int = 80, anotar_faces: bool = True) -> Optional[bytes]:
        """Codifica frame mais recente em JPEG com anotacoes YuNet+SFace (cache por frame_seq)."""
        try:
            import cv2  # type: ignore
        except Exception:
            return None

        with self._lock:
            if self._frame_atual is None:
                return None
            seq = self._frame_seq
            if anotar_faces and self._mjpeg_cache_seq == seq and self._mjpeg_cache_bytes is not None:
                return self._mjpeg_cache_bytes
            frm = self._frame_atual.copy()
            faces = list(self._faces_atuais)
            face_ms = self._face_ms
            fps = self._fps_atual

        if anotar_faces:
            frm = self._desenhar_caixas_faciais(frm, faces, face_ms=face_ms, fps=fps)

        ok, buf = cv2.imencode(".jpg", frm, [int(cv2.IMWRITE_JPEG_QUALITY), int(qualidade)])
        if not ok:
            return None
        jpg = buf.tobytes()
        if anotar_faces:
            with self._lock:
                if self._frame_seq == seq:
                    self._mjpeg_cache_seq = seq
                    self._mjpeg_cache_bytes = jpg
        return jpg

    def fechar(self) -> None:
        self._stop_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
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

    @staticmethod
    def _desenhar_caixas_faciais(
        vis: Any,
        faces: List[Dict[str, Any]],
        face_ms: float = 0.0,
        fps: float = 0.0,
    ) -> Any:
        import cv2  # type: ignore

        h, w = vis.shape[:2]
        for f in faces:
            x, y, bw, bh = f.get("bbox", (0, 0, 0, 0))
            nome = str(f.get("nome", "desconhecido"))
            score = float(f.get("score", 0.0))
            cand = str(f.get("candidato", ""))
            reconhecido = nome != "desconhecido"
            cor = (80, 230, 120) if reconhecido else (40, 190, 255)

            cv2.rectangle(vis, (x, y), (x + bw, y + bh), cor, 2)
            canto = max(8, min(18, bw // 5, bh // 5))
            for cx, dx in ((x, canto), (x + bw, -canto)):
                for cy, dy in ((y, canto), (y + bh, -canto)):
                    cv2.line(vis, (cx, cy), (cx + dx, cy), cor, 3)
                    cv2.line(vis, (cx, cy), (cx, cy + dy), cor, 3)

            for lx, ly in f.get("landmarks", []):
                if 0 <= lx < w and 0 <= ly < h:
                    cv2.circle(vis, (lx, ly), 2, (255, 225, 110), -1, cv2.LINE_AA)

            if reconhecido:
                label = _texto_hud(str(f.get("label", nome)))
            elif cand and score > 0.0:
                label = _texto_hud(f"desconhecido | ~{cand} {score:.2f}")
            else:
                label = "desconhecido"

            y_txt = max(24, y - 8)
            largura_tag = min(w - x, max(120, len(label) * 9 + 14))
            cv2.rectangle(vis, (x, y_txt - 18), (x + largura_tag, y_txt + 4), cor, -1)
            cv2.putText(vis, label, (x + 5, y_txt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 2)

        if fps > 0.0 or face_ms > 0.0:
            tag_perf = _texto_hud(f"CAM {w}x{h} | {fps:.1f} FPS | SFace {face_ms:.1f} ms")
            cv2.rectangle(vis, (8, 8), (8 + len(tag_perf) * 7 + 14, 30), (16, 20, 28), -1)
            cv2.putText(vis, tag_perf, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 225, 255), 1, cv2.LINE_AA)
        return vis

    def _desenhar_hud(self, frame: Any, faces: List[Dict[str, Any]], hud: Dict[str, str]) -> Any:
        import cv2  # type: ignore

        vis = self._desenhar_caixas_faciais(frame.copy(), faces)
        h, w = vis.shape[:2]

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
        except Exception as exc:
            self._erro_camera = f"OpenCV indisponivel: {exc}"
            self._disponivel = False
            self._ready_event.set()
            return

        cap = None
        try:
            cap = self._abrir_capture()
            if cap is None or not cap.isOpened():
                self._erro_camera = f"Nao foi possivel abrir webcam (indice {self.camera_idx})."
                self._disponivel = False
                self._ready_event.set()
                return

            # Descarta frames iniciais para estabilizar exposicao automatica
            primeiro_ok = False
            for _ in range(5):
                ok, frm = cap.read()
                if ok and frm is not None:
                    h, w = frm.shape[:2]
                    with self._lock:
                        self._frame_atual = frm
                        self._resolucao = (w, h)
                        self._frame_seq += 1
                    primeiro_ok = True
                time.sleep(0.03)

            if not primeiro_ok:
                self._erro_camera = "Webcam abriu, mas nao retornou frames."
            self._disponivel = primeiro_ok
            self._ready_event.set()
            if not primeiro_ok:
                return

            t_ultimo_face = 0.0
            t_ultimo_frame = time.monotonic()
            while not self._stop_event.is_set():
                ok, frm = cap.read()
                agora = time.monotonic()
                if ok and frm is not None:
                    dt = max(1e-3, agora - t_ultimo_frame)
                    t_ultimo_frame = agora
                    fps_inst = 1.0 / dt
                    h, w = frm.shape[:2]
                    with self._lock:
                        self._frame_atual = frm
                        self._resolucao = (w, h)
                        self._frame_seq += 1
                        self._fps_atual = fps_inst if self._fps_atual <= 0 else (0.8 * self._fps_atual + 0.2 * fps_inst)
                else:
                    time.sleep(0.03)
                    continue

                with self._lock:
                    quer_preview = self._preview_ativo
                    quer_deteccao = self._deteccao_continua
                    hud_snap = dict(self._hud)

                if (quer_preview or quer_deteccao) and self._detector_faces is not None:
                    if (agora - t_ultimo_face) >= self._intervalo_face_s:
                        t_ultimo_face = agora
                        t0_f = time.perf_counter()
                        with self._face_lock:
                            faces = self._detector_faces(frm)
                        face_ms = (time.perf_counter() - t0_f) * 1000.0
                        with self._lock:
                            self._faces_atuais = list(faces)
                            self._face_ms = face_ms
                    else:
                        with self._lock:
                            faces = list(self._faces_atuais)
                else:
                    faces = []

                with self._frame_cond:
                    self._frame_cond.notify_all()

                if quer_preview:
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
                    time.sleep(0.025)
        except Exception as exc:
            self._erro_camera = f"Falha na thread da webcam: {exc}"
            self._disponivel = False
            self._ready_event.set()
        finally:
            with self._frame_cond:
                self._frame_cond.notify_all()
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
