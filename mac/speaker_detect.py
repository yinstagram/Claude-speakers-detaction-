"""
Speaker Detection for Mac (M-series optimised)
================================================

Real-time speaker diarisation that notifies the user whenever a
different person starts speaking. Built for hearing-impaired users
who need visual + notification cues on speaker change.

Pipeline:
    🎤 mic (sounddevice, 16 kHz mono)
        → rolling 3-second buffer
        → SpeechBrain ECAPA-TDNN embedding (192-dim, MPS accelerated)
        → cosine similarity against known profiles
        → PyQt6 full-screen UI + Mac native notification on change

Why SpeechBrain ECAPA-TDNN:
    - SOTA speaker verification model
    - Freely downloadable (no HuggingFace token needed)
    - Runs on Apple Silicon via PyTorch MPS
    - ~20 MB, loads in seconds
"""

from __future__ import annotations

import os
import sys
import time
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import sounddevice as sd
import torch

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QObject
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QFrame,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3.0
PROCESS_INTERVAL = 1.2           # run detection every 1.2 s
SIMILARITY_THRESHOLD = 0.55      # cosine sim above this = same speaker
FREEZE_AFTER = 8                 # frozen profile after N successful matches
RMS_SILENCE_THRESHOLD = 0.003    # below this = silence
COLORS = [
    "#FF6B35", "#4ECDC4", "#FFE66D", "#A78BFA",
    "#F38181", "#34D399", "#FCBAD3", "#60A5FA",
]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Speaker engine
# ---------------------------------------------------------------------------
@dataclass
class SpeakerProfile:
    embedding: np.ndarray
    frozen: bool = False
    history: List[np.ndarray] = field(default_factory=list)


class SpeakerEngine:
    """ECAPA-TDNN embeddings + online clustering by cosine similarity."""

    def __init__(self) -> None:
        self.device = pick_device()
        print(f"[engine] using device: {self.device}", flush=True)
        # Lazy-import to let the UI show a loading screen first.
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/speaker_detect/ecapa"),
            run_opts={"device": str(self.device)},
        )
        self.profiles: List[SpeakerProfile] = []
        self.current_speaker: int = -1

    def reset(self) -> None:
        self.profiles.clear()
        self.current_speaker = -1

    @torch.no_grad()
    def embed(self, audio: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(wav).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def process(self, audio: np.ndarray) -> Optional[dict]:
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        db = int(20 * np.log10(max(rms, 1e-10)))
        if rms < RMS_SILENCE_THRESHOLD:
            return {"type": "silence", "debug": f"{db}dB silence"}

        emb = self.embed(audio)

        if not self.profiles:
            self.profiles.append(SpeakerProfile(embedding=emb.copy(),
                                                history=[emb.copy()]))
            self.current_speaker = 0
            return {
                "type": "speaker",
                "index": 0,
                "changed": True,
                "similarity": 1.0,
                "debug": f"{db}dB | Speaker A registered",
            }

        sims = [float(np.dot(emb, p.embedding)) for p in self.profiles]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        sims_str = " ".join(f"{chr(65+i)}={s:.2f}" for i, s in enumerate(sims))

        if best_sim >= SIMILARITY_THRESHOLD:
            changed = best_idx != self.current_speaker
            self.current_speaker = best_idx
            p = self.profiles[best_idx]
            if not p.frozen:
                p.history.append(emb.copy())
                if len(p.history) >= FREEZE_AFTER:
                    avg = np.mean(np.stack(p.history), axis=0)
                    n = np.linalg.norm(avg)
                    if n > 0:
                        avg /= n
                    self.profiles[best_idx] = SpeakerProfile(
                        embedding=avg, frozen=True, history=[]
                    )
            return {
                "type": "speaker",
                "index": best_idx,
                "changed": changed,
                "similarity": best_sim,
                "debug": f"{db}dB | {sims_str} | thr={SIMILARITY_THRESHOLD:.2f}",
            }
        else:
            new_idx = len(self.profiles)
            self.profiles.append(SpeakerProfile(embedding=emb.copy(),
                                                history=[emb.copy()]))
            self.current_speaker = new_idx
            return {
                "type": "speaker",
                "index": new_idx,
                "changed": True,
                "similarity": best_sim,
                "debug": f"{db}dB | {sims_str} | NEW Speaker {chr(65+new_idx)}",
            }


# ---------------------------------------------------------------------------
# Audio capture thread
# ---------------------------------------------------------------------------
class AudioCapture(threading.Thread):
    def __init__(self, out_queue: "queue.Queue[np.ndarray]") -> None:
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.running = False
        self.buffer = np.zeros(int(SAMPLE_RATE * BUFFER_SECONDS), dtype=np.float32)
        self.write_pos = 0
        self.filled = False
        self.lock = threading.Lock()
        self.latest_rms = 0.0

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[audio] {status}", flush=True)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        with self.lock:
            n = len(mono)
            end = self.write_pos + n
            if end <= len(self.buffer):
                self.buffer[self.write_pos:end] = mono
            else:
                first = len(self.buffer) - self.write_pos
                self.buffer[self.write_pos:] = mono[:first]
                self.buffer[: n - first] = mono[first:]
                self.filled = True
            self.write_pos = (self.write_pos + n) % len(self.buffer)
            if self.write_pos == 0:
                self.filled = True
            self.latest_rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))

    def snapshot(self) -> Optional[np.ndarray]:
        with self.lock:
            if not self.filled and self.write_pos < SAMPLE_RATE:
                return None
            if self.filled:
                return np.concatenate(
                    (self.buffer[self.write_pos:], self.buffer[: self.write_pos])
                ).copy()
            return self.buffer[: self.write_pos].copy()

    def run(self) -> None:
        self.running = True
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        ):
            while self.running:
                time.sleep(0.05)

    def stop(self) -> None:
        self.running = False


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------
def notify_mac(title: str, message: str) -> None:
    """Send a native macOS notification via osascript (always available)."""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}" sound name "Glass"',
            ],
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[notify] {exc}", flush=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class Bridge(QObject):
    """Marshalls messages from worker thread to Qt main thread."""
    speaker = pyqtSignal(dict)
    status = pyqtSignal(str)
    model_ready = pyqtSignal()
    model_error = pyqtSignal(str)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speaker Detect")
        self.resize(900, 700)
        self._build_ui()

        self.bridge = Bridge()
        self.bridge.speaker.connect(self.on_speaker)
        self.bridge.status.connect(self.on_status)
        self.bridge.model_ready.connect(self.on_model_ready)
        self.bridge.model_error.connect(self.on_model_error)

        self.engine: Optional[SpeakerEngine] = None
        self.capture: Optional[AudioCapture] = None
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.last_speaker = -1
        self.analysis_count = 0

        # Load model asynchronously
        threading.Thread(target=self._load_model, daemon=True).start()

        # Energy meter refresh
        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._update_meter)
        self.meter_timer.start(60)

    # -- UI construction -----------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root.setStyleSheet("background-color: #000;")

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Energy bar
        self.energy_container = QFrame()
        self.energy_container.setFixedHeight(6)
        self.energy_container.setStyleSheet("background-color: #111;")
        energy_layout = QHBoxLayout(self.energy_container)
        energy_layout.setContentsMargins(0, 0, 0, 0)
        self.energy_bar = QFrame()
        self.energy_bar.setStyleSheet("background-color: #444;")
        self.energy_bar.setFixedWidth(0)
        energy_layout.addWidget(self.energy_bar, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.energy_container)

        # Big speaker letter
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.letter = QLabel("?")
        self.letter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont("Helvetica", 260)
        f.setWeight(QFont.Weight.Black)
        self.letter.setFont(f)
        self.letter.setStyleSheet("color: #333;")
        center_layout.addWidget(self.letter)

        self.name_label = QLabel("LOADING MODEL...")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fn = QFont("Helvetica", 22)
        fn.setBold(True)
        self.name_label.setFont(fn)
        self.name_label.setStyleSheet("color: #555; letter-spacing: 4px;")
        center_layout.addWidget(self.name_label)

        self.status_label = QLabel("Downloading ECAPA-TDNN weights...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Helvetica", 13))
        self.status_label.setStyleSheet("color: #777; letter-spacing: 2px; margin-top: 12px;")
        center_layout.addWidget(self.status_label)

        self.debug_label = QLabel("")
        self.debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.debug_label.setFont(QFont("Menlo", 11))
        self.debug_label.setStyleSheet("color: #555; margin-top: 8px;")
        center_layout.addWidget(self.debug_label)

        layout.addWidget(center, 1)

        # Flash overlay (covers center widget)
        self.flash_overlay = QFrame(center)
        self.flash_overlay.setStyleSheet("background-color: white;")
        self.flash_overlay.setGeometry(0, 0, 10, 10)
        self.flash_overlay.hide()
        self._flash_anim = QPropertyAnimation(self.flash_overlay, b"windowOpacity")

        # Chips bar
        self.chips_bar = QWidget()
        self.chips_bar.setStyleSheet("background-color: #0a0a0a;")
        self.chips_layout = QHBoxLayout(self.chips_bar)
        self.chips_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chips_bar.setFixedHeight(70)
        layout.addWidget(self.chips_bar)

        # Buttons
        buttons = QWidget()
        bl = QHBoxLayout(buttons)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.btn_start = QPushButton("LOADING...")
        self.btn_start.setFixedHeight(72)
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #003300; color: #00FF88;"
            " font-size: 20px; font-weight: bold; letter-spacing: 4px;"
            " border: none; }"
            "QPushButton:disabled { color: #444; background-color: #111; }"
        )
        self.btn_start.clicked.connect(self.toggle)
        bl.addWidget(self.btn_start, 1)

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setFixedHeight(72)
        self.btn_reset.setFixedWidth(140)
        self.btn_reset.setStyleSheet(
            "QPushButton { background-color: #111; color: #888;"
            " font-size: 18px; font-weight: bold; letter-spacing: 3px;"
            " border: none; }"
        )
        self.btn_reset.clicked.connect(self.reset_all)
        bl.addWidget(self.btn_reset)
        layout.addWidget(buttons)

    # -- model loading -------------------------------------------------------
    def _load_model(self) -> None:
        try:
            self.bridge.status.emit("Loading ECAPA-TDNN (MPS)...")
            self.engine = SpeakerEngine()
            self.bridge.model_ready.emit()
        except Exception as exc:  # pragma: no cover
            self.bridge.model_error.emit(str(exc))

    def on_model_ready(self) -> None:
        self.name_label.setText("TAP START")
        self.status_label.setText(
            f"ECAPA-TDNN READY · {self.engine.device if self.engine else 'cpu'}"
        )
        self.debug_label.setText("192-dim speaker embeddings · click START to begin")
        self.btn_start.setText("START")
        self.btn_start.setEnabled(True)

    def on_model_error(self, msg: str) -> None:
        self.name_label.setText("MODEL FAILED")
        self.name_label.setStyleSheet("color: #FF4444; letter-spacing: 4px;")
        self.status_label.setText(f"ERROR: {msg[:120]}")
        self.debug_label.setText("Check internet + `pip install -r requirements.txt`.")

    # -- start / stop --------------------------------------------------------
    def toggle(self) -> None:
        if self.running:
            self.stop_capture()
        else:
            self.start_capture()

    def start_capture(self) -> None:
        if self.engine is None:
            return
        self.capture = AudioCapture(self.audio_queue)
        self.capture.start()
        self.running = True
        self.analysis_count = 0
        self.last_speaker = -1
        self.btn_start.setText("STOP")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #330000; color: #FF4444;"
            " font-size: 20px; font-weight: bold; letter-spacing: 4px; border: none; }"
        )
        self.letter.setText("...")
        self.name_label.setText("LISTENING")
        self.name_label.setStyleSheet("color: #555; letter-spacing: 4px;")
        self.status_label.setText("RECORDING — first analysis in 1.5 s")

        self.worker_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.worker_thread.start()

    def stop_capture(self) -> None:
        self.running = False
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.btn_start.setText("START")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #003300; color: #00FF88;"
            " font-size: 20px; font-weight: bold; letter-spacing: 4px; border: none; }"
        )
        self.status_label.setText("STOPPED")

    def reset_all(self) -> None:
        if self.engine:
            self.engine.reset()
        self.last_speaker = -1
        self.analysis_count = 0
        self.letter.setText("?")
        self.letter.setStyleSheet("color: #333;")
        self.name_label.setText("TAP START" if self.engine else "LOADING...")
        self.name_label.setStyleSheet("color: #555; letter-spacing: 4px;")
        self.status_label.setText("READY" if self.engine else "LOADING MODEL...")
        self.debug_label.setText("")
        self._clear_chips()

    # -- analysis loop -------------------------------------------------------
    def _analysis_loop(self) -> None:
        while self.running:
            time.sleep(PROCESS_INTERVAL)
            if not self.running or not self.capture or self.engine is None:
                continue
            audio = self.capture.snapshot()
            if audio is None:
                continue
            try:
                result = self.engine.process(audio)
            except Exception as exc:  # pragma: no cover
                self.bridge.status.emit(f"ERR: {exc}")
                continue
            self.analysis_count += 1
            if result is None:
                continue
            if result["type"] == "silence":
                self.bridge.status.emit(
                    f"ACTIVE — no voice · #{self.analysis_count} · {result['debug']}"
                )
            else:
                self.bridge.speaker.emit(result)

    # -- Qt slots ------------------------------------------------------------
    def on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def on_speaker(self, result: dict) -> None:
        idx = result["index"]
        color = COLORS[idx % len(COLORS)]
        letter = chr(ord("A") + idx)
        self.letter.setText(letter)
        self.letter.setStyleSheet(f"color: {color};")
        self.name_label.setText(f"SPEAKER {letter}")
        self.name_label.setStyleSheet(f"color: {color}; letter-spacing: 4px;")
        self.status_label.setText(
            f"SPEAKER {letter} · {result['similarity'] * 100:.0f}% match · #{self.analysis_count}"
        )
        self.debug_label.setText(result["debug"])
        self.energy_bar.setStyleSheet(f"background-color: {color};")

        if result["changed"] and self.last_speaker != -1:
            from_letter = chr(ord("A") + self.last_speaker)
            self._flash()
            notify_mac(
                f"Speaker Changed: {from_letter} → {letter}",
                f"Speaker {letter} is now talking",
            )
        self.last_speaker = idx
        self._update_chips(len(self.engine.profiles) if self.engine else 0, idx)

    # -- visual helpers ------------------------------------------------------
    def _flash(self) -> None:
        parent = self.flash_overlay.parentWidget()
        if parent:
            self.flash_overlay.setGeometry(0, 0, parent.width(), parent.height())
        self.flash_overlay.show()
        self.flash_overlay.setStyleSheet("background-color: white;")
        self.flash_overlay.raise_()
        QTimer.singleShot(120, self.flash_overlay.hide)
        QTimer.singleShot(240, lambda: (
            self.flash_overlay.setStyleSheet("background-color: white;"),
            self.flash_overlay.show(),
        ))
        QTimer.singleShot(360, self.flash_overlay.hide)

    def _update_meter(self) -> None:
        if not self.capture:
            self.energy_bar.setFixedWidth(0)
            return
        rms = self.capture.latest_rms
        db = 20 * np.log10(max(rms, 1e-10))
        pct = max(0.0, min(1.0, (db + 60) / 40))
        self.energy_bar.setFixedWidth(int(self.energy_container.width() * pct))

    def _clear_chips(self) -> None:
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _update_chips(self, total: int, active: int) -> None:
        self._clear_chips()
        for i in range(total):
            color = COLORS[i % len(COLORS)]
            chip = QLabel(chr(ord("A") + i))
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setFixedSize(48, 48)
            is_active = i == active
            border = 3 if is_active else 1
            bg = color + "22" if is_active else "#0A0A0A"
            chip.setStyleSheet(
                f"QLabel {{ color: {color}; background-color: {bg};"
                f" border: {border}px solid {color if is_active else '#222'};"
                f" border-radius: 10px; font-size: 22px; font-weight: bold; }}"
            )
            if not is_active:
                chip.setGraphicsEffect(None)
            self.chips_layout.addWidget(chip)

    # -- cleanup -------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self.running = False
        if self.capture:
            self.capture.stop()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Speaker Detect")

    # dark palette
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#000"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#FFF"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
