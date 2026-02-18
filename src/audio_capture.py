"""
audio_capture.py
─────────────────────────────────────────────────────────────────
Real-time audio capture with WebRTC Voice Activity Detection (VAD).
Python 3.10 · Windows 10/11

Pipeline:
  Microphone → sounddevice callback → ring-buffer of 30ms frames
             → webrtcvad → accumulate voiced frames
             → yield complete utterance as numpy array (16kHz mono int16)

Windows notes:
  - sounddevice uses WASAPI on Windows (low-latency, no extra install)
  - webrtcvad-wheels provides pre-built .pyd for Windows (no MSVC needed)
  - Default mic = Windows default recording device (set in Sound Settings)
"""
from __future__ import annotations

import time
import queue
import threading
import collections
from typing import Generator, Optional, List

import numpy as np
import sounddevice as sd
import webrtcvad

from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000        # Hz  – Whisper and VAD both need 16kHz
CHANNELS      = 1
DTYPE         = "int16"
FRAME_MS      = 30            # ms per VAD frame (10 | 20 | 30 supported)
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1_000)   # 480 samples
FRAME_BYTES   = FRAME_SAMPLES * 2                      # int16 = 2 bytes/sample


class AudioCapture:
    """
    Streams microphone audio through a WebRTC VAD.

    Usage
    -----
    >>> cap = AudioCapture(vad_aggressiveness=2, silence_ms=700)
    >>> for utterance in cap.utterances():
    ...     # utterance is a np.ndarray, shape=(N,), dtype=int16, 16kHz mono
    ...     process(utterance)
    """

    def __init__(
        self,
        vad_aggressiveness: int = 2,    # 0 (lenient) … 3 (aggressive)
        silence_ms: int = 700,          # ms of silence = end of utterance
        min_speech_ms: int = 250,       # discard clips shorter than this
        max_speech_ms: int = 10_000,    # hard cap per utterance
        device: Optional[int] = None,   # None = system default mic
        padding_ms: int = 300,          # voiced padding around speech
    ):
        self.vad              = webrtcvad.Vad(vad_aggressiveness)
        self.silence_frames   = int(silence_ms   / FRAME_MS)
        self.min_frames       = int(min_speech_ms / FRAME_MS)
        self.max_frames       = int(max_speech_ms / FRAME_MS)
        self.padding_frames   = int(padding_ms   / FRAME_MS)
        self.device           = device

        self._raw_q: queue.Queue[bytes] = queue.Queue()
        self._stop_event = threading.Event()

    # ── Internal: sounddevice callback ───────────────────────────────────────
    def _sd_callback(self, indata: np.ndarray, frames: int, t, status):
        """Called by sounddevice in a separate thread for every blocksize frames."""
        if status:
            logger.debug(f"[audio] sounddevice status: {status}")
        # indata may be larger than one VAD frame; split into FRAME_SAMPLES chunks
        flat = indata[:, 0].astype(np.int16)
        raw  = flat.tobytes()
        for i in range(0, len(raw), FRAME_BYTES):
            chunk = raw[i : i + FRAME_BYTES]
            if len(chunk) == FRAME_BYTES:
                self._raw_q.put(chunk)

    # ── Public API ───────────────────────────────────────────────────────────
    def utterances(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields complete utterances as int16 numpy arrays.
        Blocks between utterances (no CPU spin).
        """
        logger.info("[audio] Starting microphone stream (16kHz, mono, int16)")
        logger.info("[audio] Speak now – listening for voice activity…")

        ring   = collections.deque(maxlen=self.padding_frames)   # pre-speech buffer
        voiced = []       # frames confirmed as speech
        num_silent = 0    # consecutive silent frames after speech begins
        in_speech  = False

        with sd.InputStream(
            samplerate  = SAMPLE_RATE,
            channels    = CHANNELS,
            dtype       = DTYPE,
            blocksize   = FRAME_SAMPLES,
            device      = self.device,
            callback    = self._sd_callback,
        ):
            while not self._stop_event.is_set():
                try:
                    frame = self._raw_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

                if not in_speech:
                    ring.append(frame)
                    if is_speech:
                        in_speech  = True
                        num_silent = 0
                        voiced     = list(ring)   # include pre-speech padding
                        logger.debug("[audio] Speech started")
                else:
                    voiced.append(frame)
                    if is_speech:
                        num_silent = 0
                    else:
                        num_silent += 1

                    # ── Hard cap ─────────────────────────────────────────
                    if len(voiced) >= self.max_frames:
                        logger.debug("[audio] Max speech length reached, flushing")
                        yield self._frames_to_array(voiced)
                        voiced, in_speech, num_silent = [], False, 0
                        ring.clear()

                    # ── End of utterance ──────────────────────────────────
                    elif num_silent >= self.silence_frames:
                        if len(voiced) >= self.min_frames:
                            duration_ms = len(voiced) * FRAME_MS
                            logger.debug(
                                f"[audio] Utterance complete ({duration_ms}ms)"
                            )
                            yield self._frames_to_array(voiced)
                        else:
                            logger.debug("[audio] Utterance too short, discarded")
                        voiced, in_speech, num_silent = [], False, 0
                        ring.clear()

    def stop(self):
        self._stop_event.set()

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _frames_to_array(frames: List[bytes]) -> np.ndarray:
        """Concatenate raw bytes frames into a float32 array normalised to [-1, 1]."""
        raw = b"".join(frames)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        arr /= 32768.0          # normalise int16 → float32 in [-1.0, 1.0]
        return arr


class FileAudioSource:
    """
    Reads a WAV / FLAC / OGG file and yields it as a single utterance.
    Useful for batch testing and demos.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def utterances(self) -> Generator[np.ndarray, None, None]:
        import soundfile as sf
        logger.info(f"[audio] Reading from file: {self.filepath}")
        data, sr = sf.read(self.filepath, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]           # take first channel
        if sr != SAMPLE_RATE:
            # Simple linear resample (good enough for demo; use resampy in prod)
            import numpy as np
            ratio   = SAMPLE_RATE / sr
            new_len = int(len(data) * ratio)
            data    = np.interp(
                np.linspace(0, len(data), new_len),
                np.arange(len(data)),
                data,
            )
        yield data.astype(np.float32)
