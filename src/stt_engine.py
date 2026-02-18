"""
stt_engine.py
─────────────────────────────────────────────────────────────────
Faster-Whisper STT engine — Python 3.10 · Windows · 8 GB RAM

Optimisation choices for 8 GB RAM / CPU-only:
  • model_size = "base"  → 145 MB on disk, ~300 MB RAM (INT8)
  • compute_type = "int8" → quantised, 2-3× faster than float32
  • beam_size = 1         → greedy decode, eliminates beam search cost
  • vad_filter = True     → Silero VAD skips silent regions
  • warmup()              → pre-loads JIT cache at startup
"""
from __future__ import annotations

import os
# Fix: Windows OpenMP conflict between PyTorch and other libs (libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from loguru import logger


@dataclass
class Segment:
    """A single transcribed text segment with timing."""
    text: str
    start: float   # seconds from audio start
    end: float     # seconds from audio start
    no_speech_prob: float = 0.0

    def __str__(self):
        return f"[{self.start:.2f}s → {self.end:.2f}s]  {self.text}"


@dataclass
class TranscriptionResult:
    """Full result from transcribing one utterance."""
    text: str                           # full joined text
    segments: List[Segment]
    language: str
    latency_ms: float                   # wall-clock ms from call → return
    audio_duration_ms: float

    def __str__(self):
        return (
            f"Text      : {self.text}\n"
            f"Language  : {self.language}\n"
            f"Latency   : {self.latency_ms:.0f} ms\n"
            f"Audio len : {self.audio_duration_ms:.0f} ms"
        )


class STTEngine:
    """
    Wraps faster-whisper with lazy loading and INT8 quantisation.

    Parameters
    ----------
    model_size   : "tiny" | "base" | "small" (tiny=fastest, small=best quality)
    model_dir    : local cache directory for model files
    device       : "cpu" or "cuda"
    compute_type : "int8" (fastest CPU) | "float16" (GPU) | "float32"
    language     : ISO 639-1 code, e.g. "en"
    beam_size    : 1 = greedy (fastest), higher = more accurate
    vad_filter   : use Silero VAD inside faster-whisper to skip silence
    """

    def __init__(
        self,
        model_size:   str = "base",
        model_dir:    str = "models/whisper",
        device:       str = "cpu",
        compute_type: str = "int8",
        language:     str = "en",
        beam_size:    int = 1,
        vad_filter:   bool = True,
        vad_threshold: float = 0.5,
    ):
        self.model_size    = model_size
        self.model_dir     = model_dir
        self.device        = device
        self.compute_type  = compute_type
        self.language      = language
        self.beam_size     = beam_size
        self.vad_filter    = vad_filter
        self.vad_threshold = vad_threshold
        self._model        = None      # loaded lazily

    # ── Lazy loader ──────────────────────────────────────────────────────────
    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

        logger.info(
            f"[stt] Loading Whisper '{self.model_size}' "
            f"({self.device}, {self.compute_type}) …"
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device         = self.device,
            compute_type   = self.compute_type,
            download_root  = self.model_dir,
        )
        elapsed = (time.perf_counter() - t0) * 1_000
        logger.info(f"[stt] Model loaded in {elapsed:.0f} ms")

    def warmup(self):
        """
        Run one silent inference to warm up the model and JIT caches.
        Should be called once at startup to minimise first-utterance latency.
        """
        self._load_model()
        logger.info("[stt] Warming up model …")
        silence = np.zeros(16_000, dtype=np.float32)   # 1 s of silence
        self.transcribe(silence)
        logger.info("[stt] Warmup complete")

    # ── Core transcription ────────────────────────────────────────────────────
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a float32 numpy array (16kHz, mono, [-1,1]).

        Returns a TranscriptionResult with latency measurements.
        """
        self._load_model()

        audio_duration_ms = len(audio) / 16_000 * 1_000
        t0 = time.perf_counter()

        segments_iter, info = self._model.transcribe(
            audio,
            language      = self.language,
            beam_size     = self.beam_size,
            vad_filter    = self.vad_filter,
            vad_parameters= {"threshold": self.vad_threshold},
            condition_on_previous_text = False,   # no hallucination carry-over
            word_timestamps = False,               # segment-level is sufficient
        )

        # Force evaluation of the lazy generator
        segments: List[Segment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append(
                    Segment(
                        text           = text,
                        start          = seg.start,
                        end            = seg.end,
                        no_speech_prob = seg.no_speech_prob,
                    )
                )

        latency_ms = (time.perf_counter() - t0) * 1_000
        full_text  = " ".join(s.text for s in segments).strip()

        result = TranscriptionResult(
            text              = full_text,
            segments          = segments,
            language          = info.language,
            latency_ms        = latency_ms,
            audio_duration_ms = audio_duration_ms,
        )

        logger.debug(
            f"[stt] '{full_text}' | "
            f"latency={latency_ms:.0f}ms | "
            f"audio={audio_duration_ms:.0f}ms"
        )

        if latency_ms > 500:
            logger.warning(
                f"[stt] Latency {latency_ms:.0f}ms exceeds 500ms target. "
                "Consider switching to 'tiny' model or enabling INT8."
            )

        return result

    # ── Convenience: transcribe from file ────────────────────────────────────
    def transcribe_file(self, filepath: str) -> TranscriptionResult:
        """Transcribe directly from a WAV/FLAC/OGG file path."""
        import soundfile as sf
        data, sr = sf.read(filepath, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if sr != 16_000:
            import numpy as np
            ratio   = 16_000 / sr
            new_len = int(len(data) * ratio)
            data    = np.interp(
                np.linspace(0, len(data), new_len),
                np.arange(len(data)),
                data,
            )
        return self.transcribe(data)
