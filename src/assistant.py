"""
assistant.py
─────────────────────────────────────────────────────────────────
Orchestrator: wires up all components into the complete pipeline.
Python 3.10 · Windows 10/11 · 8 GB RAM

Microphone → Audio Capture → VAD → STT → RAG → Intent → Executor → Feedback

Usage:

  python src/assistant.py                        # microphone mode
  python src/assistant.py --model tiny           # faster, slightly less accurate
  python src/assistant.py --mode file --file x.wav
"""
from __future__ import annotations
import os
# Fix: Windows OpenMP conflict between PyTorch and other libs (libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.columns import Columns

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config        import get_config
from audio_capture import AudioCapture, FileAudioSource
from stt_engine    import STTEngine
from rag_pipeline  import RAGPipeline
from intent_parser import IntentParser
from action_executor import ActionExecutor

console = Console()

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = """
 ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
 ██║   ██║██╔═══██╗██║██╔════╝██╔════╝
 ██║   ██║██║   ██║██║██║     █████╗  
 ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  
  ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
   ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
   Offline Speech Assistant  v1.0
"""


class VoiceAssistant:
    """
    End-to-end offline voice assistant.
    
    Parameters
    ----------
    config_path : path to config.yaml (default: project root)
    mode        : "mic" for real microphone | "file" for audio file input
    audio_file  : path to audio file (only used when mode="file")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        mode: str = "mic",
        audio_file: Optional[str] = None,
    ):
        self.cfg        = get_config()
        self.mode       = mode
        self.audio_file = audio_file
        self._running   = False
        self._stats     = {
            "total_utterances":   0,
            "successful_actions": 0,
            "failed_actions":     0,
            "avg_stt_latency_ms": 0.0,
            "avg_rag_latency_ms": 0.0,
        }

        # ── Configure logging ─────────────────────────────────────────────
        logger.remove()
        log_level = "DEBUG" if self.cfg["ui"]["verbose"] else "INFO"
        logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level} | {message}")
        logger.add(
            Path(self.cfg["paths"]["logs_dir"]) / "assistant.log",
            rotation="10 MB",
            level="DEBUG",
            format="{time} | {level} | {message}",
        )

    def _build_components(self):
        """Initialise all components (called once at startup)."""
        cfg = self.cfg

        # ── STT ───────────────────────────────────────────────────────────
        self.stt = STTEngine(
            model_size    = cfg["stt"]["model_size"],
            model_dir     = cfg["stt"]["model_dir"],
            device        = cfg["stt"]["device"],
            compute_type  = cfg["stt"]["compute_type"],
            language      = cfg["stt"]["language"],
            beam_size     = cfg["stt"]["beam_size"],
            vad_filter    = cfg["stt"]["vad_filter"],
            vad_threshold = cfg["stt"]["vad_threshold"],
        )

        # ── RAG ───────────────────────────────────────────────────────────
        self.rag = RAGPipeline(
            actions_path         = cfg["paths"]["actions_dir"] + "/actions.json",
            embedding_model      = cfg["rag"]["embedding_model"],
            index_path           = cfg["rag"]["index_path"],
            metadata_path        = cfg["rag"]["metadata_path"],
            top_k                = cfg["rag"]["top_k"],
            confidence_threshold = cfg["rag"]["confidence_threshold"],
        )

        # ── Intent + Executor ─────────────────────────────────────────────
        self.intent_parser = IntentParser()
        self.executor      = ActionExecutor(
            action_log_path      = cfg["executor"]["action_log_path"],
            require_confirmation = cfg["executor"]["require_confirmation_for_destructive"],
        )

        # ── Audio source ──────────────────────────────────────────────────
        if self.mode == "file" and self.audio_file:
            self.audio_source = FileAudioSource(self.audio_file)
        else:
            self.audio_source = AudioCapture(
                vad_aggressiveness = 2,
                silence_ms         = cfg["stt"]["silence_threshold_ms"],
                min_speech_ms      = cfg["stt"]["min_speech_ms"],
                max_speech_ms      = cfg["stt"]["max_speech_ms"],
                device             = cfg["audio"]["device"],
            )

    def startup(self):
        """Warm-up sequence: load models, build index, warmup inference."""
        console.print(BANNER, style="bold cyan")
        console.print("─" * 60, style="dim")
        console.print("[bold]Starting up… (first run may download models)[/bold]\n")

        self._build_components()

        # Warm up STT (loads model + runs one silent inference)
        console.print("[dim]• Loading STT model…[/dim]")
        self.stt.warmup()

        # Build / load RAG index
        console.print("[dim]• Building RAG index…[/dim]")
        self.rag.setup()

        console.print("\n[bold green]✓ System ready![/bold green]")
        self._print_status_table()
        console.print("\n[bold yellow]Listening for commands… (Ctrl+C to quit)[/bold yellow]\n")

    def run(self):
        """Main continuous listening loop."""
        self.startup()
        self._running = True

        try:
            for audio_chunk in self.audio_source.utterances():
                if not self._running:
                    break
                self._process_utterance(audio_chunk)
        except KeyboardInterrupt:
            console.print("\n\n[bold red]Shutting down…[/bold red]")
            self._print_final_stats()

    # ── Core pipeline ─────────────────────────────────────────────────────────
    def _process_utterance(self, audio_chunk):
        """
        Full pipeline: audio array → STT → RAG → Intent → Action → Feedback.
        """
        total_t0 = time.perf_counter()
        self._stats["total_utterances"] += 1

        # ── 1. STT ────────────────────────────────────────────────────────
        stt_result = self.stt.transcribe(audio_chunk)
        text       = stt_result.text.strip()

        if not text:
            logger.debug("[pipeline] Empty transcription, skipping")
            return

        # Update running average
        n = self._stats["total_utterances"]
        self._stats["avg_stt_latency_ms"] = (
            (self._stats["avg_stt_latency_ms"] * (n - 1) + stt_result.latency_ms) / n
        )

        # ── Display transcription ─────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        console.print(
            f"\n[dim]{ts}[/dim]  [bold white]🎤 You:[/bold white] [italic]{text}[/italic]"
        )

        # ── 2. RAG retrieval ──────────────────────────────────────────────
        rag_t0    = time.perf_counter()
        matches   = self.rag.retrieve(text)
        rag_ms    = (time.perf_counter() - rag_t0) * 1_000

        n = self._stats["total_utterances"]
        self._stats["avg_rag_latency_ms"] = (
            (self._stats["avg_rag_latency_ms"] * (n - 1) + rag_ms) / n
        )

        if not matches:
            console.print(
                "   [yellow]⚠ Command not recognised. Try rephrasing.[/yellow]"
            )
            self._stats["failed_actions"] += 1
            return

        best = matches[0]
        if self.cfg["ui"]["show_confidence"]:
            console.print(
                f"   [dim]→ Matched: [cyan]{best.name}[/cyan] "
                f"(confidence: {best.confidence:.0%})[/dim]"
            )

        # ── 3. Intent parsing ─────────────────────────────────────────────
        intent = self.intent_parser.parse(text, best)

        # ── 4. Execute ────────────────────────────────────────────────────
        result = self.executor.execute(intent)

        total_ms = (time.perf_counter() - total_t0) * 1_000

        if result.success:
            self._stats["successful_actions"] += 1
        else:
            self._stats["failed_actions"] += 1

        if self.cfg["ui"]["show_timestamps"]:
            console.print(
                f"   [dim]Pipeline: STT={stt_result.latency_ms:.0f}ms  "
                f"RAG={rag_ms:.0f}ms  "
                f"Total={total_ms:.0f}ms[/dim]"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _print_status_table(self):
        cfg = self.cfg
        t   = Table(show_header=False, box=None, pad_edge=False)
        t.add_column("Key",   style="dim", width=22)
        t.add_column("Value", style="bold")
        t.add_row("STT Model",     cfg["stt"]["model_size"])
        t.add_row("Compute Type",  cfg["stt"]["compute_type"])
        t.add_row("Device",        cfg["stt"]["device"])
        t.add_row("Embedding",     cfg["rag"]["embedding_model"].split("/")[-1])
        t.add_row("Language",      cfg["stt"]["language"].upper())
        t.add_row("Mode",          self.mode)
        console.print(t)

    def _print_final_stats(self):
        console.print("\n[bold]Session Stats:[/bold]")
        s = self._stats
        console.print(f"  Utterances     : {s['total_utterances']}")
        console.print(f"  Successful     : {s['successful_actions']}")
        console.print(f"  Failed         : {s['failed_actions']}")
        console.print(f"  Avg STT Latency: {s['avg_stt_latency_ms']:.0f} ms")
        console.print(f"  Avg RAG Latency: {s['avg_rag_latency_ms']:.1f} ms")

    def stop(self):
        self._running = False
        if hasattr(self.audio_source, "stop"):
            self.audio_source.stop()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline Voice Assistant")
    parser.add_argument("--mode",  choices=["mic", "file"], default="mic",
                        help="Input mode: microphone or audio file")
    parser.add_argument("--file",  default=None,
                        help="Path to audio file (when --mode file)")
    parser.add_argument("--model", default=None,
                        choices=["tiny", "base", "small"],
                        help="Override STT model size")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debug output")
    args = parser.parse_args()

    va = VoiceAssistant(mode=args.mode, audio_file=args.file)
    if args.model:
        va.cfg["stt"]["model_size"] = args.model
    if args.verbose:
        va.cfg["ui"]["verbose"] = True

    va.run()
