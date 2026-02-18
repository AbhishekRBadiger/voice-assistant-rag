"""
benchmark.py
─────────────────────────────────────────────────────────────────
Performance benchmarking for the voice assistant pipeline.

Measures:
  • STT latency across multiple audio durations
  • RAG retrieval latency
  • End-to-end pipeline latency
  • Model cold-start time

Run:
  python benchmark.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import statistics
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config       import get_config
from stt_engine   import STTEngine
from rag_pipeline import RAGPipeline
from intent_parser import IntentParser

console = Console()

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "open chrome browser",
    "what is the weather today",
    "create a file named notes.txt",
    "search the web for python tutorials",
    "what time is it",
    "open calculator",
    "set a reminder to call mom at 3pm",
    "calculate 15 times 24",
    "take a screenshot",
    "show system information",
    "open file explorer",
    "delete the file report.txt",
    "find files named readme",
    "open terminal",
    "search the web for machine learning",
]


def benchmark_stt(cfg: dict, n_runs: int = 5) -> dict:
    """Benchmark STT latency across different audio lengths."""
    console.print("\n[bold cyan]── STT Benchmark ──[/bold cyan]")

    stt = STTEngine(
        model_size   = cfg["stt"]["model_size"],
        model_dir    = cfg["stt"]["model_dir"],
        device       = cfg["stt"]["device"],
        compute_type = cfg["stt"]["compute_type"],
        language     = cfg["stt"]["language"],
        beam_size    = cfg["stt"]["beam_size"],
    )

    # Warmup
    console.print("[dim]Loading model and warming up…[/dim]")
    t_load = time.perf_counter()
    stt.warmup()
    cold_start_ms = (time.perf_counter() - t_load) * 1000
    console.print(f"Cold-start: {cold_start_ms:.0f} ms")

    results = {}
    for duration_s in [1, 2, 3, 5]:
        latencies = []
        audio = np.random.normal(0, 0.01, 16_000 * duration_s).astype(np.float32)
        for _ in range(n_runs):
            r = stt.transcribe(audio)
            latencies.append(r.latency_ms)
        results[duration_s] = {
            "mean_ms": statistics.mean(latencies),
            "min_ms":  min(latencies),
            "max_ms":  max(latencies),
            "p95_ms":  sorted(latencies)[int(0.95 * len(latencies))],
        }

    # Print table
    t = Table(title="STT Latency (ms)", show_header=True, header_style="bold")
    t.add_column("Audio Length", style="cyan")
    t.add_column("Mean",  justify="right")
    t.add_column("Min",   justify="right")
    t.add_column("Max",   justify="right")
    t.add_column("P95",   justify="right")
    t.add_column("Pass?", justify="center")
    for dur, r in results.items():
        ok = "✓" if r["mean_ms"] < 500 else "✗"
        t.add_row(
            f"{dur}s",
            f"{r['mean_ms']:.0f}",
            f"{r['min_ms']:.0f}",
            f"{r['max_ms']:.0f}",
            f"{r['p95_ms']:.0f}",
            f"[green]{ok}[/green]" if ok == "✓" else f"[red]{ok}[/red]",
        )
    console.print(t)

    return {"cold_start_ms": cold_start_ms, "per_duration": results}


def benchmark_rag(cfg: dict, n_runs: int = 20) -> dict:
    """Benchmark RAG retrieval latency."""
    console.print("\n[bold cyan]── RAG Benchmark ──[/bold cyan]")

    rag = RAGPipeline(
        actions_path         = cfg["paths"]["actions_dir"] + "/actions.json",
        embedding_model      = cfg["rag"]["embedding_model"],
        index_path           = cfg["rag"]["index_path"],
        metadata_path        = cfg["rag"]["metadata_path"],
        top_k                = cfg["rag"]["top_k"],
        confidence_threshold = cfg["rag"]["confidence_threshold"],
    )

    console.print("[dim]Loading embedder and building index…[/dim]")
    t_load = time.perf_counter()
    rag.setup()
    index_build_ms = (time.perf_counter() - t_load) * 1000
    console.print(f"Index build / load: {index_build_ms:.0f} ms")

    # Benchmark query latency
    latencies   = []
    top1_scores = []
    parser      = IntentParser()

    for i in range(n_runs):
        q = TEST_QUERIES[i % len(TEST_QUERIES)]
        t0 = time.perf_counter()
        matches = rag.retrieve(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        if matches:
            top1_scores.append(matches[0].confidence)

    t = Table(title="RAG Retrieval Latency (ms)", show_header=True, header_style="bold")
    t.add_column("Metric",  style="cyan")
    t.add_column("Value",   justify="right")
    t.add_row("Mean",  f"{statistics.mean(latencies):.1f} ms")
    t.add_row("Min",   f"{min(latencies):.1f} ms")
    t.add_row("Max",   f"{max(latencies):.1f} ms")
    t.add_row("P95",   f"{sorted(latencies)[int(0.95 * len(latencies))]:.1f} ms")
    t.add_row("Avg Top-1 Confidence", f"{statistics.mean(top1_scores):.2%}" if top1_scores else "N/A")
    console.print(t)

    return {
        "index_build_ms": index_build_ms,
        "mean_latency_ms": statistics.mean(latencies),
        "avg_confidence": statistics.mean(top1_scores) if top1_scores else 0,
    }


def benchmark_accuracy(cfg: dict) -> dict:
    """Check that each test query maps to the expected action."""
    console.print("\n[bold cyan]── Accuracy Benchmark ──[/bold cyan]")

    rag = RAGPipeline(
        actions_path         = cfg["paths"]["actions_dir"] + "/actions.json",
        embedding_model      = cfg["rag"]["embedding_model"],
        index_path           = cfg["rag"]["index_path"],
        metadata_path        = cfg["rag"]["metadata_path"],
        top_k                = 1,
        confidence_threshold = 0.0,   # show all for accuracy check
    )
    rag.setup()

    EXPECTED = {
        "open chrome browser":              "open_browser",
        "what is the weather today":        "get_weather",
        "create a file named notes.txt":    "create_file",
        "search the web for python":        "web_search",
        "what time is it":                  "get_time",
        "open calculator":                  "open_calculator",
        "calculate 15 times 24":            "calculate",
        "take a screenshot":                "take_screenshot",
        "show system information":          "system_info",
        "open file explorer":               "open_file_manager",
        "delete the file report.txt":       "delete_file",
        "find files named readme":          "search_files",
        "open terminal":                    "open_terminal",
    }

    correct = 0
    t = Table(title="Action Matching Accuracy", show_header=True, header_style="bold")
    t.add_column("Query",    style="cyan", max_width=40)
    t.add_column("Expected", style="dim")
    t.add_column("Got",      style="bold")
    t.add_column("Conf",     justify="right")
    t.add_column("✓/✗",      justify="center")

    for query, expected_id in EXPECTED.items():
        matches = rag.retrieve(query)
        got_id  = matches[0].action_id if matches else "no_match"
        conf    = f"{matches[0].confidence:.0%}" if matches else "—"
        ok      = "✓" if got_id == expected_id else "✗"
        if ok == "✓":
            correct += 1
        t.add_row(
            query[:40],
            expected_id,
            got_id,
            conf,
            f"[green]{ok}[/green]" if ok == "✓" else f"[red]{ok}[/red]",
        )

    accuracy = correct / len(EXPECTED) * 100
    console.print(t)
    console.print(f"\n[bold]Accuracy: {correct}/{len(EXPECTED)} ({accuracy:.0f}%)[/bold]")
    return {"accuracy_pct": accuracy, "correct": correct, "total": len(EXPECTED)}


def main():
    console.print("[bold]Voice Assistant Performance Benchmark[/bold]")
    console.print(f"{'─' * 50}")

    cfg = get_config()
    results = {}

    results["rag"]      = benchmark_rag(cfg)
    results["accuracy"] = benchmark_accuracy(cfg)
    results["stt"]      = benchmark_stt(cfg)

    # ── Summary ───────────────────────────────────────────────────────────
    console.print("\n[bold cyan]── Summary ──[/bold cyan]")
    s = Table(show_header=False, box=None)
    s.add_column("Metric",  style="dim", width=30)
    s.add_column("Value",   style="bold")
    s.add_row("STT Cold-Start",    f"{results['stt']['cold_start_ms']:.0f} ms")
    s.add_row("STT Mean (2s clip)",f"{results['stt']['per_duration'][2]['mean_ms']:.0f} ms")
    s.add_row("RAG Mean Latency",  f"{results['rag']['mean_latency_ms']:.1f} ms")
    s.add_row("RAG Index Build",   f"{results['rag']['index_build_ms']:.0f} ms")
    s.add_row("Accuracy",          f"{results['accuracy']['accuracy_pct']:.0f}%")
    console.print(s)

    target_met = results["stt"]["per_duration"][2]["mean_ms"] < 500
    console.print(
        f"\n{'[bold green]✓ Latency target (<500ms) MET[/bold green]' if target_met else '[bold red]✗ Latency target MISSED – consider tiny model or INT8[/bold red]'}"
    )


if __name__ == "__main__":
    main()
