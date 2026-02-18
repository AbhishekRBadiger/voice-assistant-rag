"""
download_models.py
─────────────────────────────────────────────────────────────────
Downloads all required offline models.
Python 3.10 · Windows 10/11 · 8 GB RAM

Downloads:
  1. Faster-Whisper "base" model  (~145 MB) → models/whisper/
  2. all-MiniLM-L6-v2             (~22 MB)  → models/embeddings/
  3. Builds FAISS index from action docs    → models/faiss/
  4. spaCy en_core_web_sm         (~12 MB)  → site-packages

Run once before first use (requires internet):
  python download_models.py
  python download_models.py --model tiny    # smaller/faster model
  python download_models.py --force         # re-download everything

Windows note:
  Run from the activated virtual environment:
    venv\\Scripts\\activate
    python download_models.py
"""
import os
# Fix: Windows OpenMP conflict between PyTorch and other libs (libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import time
from pathlib import Path


# Search upward from this file until we find config.yaml
def _find_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "config.yaml").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent  # fallback

ROOT = _find_root()
sys.path.insert(0, str(ROOT / "src"))


def step(msg: str):
    print(f"\n{'─' * 50}")
    print(f"  {msg}")
    print(f"{'─' * 50}")


def download_whisper(model_size: str, model_dir: Path):
    step(f"Downloading Faster-Whisper '{model_size}' model…")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  ✗ faster-whisper not installed.")
        print("    Run: pip install faster-whisper")
        return False

    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8",
                             download_root=str(model_dir))
        elapsed = time.perf_counter() - t0
        print(f"  ✓ Whisper '{model_size}' downloaded in {elapsed:.1f}s")
        # Warmup to verify
        import numpy as np
        silence = np.zeros(16_000, dtype=np.float32)
        list(model.transcribe(silence)[0])   # exhaust generator
        print("  ✓ Model warmup successful")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_embeddings(model_name: str, cache_dir: Path):
    step(f"Downloading sentence-transformer '{model_name}'…")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ✗ sentence-transformers not installed.")
        print("    Run: pip install sentence-transformers")
        return False

    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        elapsed = time.perf_counter() - t0
        # Quick test
        vecs = model.encode(["hello world"], normalize_embeddings=True)
        dim  = vecs.shape[1]
        print(f"  ✓ Embedding model downloaded in {elapsed:.1f}s (dim={dim})")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def build_faiss_index(force: bool = False):
    step("Building FAISS index from action documents…")
    try:
        from config       import get_config
        from rag_pipeline import RAGPipeline

        cfg = get_config()
        rag = RAGPipeline(
            actions_path    = cfg["paths"]["actions_dir"] + "/actions.json",
            embedding_model = cfg["rag"]["embedding_model"],
            index_path      = cfg["rag"]["index_path"],
            metadata_path   = cfg["rag"]["metadata_path"],
        )
        rag.setup(force_rebuild=force)
        print("  ✓ FAISS index ready")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_spacy(model: str = "en_core_web_sm"):
    step(f"Downloading spaCy model '{model}'…")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", model],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✓ spaCy '{model}' ready")
        return True
    else:
        print(f"  ! spaCy download skipped (optional): {result.stderr.strip()[:100]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download all required models")
    parser.add_argument("--model",  default="base",
                        choices=["tiny", "base", "small"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--force",  action="store_true",
                        help="Re-download and rebuild everything")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Voice Assistant – Model Setup               ║")
    print("╚══════════════════════════════════════════════╝")

    import yaml
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Patch model size if overridden
    if args.model != "base":
        cfg["stt"]["model_size"] = args.model

    model_dir   = ROOT / cfg["stt"]["model_dir"]
    emb_dir     = ROOT / "models" / "embeddings"
    model_size  = cfg["stt"]["model_size"]
    emb_model   = cfg["rag"]["embedding_model"]

    results = {
        "whisper":    download_whisper(model_size, model_dir),
        "embeddings": download_embeddings(emb_model, emb_dir),
        "faiss":      build_faiss_index(force=args.force),
        "spacy":      download_spacy(),
    }

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Setup Complete                              ║")
    print("╠══════════════════════════════════════════════╣")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"║  {status} {name:<40} ║")
    print("╚══════════════════════════════════════════════╝")

    all_ok = all(results[k] for k in ["whisper", "embeddings", "faiss"])
    if all_ok:
        print("\n✓ All critical models ready. You can now run:")
        print("    python src/assistant.py")
        print("    python src/assistant.py --mode file --file audio.wav")
        print("    python benchmark.py")
    else:
        print("\n✗ Some downloads failed. Check errors above and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
