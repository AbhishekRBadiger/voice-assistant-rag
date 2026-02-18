# 🎙️ Offline Voice Assistant with RAG-based Action Execution

> An offline speech-to-text system with intelligent command recognition and automated action execution, built for low-latency performance.

---

## 📋 Table of Contents
- [Demo Video](#-demo-video)
- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [Architecture Overview](#️-architecture-overview)
- [STT Model Choice](#-stt-model-choice)
- [RAG Implementation](#-rag-implementation)
- [Latency Optimizations](#-latency-optimizations)
- [Supported Commands](#-supported-commands)
- [Configuration](#️-configuration)
- [Building the Executable](#-building-the-windows-executable)
- [Running Benchmarks](#-running-benchmarks)
- [Running Tests](#-running-tests)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---
## 🎥 Demo Video

**Watch the complete demo:** https://drive.google.com/file/d/1Hr80yzj0ozNu5HeRSg-KngD9nz46koQT/view?usp=sharing

**Video Contents (3.16 minutes):**
- ✅ System startup and architecture explanation
- ✅ Live voice commands with latency measurements
- ✅ 15 action types demonstrated (calculator, file ops, web search, etc.)
- ✅ Error handling (unrecognized commands + confirmation prompts)
- ✅ Performance metrics (<500ms latency achieved)
- ✅ Action logging and file structure walkthrough


## ⚡ Quick Start (Windows)

```bat
REM 1. Clone the project
git clone https://github.com/AbhishekRBadiger/voice-assistant-rag
cd voice_assistant

REM 2. One-command setup (creates venv, installs all deps, downloads models)
setup_windows.bat

REM 3. Activate venv and run
venv\Scripts\activate
python src\assistant.py
```

**Or step-by-step manually:**

```bat
REM Create + activate virtual environment
python -m venv venv
venv\Scripts\activate

REM Install PyTorch CPU-only first (saves ~2 GB vs CUDA build)
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

REM Install all other dependencies
pip install -r requirements.txt

REM Download models (one-time, ~180 MB total, needs internet)
python download_models.py

REM Run the assistant
python src\assistant.py

REM Other useful commands:
python src\assistant.py --model tiny       REM fastest option
python benchmark.py                        REM measure latency
python -m pytest tests\ -v                 REM run tests
python build_exe.py                        REM build Windows .exe
```

---
---

## 🔌 Working Offline

After initial setup, the system works **100% offline**. To ensure offline operation:
```bat
REM Set offline environment variables
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

REM Run assistant
python src\assistant.py
```

**Note:** Models must be downloaded once with internet before offline use works.

---

## 💻 System Requirements

| Requirement | Your Setup | Notes |
|-------------|-----------|-------|
| Python      | **3.10 or 3.12** ✓ | Use python.org installer, check "Add to PATH" |
| OS          | **Windows 10/11** ✓ | 64-bit required |
| RAM         | **8 GB** ✓ | Base model uses ~1.2 GB at runtime |
| Disk        | 2 GB free | Models + venv |
| CPU         | Any x64 | 4+ cores recommended for <200ms latency |
| Microphone  | Required | Set as default recording device in Windows Sound settings |
| Internet    | First run only | For downloading models (~180 MB) |

> **RAM breakdown at runtime:** Whisper base (INT8) ~300 MB + MiniLM embeddings ~150 MB + Python + OS = ~1.2 GB total. Leaves 6.8 GB free on your 8 GB machine.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     VOICE ASSISTANT PIPELINE                      │
│                                                                    │
│  🎤 Microphone                                                     │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────┐                                              │
│  │  Audio Capture   │  sounddevice, 16kHz mono int16              │
│  │  (audio_capture) │  blocksize=512 for low latency              │
│  └────────┬────────┘                                              │
│           │  raw PCM frames (30ms chunks)                          │
│           ▼                                                        │
│  ┌─────────────────┐                                              │
│  │  Voice Activity  │  WebRTC VAD (aggressiveness=2)              │
│  │  Detection (VAD) │  Detects speech start/end                   │
│  └────────┬────────┘                                              │
│           │  complete utterance (float32 numpy array)              │
│           ▼                                                        │
│  ┌─────────────────┐                                              │
│  │   STT Engine     │  Faster-Whisper (base, INT8 quantised)      │
│  │  (stt_engine)    │  Target: <500ms latency                     │
│  └────────┬────────┘                                              │
│           │  transcribed text + timestamps                         │
│           ▼                                                        │
│  ┌─────────────────┐                                              │
│  │  RAG Pipeline    │  sentence-transformers + FAISS              │
│  │  (rag_pipeline)  │  Top-K cosine similarity search             │
│  └────────┬────────┘                                              │
│           │  ActionMatch with confidence score                     │
│           ▼                                                        │
│  ┌─────────────────┐                                              │
│  │  Intent Parser   │  Rule-based parameter extraction            │
│  │ (intent_parser)  │  filename, URL, query, time, location       │
│  └────────┬────────┘                                              │
│           │  IntentResult with typed parameters                    │
│           ▼                                                        │
│  ┌─────────────────┐                                              │
│  │ Action Executor  │  15 action types, JSON logging              │
│  │(action_executor) │  Safe execution + error handling            │
│  └────────┬────────┘                                              │
│           │                                                        │
│           ▼                                                        │
│  👤 User Feedback (Rich terminal output)                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧠 STT Model Choice

**Chosen: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) with `base` model**

| Model | Size  | VRAM | WER  | Latency (2s clip) |
|-------|-------|------|------|-------------------|
| tiny  | 75MB  | 1GB  | ~15% | ~80ms             |
| base  | 145MB | 1GB  | ~10% | ~150ms ✓          |
| small | 466MB | 2GB  | ~7%  | ~350ms            |

**Why Faster-Whisper over stock Whisper?**
- Uses CTranslate2 engine → **4× faster inference**
- INT8 quantisation → **50% less memory**
- Identical accuracy to original Whisper
- Built-in Silero VAD filter

**Why `base` over `tiny`?**
- Base achieves significantly better WER on natural speech
- Still comfortably within 500ms latency target
- Better handling of accents and background noise

**Optimisations applied:**
- `compute_type="int8"` – quantised weights, fastest CPU inference
- `beam_size=1` – greedy decoding, skips beam search overhead
- `vad_filter=True` – internal silence skipping
- `condition_on_previous_text=False` – prevents hallucination carry-over
- Model warmup on startup – JIT cache pre-loaded

---

## 🔍 RAG Implementation

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional dense vectors
- 22MB model size
- ~14ms per sentence on CPU
- Fully offline after first download

**Vector store:** FAISS `IndexFlatIP`
- Inner product on unit-normalised vectors = cosine similarity
- O(n) brute-force search (n ≈ 150 example phrases; no approximation needed)
- Sub-millisecond retrieval latency
- Index persisted to disk, loaded instantly on subsequent runs

**Knowledge base:** 15 action documents in `data/actions/actions.json`
- Each action has 7–9 example phrases
- Covers: app launching, file ops, system commands, web search, calculations

**Retrieval flow:**
1. Embed query with MiniLM-L6-v2 (normalised)
2. FAISS top-K search (K=9 before deduplication)
3. Deduplicate: keep best score per `action_id`
4. Filter by `confidence_threshold=0.55`
5. Return top-3 ranked by cosine similarity

---

## ⚡ Latency Optimizations

### STT Optimizations
| Technique | Impact |
|-----------|--------|
| INT8 quantisation (`compute_type="int8"`) | 2–3× faster vs float32 |
| Greedy decoding (`beam_size=1`)           | Eliminates beam search |
| VAD filter (skip silence)                 | Reduces input duration |
| Model warmup at startup                   | Removes JIT overhead   |
| Lazy loading (load once, reuse)           | No reload per utterance |

### Audio Capture Optimizations
| Technique | Impact |
|-----------|--------|
| 30ms VAD frames | Minimal detection lag |
| `blocksize=512` | Low audio callback latency |
| Pre-speech padding ring buffer | Captures word beginnings |

### RAG Optimizations
| Technique | Impact |
|-----------|--------|
| Pre-built FAISS index (disk cache) | Skip re-embedding at startup |
| Batch embedding at build time | Amortised embedding cost |
| Unit-normalised vectors | Cosine = dot product (faster) |
| Small embedding model (MiniLM) | 5× faster than large models |

### Typical End-to-End Latency
```
Audio capture (VAD)    :   0ms  (overlapped with speaking)
STT inference (2s clip): ~150ms
RAG retrieval          :  ~5ms
Intent parsing         :  ~1ms
Action execution       :  ~5ms
─────────────────────────────
Total                  : ~160ms ✓ (well under 500ms target)
```

---

## 🎯 Supported Commands

| Category | Command Examples | Action |
|----------|-----------------|--------|
| **Apps** | "open chrome", "launch browser" | Opens web browser |
| **Apps** | "open notepad", "text editor" | Opens text editor |
| **Apps** | "open calculator", "calc" | Opens calculator |
| **Apps** | "open file explorer", "browse files" | Opens file manager |
| **Apps** | "open terminal", "open cmd" | Opens terminal |
| **Files** | "create a file named notes.txt" | Creates new file |
| **Files** | "delete the file report.txt" | Deletes file (with confirmation) |
| **Files** | "find files named readme" | Searches filesystem |
| **System** | "what time is it", "today's date" | Shows time/date |
| **System** | "take a screenshot" | Captures screen |
| **System** | "system information", "cpu usage" | Shows CPU/RAM/disk stats |
| **System** | "weather in London" | Opens weather in browser |
| **System** | "set reminder to call mom at 3pm" | Sets timed reminder |
| **Math** | "calculate 15 times 24" | Evaluates expression |
| **Search** | "search the web for Python tutorials" | Google search in browser |

---

## ⚙️ Configuration

Edit `config.yaml` to tune the system:

```yaml
stt:
  model_size: "base"       # tiny | base | small
  compute_type: "int8"     # int8 (CPU) | float16 (GPU)
  beam_size: 1             # 1=fastest, higher=more accurate
  silence_threshold_ms: 700 # ms of silence = utterance end

rag:
  confidence_threshold: 0.55  # lower = more permissive matching
  top_k: 3                    # candidates returned

intent:
  confidence_high: 0.80       # auto-execute above this
  confidence_medium: 0.60     # confirm before executing
```

---

## 📦 Building the Windows Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Download all models first (they'll be bundled)
python download_models.py

# Build the .exe
python build_exe.py

# Output will be in: dist/VoiceAssistant/
# Copy the entire folder to any Windows machine - no Python needed!
```

The resulting `dist/VoiceAssistant/` folder contains:
- `VoiceAssistant.exe` – main executable
- All bundled models (Whisper + embeddings)
- All dependencies (no Python installation required)

---
**Download Pre-built Executable:**

If you don't want to build it yourself, download the ready-to-use .exe from [GitHub Releases](https://github.com/AbhishekRBadiger/voice-assistant-rag/releases/tag/v1.0).

---

## 📊 Running Benchmarks

```bash
python benchmark.py
```

Outputs:
- STT latency across 1s, 2s, 3s, 5s audio clips
- RAG retrieval latency (mean, min, max, P95)
- Action matching accuracy (% of test queries matched correctly)
- Whether latency target (<500ms) is met

---

## 🧪 Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_pipeline.py::TestRAGPipeline -v
python -m pytest tests/test_pipeline.py::TestActionExecutor -v
python -m pytest tests/test_pipeline.py::TestIntegration -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📁 Project Structure

```
voice_assistant/
├── src/
│   ├── assistant.py          # Main orchestrator
│   ├── audio_capture.py      # Microphone + WebRTC VAD
│   ├── stt_engine.py         # Faster-Whisper STT wrapper
│   ├── rag_pipeline.py       # FAISS + sentence-transformers RAG
│   ├── intent_parser.py      # Parameter extraction
│   ├── action_executor.py    # 15 action implementations
│   └── config.py             # Config loader
├── data/
│   └── actions/
│       └── actions.json      # 15 action documents
├── models/
│   ├── whisper/              # Faster-Whisper model cache
│   ├── faiss/                # Persisted FAISS index
│   └── embeddings/           # Sentence-transformer cache
├── tests/
│   └── test_pipeline.py      # Unit + integration tests
├── logs/
│   ├── assistant.log         # Application logs
│   └── actions.jsonl         # Action execution log (JSON lines)
├── config.yaml               # Master configuration
├── requirements.txt          # Python dependencies
├── download_models.py        # One-time model setup
├── benchmark.py              # Performance benchmarks
└── build_exe.py              # PyInstaller build script
```

---

## 🔧 Troubleshooting (Windows)

**"No module named 'faster_whisper'"**
```bat
venv\Scripts\activate
pip install faster-whisper==1.0.3
```

**Microphone not detected / silent recordings**
- Open Windows Settings → System → Sound → Input
- Make sure your microphone is set as the default input device
- Test it with the Windows Voice Recorder app first
- If using a USB headset, plug it in before running the script
- To list available devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Set the device index in `config.yaml` → `audio.device: 1` (use the index from the list)

**"webrtcvad not found" or install error**
```bat
REM Use the pre-built wheels variant (no Visual C++ needed)
pip install webrtcvad-wheels==2.0.14
```

**"Microsoft Visual C++ 14.0 is required" during pip install**
- Install the free [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Or use pre-built alternatives already specified in requirements.txt

**PyTorch taking forever to install / wrong version**
```bat
REM Install CPU-only build explicitly (saves 2 GB download)
pip install torch==2.3.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

**Latency over 500ms on your machine**
```bat
REM Switch to tiny model (80ms vs 150ms for base)
python src\assistant.py --model tiny
```
Or edit `config.yaml` → `stt.model_size: "tiny"`

**FAISS import error**
```bat
pip install faiss-cpu==1.8.0
```

**calc.exe / notepad.exe not opening**
- These are Windows system apps and should always be available
- If they're blocked by your organization's policy, the error will be caught gracefully

**PowerShell execution policy blocks .bat / .ps1**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"Access is denied" when taking screenshots**
- Run VSCode or cmd as Administrator, or check Windows privacy settings
- Settings → Privacy → Screen capture → allow the app

---

## 📄 License

MIT License – see LICENSE file for details.
