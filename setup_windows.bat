@echo off
REM ═══════════════════════════════════════════════════════════════
REM  Voice Assistant - Windows Setup Script
REM  Python 3.10 · Windows 10/11 · 8 GB RAM
REM
REM  Run this script ONCE from the project root:
REM    setup_windows.bat
REM ═══════════════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       Voice Assistant - Windows Setup                    ║
echo ║       Python 3.10  ·  Windows 10/11  ·  8 GB RAM        ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM ── Check Python version ─────────────────────────────────────
python --version 2>NUL
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 from python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version') do set PYVER=%%v
echo Detected Python: %PYVER%
echo.

REM ── Create virtual environment ───────────────────────────────
echo [1/6] Creating virtual environment...
if exist venv (
    echo        venv already exists, skipping.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo        Created: venv\
)

REM ── Activate venv ────────────────────────────────────────────
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate venv
    pause
    exit /b 1
)
echo        Active: %VIRTUAL_ENV%

REM ── Upgrade pip ──────────────────────────────────────────────
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo        pip upgraded.

REM ── Install PyTorch CPU-only first (avoids downloading CUDA) ─
echo [4/6] Installing PyTorch CPU-only build...
echo        (This is ~200 MB - may take a few minutes)
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu ^
    --extra-index-url https://download.pytorch.org/whl/cpu ^
    --quiet
if errorlevel 1 (
    echo [WARNING] PyTorch install had issues. Trying standard install...
    pip install torch torchvision --quiet
)
echo        PyTorch installed.

REM ── Install remaining requirements ───────────────────────────
echo [5/6] Installing all dependencies...
pip install -r requirements.txt --quiet --no-deps-only 2>NUL
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Some packages failed to install.
    echo         Check errors above and re-run.
    pause
    exit /b 1
)
echo        All dependencies installed.

REM ── Download models ──────────────────────────────────────────
echo [6/6] Downloading AI models (requires internet, ~500 MB)...
echo        Whisper base model    : ~145 MB
echo        MiniLM-L6-v2          : ~22 MB
echo        spaCy en_core_web_sm  : ~12 MB
echo.
python download_models.py
if errorlevel 1 (
    echo [ERROR] Model download failed. Check internet connection.
    pause
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  ✓  Setup complete!                                      ║
echo ║                                                          ║
echo ║  To run the assistant:                                   ║
echo ║    venv\Scripts\activate                                 ║
echo ║    python src\assistant.py                               ║
echo ║                                                          ║
echo ║  To run benchmarks:                                      ║
echo ║    python benchmark.py                                   ║
echo ║                                                          ║
echo ║  To build Windows .exe:                                  ║
echo ║    python build_exe.py                                   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
pause
