"""
action_executor.py
─────────────────────────────────────────────────────────────────
Executes actions — Python 3.10 · Windows 10/11 · 8 GB RAM

Windows-specific notes:
  • App launching uses Windows executable names (notepad.exe, calc.exe, etc.)
  • File manager opens Explorer via subprocess
  • Terminal opens cmd.exe (or powershell.exe)
  • Screenshots use pyautogui (works without display server on Windows)
  • Weather opens wttr.in in default browser (offline-compatible display)
"""
from __future__ import annotations
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Dict

import psutil
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from intent_parser import IntentResult

console = Console()

# ── Result object ─────────────────────────────────────────────────────────────


class ExecutionResult:
    def __init__(
        self,
        success: bool,
        message: str,
        action_id: str = "",
        details: Optional[dict] = None,
    ):
        self.success   = success
        self.message   = message
        self.action_id = action_id
        self.details   = details or {}
        self.timestamp = datetime.now().isoformat()

    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.message}"


# ── Logger ────────────────────────────────────────────────────────────────────


class ActionLogger:
    def __init__(self, log_path: str = "logs/actions.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, intent: IntentResult, result: ExecutionResult):
        entry = {
            "timestamp":  result.timestamp,
            "action_id":  result.action_id,
            "executor":   intent.executor,
            "params":     intent.params,
            "confidence": round(intent.confidence, 4),
            "raw_text":   intent.raw_text,
            "success":    result.success,
            "message":    result.message,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Main Executor ─────────────────────────────────────────────────────────────


class ActionExecutor:
    """
    Routes an IntentResult to the correct execution function.
    """

    def __init__(
        self,
        action_log_path: str = "logs/actions.jsonl",
        require_confirmation: bool = True,
        confirm_callback: Optional[Callable[[str], bool]] = None,
    ):
        self._logger  = ActionLogger(action_log_path)
        self.require_confirmation = require_confirmation
        self._confirm = confirm_callback or self._default_confirm

        # Map executor name → method
        self._registry: Dict[str, Callable] = {
            "open_browser":      self._open_browser,
            "open_notepad":      self._open_notepad,
            "open_calculator":   self._open_calculator,
            "open_file_manager": self._open_file_manager,
            "open_terminal":     self._open_terminal,
            "create_file":       self._create_file,
            "delete_file":       self._delete_file,
            "search_files":      self._search_files,
            "get_weather":       self._get_weather,
            "set_reminder":      self._set_reminder,
            "calculate":         self._calculate,
            "web_search":        self._web_search,
            "get_time":          self._get_time,
            "take_screenshot":   self._take_screenshot,
            "system_info":       self._system_info,
        }

    # ── Dispatcher ────────────────────────────────────────────────────────────
    def execute(self, intent: IntentResult) -> ExecutionResult:
        t0 = time.perf_counter()

        if not intent.is_clear:
            result = ExecutionResult(
                success   = False,
                message   = "Command unclear – could you rephrase?",
                action_id = intent.action_id,
            )
            self._show(result, style="yellow")
            return result

        if intent.needs_confirmation and self.require_confirmation:
            msg = (
                f"This is a destructive action: {intent.action_id.replace('_', ' ')} "
                f"with {intent.params}. Confirm? (yes/no)"
            )
            if not self._confirm(msg):
                result = ExecutionResult(
                    success   = False,
                    message   = "Action cancelled by user.",
                    action_id = intent.action_id,
                )
                self._show(result, style="yellow")
                return result

        handler = self._registry.get(intent.executor)
        if handler is None:
            result = ExecutionResult(
                success   = False,
                message   = f"Unknown executor: {intent.executor}",
                action_id = intent.action_id,
            )
            self._show(result, style="red")
            self._logger.log(intent, result)
            return result

        try:
            result = handler(intent.params)
            result.action_id = intent.action_id
        except Exception as exc:
            logger.exception(f"[executor] Error in {intent.executor}: {exc}")
            result = ExecutionResult(
                success   = False,
                message   = f"Error: {exc}",
                action_id = intent.action_id,
            )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"[executor] {intent.executor} finished in {elapsed:.0f}ms")

        self._show(result, style="green" if result.success else "red")
        self._logger.log(intent, result)
        return result

    # ── Action Implementations ────────────────────────────────────────────────

    def _open_browser(self, params: dict) -> ExecutionResult:
        url = params.get("url", "https://www.google.com")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        webbrowser.open(url)
        return ExecutionResult(True, f"Opened browser at {url}")

    def _open_notepad(self, params: dict) -> ExecutionResult:
        # Windows: notepad.exe is always available in System32
        # Linux/Mac fallbacks kept for portability
        if _is_windows():
            subprocess.Popen(["notepad.exe"])
        elif _is_mac():
            subprocess.Popen(["open", "-a", "TextEdit"])
        else:
            self._launch(["gedit", "mousepad", "nano"])
        return ExecutionResult(True, "Opened text editor")

    def _open_calculator(self, params: dict) -> ExecutionResult:
        if _is_windows():
            # calc.exe lives in System32 on all Windows versions
            subprocess.Popen(["calc.exe"])
        elif _is_mac():
            subprocess.Popen(["open", "-a", "Calculator"])
        else:
            self._launch(["gnome-calculator", "kcalc", "xcalc"])
        return ExecutionResult(True, "Opened calculator")

    def _open_file_manager(self, params: dict) -> ExecutionResult:
        path = params.get("path") or str(Path.home())
        if _is_windows():
            subprocess.Popen(["explorer.exe", path])
        elif _is_mac():
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return ExecutionResult(True, f"Opened file manager at {path}")

    def _open_terminal(self, params: dict) -> ExecutionResult:
        if _is_windows():
            subprocess.Popen(["cmd.exe"])
        elif _is_mac():
            subprocess.Popen(["open", "-a", "Terminal"])
        else:
            for term in ["gnome-terminal", "xterm", "konsole"]:
                if self._which(term):
                    subprocess.Popen([term])
                    break
        return ExecutionResult(True, "Opened terminal")

    def _create_file(self, params: dict) -> ExecutionResult:
        filename  = params.get("filename", "new_file.txt")
        directory = params.get("directory", str(Path.home() / "Desktop"))
        directory = os.path.expanduser(directory)
        os.makedirs(directory, exist_ok=True)
        filepath = Path(directory) / filename
        filepath.touch(exist_ok=True)
        return ExecutionResult(True, f"Created file: {filepath}", details={"path": str(filepath)})

    def _delete_file(self, params: dict) -> ExecutionResult:
        filename = params.get("filename")
        if not filename:
            return ExecutionResult(False, "No filename specified for deletion")
        path = Path(os.path.expanduser(filename))
        if not path.exists():
            # try Desktop
            desktop_path = Path.home() / "Desktop" / filename
            if desktop_path.exists():
                path = desktop_path
            else:
                return ExecutionResult(False, f"File not found: {filename}")
        path.unlink()
        return ExecutionResult(True, f"Deleted file: {path}")

    def _search_files(self, params: dict) -> ExecutionResult:
        query     = params.get("query", "")
        directory = os.path.expanduser(params.get("directory", "~"))
        if not query:
            return ExecutionResult(False, "No search query provided")

        found = []
        try:
            for root, dirs, files in os.walk(directory):
                # Skip hidden / system directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in files:
                    if query.lower() in fname.lower():
                        found.append(os.path.join(root, fname))
                if len(found) >= 20:
                    break
        except PermissionError:
            pass

        if found:
            msg = f"Found {len(found)} file(s) matching '{query}':\n" + "\n".join(found[:10])
        else:
            msg = f"No files found matching '{query}' in {directory}"
        return ExecutionResult(bool(found), msg, details={"files": found})

    def _get_weather(self, params: dict) -> ExecutionResult:
        location = params.get("location", "current")
        # Offline: use wttr.in via browser (doesn't require API key)
        if location and location != "current":
            url = f"https://wttr.in/{location.replace(' ', '+')}"
        else:
            url = "https://wttr.in"
        webbrowser.open(url)
        return ExecutionResult(True, f"Opening weather for {location} in browser")

    def _set_reminder(self, params: dict) -> ExecutionResult:
        message     = params.get("message", "Reminder")
        time_str    = params.get("time")
        delay_secs  = 60  # default 1 minute

        if time_str:
            # Try to parse "in N minutes/hours"
            m = re.search(r"in\s+(\d+)\s+(minute|hour)", time_str)
            if m:
                n    = int(m.group(1))
                unit = m.group(2)
                delay_secs = n * (60 if unit == "minute" else 3600)
            # Try HH:MM or H:MM am/pm (very basic)
            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str)
            if m:
                hour = int(m.group(1))
                mins = int(m.group(2) or 0)
                ampm = (m.group(3) or "").lower()
                if ampm == "pm" and hour < 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0
                now   = datetime.now()
                then  = now.replace(hour=hour, minute=mins, second=0, microsecond=0)
                delta = (then - now).total_seconds()
                if delta < 0:
                    delta += 86400   # tomorrow
                delay_secs = int(delta)

        def _fire():
            time.sleep(delay_secs)
            console.print(
                Panel(f"⏰ REMINDER: {message}", style="bold yellow"),
                justify="center",
            )

        t = threading.Thread(target=_fire, daemon=True)
        t.start()
        return ExecutionResult(
            True,
            f"Reminder set: '{message}' in {delay_secs}s",
            details={"delay_seconds": delay_secs},
        )

    def _calculate(self, params: dict) -> ExecutionResult:
        expression = params.get("expression", "")
        if not expression:
            return ExecutionResult(False, "No expression to calculate")
        # Safe eval: only allow math characters
        clean = re.sub(r"[^0-9\+\-\*\/\(\)\.\s\%\^]", "", expression)
        clean = clean.replace("^", "**")
        try:
            # Provide safe builtins
            safe_globals = {
                "__builtins__": {},
                "sqrt": math.sqrt,
                "sin":  math.sin,
                "cos":  math.cos,
                "tan":  math.tan,
                "pi":   math.pi,
                "e":    math.e,
                "abs":  abs,
                "round": round,
            }
            result = eval(clean, safe_globals)
            msg = f"{expression} = {result}"
            return ExecutionResult(True, msg, details={"result": result})
        except Exception as exc:
            return ExecutionResult(False, f"Could not evaluate '{expression}': {exc}")

    def _web_search(self, params: dict) -> ExecutionResult:
        query = params.get("query", "")
        if not query:
            return ExecutionResult(False, "No search query provided")
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return ExecutionResult(True, f"Searching the web for: {query}")

    def _get_time(self, params: dict) -> ExecutionResult:
        now = datetime.now()
        msg = (
            f"Current time: {now.strftime('%I:%M %p')}\n"
            f"Today's date: {now.strftime('%A, %B %d, %Y')}"
        )
        return ExecutionResult(True, msg)

    def _take_screenshot(self, params: dict) -> ExecutionResult:
        try:
            import pyautogui
            filename = params.get("filename") or f"screenshot_{int(time.time())}.png"
            filepath = Path.home() / "Desktop" / filename
            screenshot = pyautogui.screenshot()
            screenshot.save(str(filepath))
            return ExecutionResult(True, f"Screenshot saved: {filepath}", details={"path": str(filepath)})
        except ImportError:
            return ExecutionResult(False, "pyautogui not installed. Run: pip install pyautogui")

    def _system_info(self, params: dict) -> ExecutionResult:
        metric = params.get("metric", "all")
        info   = {}
        lines  = []

        if metric in ("cpu", "all"):
            cpu = psutil.cpu_percent(interval=0.5)
            freq = psutil.cpu_freq()
            info["cpu_percent"] = cpu
            lines.append(
                f"CPU  : {cpu}% | {psutil.cpu_count()} cores"
                + (f" @ {freq.current:.0f} MHz" if freq else "")
            )

        if metric in ("ram", "memory", "all"):
            vm = psutil.virtual_memory()
            info["ram_total_gb"] = round(vm.total / 1e9, 1)
            info["ram_used_percent"] = vm.percent
            lines.append(
                f"RAM  : {vm.percent}% used of {vm.total/1e9:.1f} GB"
            )

        if metric in ("disk", "storage", "all"):
            disk = psutil.disk_usage("/")
            info["disk_total_gb"] = round(disk.total / 1e9, 1)
            info["disk_used_percent"] = disk.percent
            lines.append(
                f"Disk : {disk.percent}% used of {disk.total/1e9:.1f} GB"
            )

        msg = "\n".join(lines) if lines else "No metrics collected"
        return ExecutionResult(True, msg, details=info)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _launch(candidates: List[str]):
        """Try each executable in candidates until one succeeds."""
        for cmd in candidates:
            try:
                subprocess.Popen([cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except (FileNotFoundError, OSError):
                continue
        logger.warning(f"[executor] Could not launch any of: {candidates}")

    @staticmethod
    def _which(cmd: str) -> bool:
        import shutil
        return shutil.which(cmd) is not None

    @staticmethod
    def _default_confirm(prompt: str) -> bool:
        console.print(f"\n[bold yellow]{prompt}[/bold yellow]")
        answer = input("→ ").strip().lower()
        return answer in ("yes", "y")

    def _show(self, result: ExecutionResult, style: str = "green"):
        icon = "✓" if result.success else "✗"
        console.print(
            Panel(
                Text(f"{icon}  {result.message}", style=f"bold {style}"),
                border_style=style,
            )
        )


# ── Platform helpers ──────────────────────────────────────────────────────────
def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _is_mac() -> bool:
    return sys.platform == "darwin"
