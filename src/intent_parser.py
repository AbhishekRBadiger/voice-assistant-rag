"""
intent_parser.py
─────────────────────────────────────────────────────────────────
Rule-based parameter extractor — Python 3.10 · Windows · offline

Extracts structured intent + parameters from raw transcribed text.
No external APIs or LLM required — works 100% offline.
"""
from __future__ import annotations

import re
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger

from rag_pipeline import ActionMatch


# ── Helpers ───────────────────────────────────────────────────────────────────
_MATH_WORDS = {
    "plus": "+", "add": "+", "and": "+",
    "minus": "-", "subtract": "-", "take away": "-",
    "times": "*", "multiplied by": "*", "multiply": "*",
    "divided by": "/", "divide": "/", "over": "/",
    "squared": "**2", "cubed": "**3",
    "percent": "/100",
}

_APP_MAP = {
    "chrome": "chrome", "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge", "microsoft edge": "msedge",
    "safari": "safari",
    "notepad": "notepad", "note": "notepad",
    "wordpad": "wordpad",
    "calculator": "calc", "calc": "calc",
    "explorer": "explorer", "file explorer": "explorer",
    "terminal": "cmd", "command prompt": "cmd", "cmd": "cmd",
    "powershell": "powershell",
    "paint": "mspaint",
    "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
    "vlc": "vlc", "spotify": "spotify", "discord": "discord",
    "vscode": "code", "vs code": "code", "visual studio code": "code",
}


@dataclass
class IntentResult:
    """Fully parsed intent ready for the executor."""
    action_id:   str
    executor:    str
    params:      dict = field(default_factory=dict)
    confidence:  float = 0.0
    raw_text:    str = ""
    is_clear:    bool = True         # False = ask user to rephrase
    needs_confirmation: bool = False  # for destructive actions


class IntentParser:
    """
    Rule-based parameter extractor that works fully offline.

    No external APIs or LLM required.
    """

    def parse(
        self,
        text: str,
        action_match: ActionMatch,
    ) -> IntentResult:
        """
        Parse `text` given the matched action and extract its parameters.
        """
        t0     = time.perf_counter()
        lower  = text.lower().strip()
        params = {}

        # ── Per-executor extraction ───────────────────────────────────────
        executor = action_match.executor

        if executor == "open_browser":
            params["url"] = self._extract_url(lower)

        elif executor == "open_notepad":
            pass   # no params

        elif executor == "open_calculator":
            pass

        elif executor == "open_file_manager":
            params["path"] = self._extract_path(lower)

        elif executor == "create_file":
            params["filename"]  = self._extract_filename(lower) or "new_file.txt"
            params["directory"] = self._extract_directory(lower) or "~/Desktop"

        elif executor == "delete_file":
            params["filename"] = self._extract_filename(lower)

        elif executor == "search_files":
            params["query"]     = self._extract_search_query(lower, "file")
            params["directory"] = self._extract_directory(lower) or "~"

        elif executor == "get_weather":
            params["location"] = self._extract_location(lower) or "current"

        elif executor == "set_reminder":
            params["message"] = self._extract_reminder_message(lower)
            params["time"]    = self._extract_time(lower)

        elif executor == "calculate":
            params["expression"] = self._extract_math_expression(lower)
            if not params["expression"]:
                # mark as unclear so the UI can ask for the expression
                return IntentResult(
                    action_id  = action_match.action_id,
                    executor   = executor,
                    params     = {},
                    confidence = action_match.confidence,
                    raw_text   = text,
                    is_clear   = False,
                )

        elif executor == "web_search":
            params["query"] = self._extract_search_query(lower, "web")

        elif executor == "get_time":
            pass

        elif executor == "take_screenshot":
            params["filename"] = self._extract_filename(lower)

        elif executor == "system_info":
            params["metric"] = self._extract_system_metric(lower)

        elif executor == "open_terminal":
            pass

        elif executor in ("open_browser", "open_notepad", "open_calculator"):
            # open_application group
            params["app_name"] = self._extract_app_name(lower)

        # ── Determine clarity ─────────────────────────────────────────────
        # A command is "unclear" if a required parameter is missing
        required_missing = any(
            p_def.get("required", False) and not params.get(p_name)
            for p_name, p_def in action_match.parameters.items()
        )
        is_clear = not required_missing

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug(
            f"[intent] {executor}({params}) "
            f"conf={action_match.confidence:.2f} "
            f"clear={is_clear} "
            f"in {elapsed_ms:.1f}ms"
        )

        return IntentResult(
            action_id           = action_match.action_id,
            executor            = executor,
            params              = params,
            confidence          = action_match.confidence,
            raw_text            = text,
            is_clear            = is_clear,
            needs_confirmation  = action_match.confirmation_required,
        )

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_url(self, text: str) -> str:
        # explicit URL
        m = re.search(r"https?://\S+", text)
        if m:
            return m.group()
        # "go to X" / "open X" / "navigate to X"
        m = re.search(
            r"(?:go to|navigate to|open|visit|browse to)\s+([\w\.-]+(?:\.com|\.org|\.net|\.io|\.edu)[/\w\.-]*)",
            text,
        )
        if m:
            return "https://" + m.group(1)
        return "https://www.google.com"

    def _extract_filename(self, text: str) -> Optional[str]:
        # "named X" / "called X" / file with extension
        m = re.search(r"(?:named?|called?|file)\s+([\w\-. ]+\.\w+)", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"([\w\-]+\.\w+)", text)
        if m:
            return m.group(1)
        # bare word after keyword
        m = re.search(
            r"(?:create|delete|remove|make|new)\s+(?:a\s+)?(?:file|document)?\s*([\w\-]+)",
            text,
        )
        if m and len(m.group(1)) > 1:
            name = m.group(1).strip()
            return name if "." in name else name + ".txt"
        return None

    def _extract_directory(self, text: str) -> Optional[str]:
        m = re.search(r"(?:in|inside|at|to|from)\s+([\~/\w\\ ]+)", text)
        if m:
            return m.group(1).strip()
        return None

    def _extract_path(self, text: str) -> Optional[str]:
        m = re.search(r"([A-Za-z]:[/\\][\w/\\ ]+|/[\w/\\ ]+)", text)
        if m:
            return m.group(1)
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        # "weather in London" / "weather for New York"
        m = re.search(
            r"(?:weather|temperature|forecast|rain|sun)\s+(?:in|at|for|of)\s+([\w ]+)",
            text,
        )
        if m:
            return m.group(1).strip().title()
        # "in London weather"
        m = re.search(r"in\s+([\w ]+)\s+weather", text)
        if m:
            return m.group(1).strip().title()
        return None

    def _extract_time(self, text: str) -> Optional[str]:
        # "at 3pm" / "at 15:30" / "in 10 minutes"
        m = re.search(
            r"(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
            r"|(?:in\s+(\d+)\s+(?:minutes?|hours?))",
            text,
        )
        if m:
            return m.group(0).strip()
        return None

    def _extract_reminder_message(self, text: str) -> str:
        # "remind me to <message>" / "reminder to <message>"
        m = re.search(r"remind\s+me\s+to\s+(.+?)(?:\s+at|\s+in|$)", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"reminder\s+(?:to\s+)?(.+?)(?:\s+at|\s+in|$)", text)
        if m:
            return m.group(1).strip()
        return "Reminder"

    def _extract_math_expression(self, text: str) -> Optional[str]:
        # Normalise word numbers / operators
        expr = text
        for word, sym in sorted(_MATH_WORDS.items(), key=lambda x: -len(x[0])):
            expr = re.sub(r"\b" + re.escape(word) + r"\b", sym, expr)

        # Try to pull out a numeric expression
        m = re.search(r"[\d\s\+\-\*\/\(\)\.\%\^]+", expr)
        if m:
            candidate = m.group().strip()
            if any(c.isdigit() for c in candidate):
                return candidate
        return None

    def _extract_search_query(self, text: str, mode: str = "web") -> Optional[str]:
        patterns = [
            r"(?:search(?:ing)? (?:the (?:web|internet) )?for|google|look up|find|search)\s+(.+)",
            r"(?:search|find)\s+(.+)",
        ]
        if mode == "file":
            patterns.insert(
                0,
                r"(?:search|find|look for|locate)\s+(?:file|files|document)?\s*(?:named?|called?)?\s*(.+)",
            )
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()
        return text  # fall back to full text

    def _extract_app_name(self, text: str) -> str:
        for key, val in sorted(_APP_MAP.items(), key=lambda x: -len(x[0])):
            if key in text:
                return val
        # generic: last word
        words = text.split()
        return words[-1] if words else "notepad"

    def _extract_system_metric(self, text: str) -> str:
        for metric in ("cpu", "ram", "memory", "disk", "storage"):
            if metric in text:
                return "ram" if metric == "memory" else metric
        return "all"
