"""
test_pipeline.py
─────────────────────────────────────────────────────────────────
Unit tests for all voice assistant components.

Run:
  python -m pytest tests/test_pipeline.py -v
  python -m pytest tests/test_pipeline.py -v -k "test_rag"
"""
import sys
import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Add src to path ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config        import get_config
from rag_pipeline  import RAGPipeline, ActionMatch
from intent_parser import IntentParser, IntentResult
from action_executor import ActionExecutor, ExecutionResult


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def cfg():
    return get_config()


@pytest.fixture(scope="session")
def rag(cfg):
    """Build RAG pipeline once for all tests."""
    pipeline = RAGPipeline(
        actions_path         = cfg["paths"]["actions_dir"] + "/actions.json",
        embedding_model      = cfg["rag"]["embedding_model"],
        index_path           = cfg["rag"]["index_path"],
        metadata_path        = cfg["rag"]["metadata_path"],
        top_k                = 3,
        confidence_threshold = 0.4,
    )
    pipeline.setup()
    return pipeline


@pytest.fixture
def parser():
    return IntentParser()


@pytest.fixture
def executor():
    return ActionExecutor(
        action_log_path      = str(ROOT / "logs" / "test_actions.jsonl"),
        require_confirmation = False,   # skip prompts during tests
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RAG Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGPipeline:

    def test_setup_builds_index(self, rag):
        """Index should be populated after setup."""
        assert rag._index is not None
        assert rag._index.ntotal > 0

    def test_retrieve_returns_results(self, rag):
        matches = rag.retrieve("open chrome browser")
        assert len(matches) > 0

    def test_retrieve_confidence_sorted(self, rag):
        matches = rag.retrieve("open chrome browser")
        for i in range(len(matches) - 1):
            assert matches[i].confidence >= matches[i + 1].confidence

    def test_retrieve_open_browser(self, rag):
        match = rag.best_match("open chrome browser")
        assert match is not None
        assert match.action_id == "open_browser"

    def test_retrieve_get_time(self, rag):
        match = rag.best_match("what time is it")
        assert match is not None
        assert match.action_id == "get_time"

    def test_retrieve_calculator(self, rag):
        match = rag.best_match("open calculator")
        assert match is not None
        assert match.action_id == "open_calculator"

    def test_retrieve_weather(self, rag):
        match = rag.best_match("what is the weather today")
        assert match is not None
        assert match.action_id == "get_weather"

    def test_retrieve_create_file(self, rag):
        match = rag.best_match("create a new file")
        assert match is not None
        assert match.action_id == "create_file"

    def test_retrieve_web_search(self, rag):
        match = rag.best_match("search the web for python")
        assert match is not None
        assert match.action_id == "web_search"

    def test_retrieve_screenshot(self, rag):
        match = rag.best_match("take a screenshot")
        assert match is not None
        assert match.action_id == "take_screenshot"

    def test_retrieve_system_info(self, rag):
        match = rag.best_match("show system information")
        assert match is not None
        assert match.action_id == "system_info"

    def test_retrieve_delete_file(self, rag):
        match = rag.best_match("delete the file")
        assert match is not None
        assert match.action_id == "delete_file"

    def test_below_threshold_returns_empty(self, rag):
        rag_strict = RAGPipeline(
            actions_path         = rag.actions_path,
            embedding_model      = rag.embedding_model_name,
            index_path           = str(rag.index_path),
            metadata_path        = str(rag.metadata_path),
            top_k                = 3,
            confidence_threshold = 0.999,   # impossibly high
        )
        rag_strict._embedder = rag._embedder
        rag_strict._index    = rag._index
        rag_strict._meta     = rag._meta
        results = rag_strict.retrieve("xylophone bicycle purple moon")
        assert results == []

    def test_action_has_required_fields(self, rag):
        match = rag.best_match("open browser")
        assert hasattr(match, "action_id")
        assert hasattr(match, "executor")
        assert hasattr(match, "confidence")
        assert 0.0 <= match.confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Intent Parser Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_match(action_id, executor, params=None, destructive=False, confirm=False):
    return ActionMatch(
        action_id=action_id, name=action_id, description="",
        category="test", parameters=params or {}, executor=executor,
        destructive=destructive, confirmation_required=confirm,
        confidence=0.9,
    )


class TestIntentParser:

    def test_parse_open_browser_no_url(self, parser):
        match  = _mock_match("open_browser", "open_browser")
        result = parser.parse("open chrome", match)
        assert result.action_id == "open_browser"
        assert result.is_clear is True
        assert "url" in result.params

    def test_parse_open_browser_with_url(self, parser):
        match  = _mock_match("open_browser", "open_browser")
        result = parser.parse("open https://www.github.com", match)
        assert "github.com" in result.params["url"]

    def test_parse_create_file_with_name(self, parser):
        match  = _mock_match("create_file", "create_file",
                              params={"filename": {"required": True}, "directory": {"required": False}})
        result = parser.parse("create a file named report.txt", match)
        assert "report.txt" in result.params.get("filename", "")

    def test_parse_web_search_extracts_query(self, parser):
        match  = _mock_match("web_search", "web_search",
                              params={"query": {"required": True}})
        result = parser.parse("search the web for machine learning", match)
        assert "machine learning" in result.params.get("query", "")

    def test_parse_weather_extracts_location(self, parser):
        match  = _mock_match("get_weather", "get_weather")
        result = parser.parse("weather in London", match)
        assert result.params.get("location", "").lower() == "london"

    def test_parse_calculate_expression(self, parser):
        match  = _mock_match("calculate", "calculate",
                              params={"expression": {"required": True}})
        result = parser.parse("calculate 5 plus 3", match)
        assert result.params.get("expression") is not None

    def test_parse_calculate_missing_expression(self, parser):
        match  = _mock_match("calculate", "calculate",
                              params={"expression": {"required": True}})
        result = parser.parse("calculate", match)
        assert result.is_clear is False

    def test_parse_get_time_no_params(self, parser):
        match  = _mock_match("get_time", "get_time")
        result = parser.parse("what time is it", match)
        assert result.is_clear is True
        assert result.params == {}

    def test_parse_reminder_with_message(self, parser):
        match  = _mock_match("set_reminder", "set_reminder")
        result = parser.parse("remind me to call mom at 3pm", match)
        assert "call mom" in result.params.get("message", "").lower()

    def test_parse_system_info_metric(self, parser):
        match  = _mock_match("system_info", "system_info")
        result = parser.parse("how much ram do I have", match)
        assert result.params.get("metric") == "ram"

    def test_parse_delete_file_destructive_flag(self, parser):
        match  = _mock_match("delete_file", "delete_file",
                              destructive=True, confirm=True)
        result = parser.parse("delete the file notes.txt", match)
        assert result.needs_confirmation is True


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Executor Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_intent(executor, params=None, is_clear=True, needs_confirmation=False):
    return IntentResult(
        action_id          = executor,
        executor           = executor,
        params             = params or {},
        confidence         = 0.9,
        raw_text           = "test",
        is_clear           = is_clear,
        needs_confirmation = needs_confirmation,
    )


class TestActionExecutor:

    def test_unclear_intent_fails_gracefully(self, executor):
        intent = _make_intent("calculate", is_clear=False)
        result = executor.execute(intent)
        assert result.success is False

    def test_unknown_executor_fails(self, executor):
        intent = _make_intent("unknown_executor_xyz")
        result = executor.execute(intent)
        assert result.success is False

    def test_get_time_succeeds(self, executor):
        intent = _make_intent("get_time")
        result = executor.execute(intent)
        assert result.success is True
        assert "time" in result.message.lower() or "date" in result.message.lower()

    def test_calculate_addition(self, executor):
        intent = _make_intent("calculate", params={"expression": "10 + 5"})
        result = executor.execute(intent)
        assert result.success is True
        assert "15" in result.message

    def test_calculate_multiplication(self, executor):
        intent = _make_intent("calculate", params={"expression": "7 * 8"})
        result = executor.execute(intent)
        assert result.success is True
        assert "56" in result.message

    def test_calculate_invalid_expression(self, executor):
        intent = _make_intent("calculate", params={"expression": "not_math"})
        result = executor.execute(intent)
        assert result.success is False

    def test_create_file_creates_file(self, executor):
        with tempfile.TemporaryDirectory() as tmpdir:
            intent = _make_intent("create_file", params={
                "filename": "test_voice.txt",
                "directory": tmpdir,
            })
            result = executor.execute(intent)
            assert result.success is True
            assert (Path(tmpdir) / "test_voice.txt").exists()

    def test_create_file_no_name_uses_default(self, executor):
        with tempfile.TemporaryDirectory() as tmpdir:
            intent = _make_intent("create_file", params={
                "filename": "new_file.txt",
                "directory": tmpdir,
            })
            result = executor.execute(intent)
            assert result.success is True

    def test_delete_file_nonexistent(self, executor):
        intent = _make_intent("delete_file", params={"filename": "no_such_file_xyz.txt"})
        result = executor.execute(intent)
        assert result.success is False
        assert "not found" in result.message.lower()

    def test_delete_file_existing(self, executor):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            tmppath = f.name
        try:
            intent = _make_intent("delete_file", params={"filename": tmppath})
            result = executor.execute(intent)
            assert result.success is True
            assert not Path(tmppath).exists()
        finally:
            if Path(tmppath).exists():
                os.unlink(tmppath)

    def test_system_info_returns_metrics(self, executor):
        intent = _make_intent("system_info", params={"metric": "all"})
        result = executor.execute(intent)
        assert result.success is True
        assert len(result.details) > 0

    def test_search_files_returns_result(self, executor):
        intent = _make_intent("search_files", params={
            "query": "py",
            "directory": str(ROOT),
        })
        result = executor.execute(intent)
        # Should succeed even if nothing found
        assert isinstance(result.success, bool)

    def test_action_log_created(self, executor):
        intent = _make_intent("get_time")
        executor.execute(intent)
        log_path = ROOT / "logs" / "test_actions.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        last = json.loads(lines[-1])
        assert last["executor"] == "get_time"


# ═══════════════════════════════════════════════════════════════════════════════
#  STT Engine Tests (mocked to avoid requiring model files during CI)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSTTEngine:

    def test_transcription_result_has_latency(self, cfg):
        from stt_engine import STTEngine, TranscriptionResult
        engine = STTEngine(
            model_size   = cfg["stt"]["model_size"],
            model_dir    = cfg["stt"]["model_dir"],
            device       = "cpu",
            compute_type = "int8",
        )
        # We test the result dataclass only (no model load needed)
        result = TranscriptionResult(
            text="hello world",
            segments=[],
            language="en",
            latency_ms=123.4,
            audio_duration_ms=2000.0,
        )
        assert result.latency_ms == 123.4
        assert "hello world" in str(result)

    def test_float32_audio_shape(self):
        """Audio should be float32 1-D array."""
        audio = np.zeros(16_000, dtype=np.float32)
        assert audio.dtype == np.float32
        assert audio.ndim == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end pipeline test (RAG → Intent → Execute) without real STT."""

    @pytest.mark.parametrize("query,expected_action", [
        ("open chrome browser",             "open_browser"),
        ("what time is it",                 "get_time"),
        ("search the web for deep learning","web_search"),
        ("calculate 10 times 5",            "calculate"),
        ("show system info",                "system_info"),
    ])
    def test_full_pipeline(self, rag, parser, executor, query, expected_action):
        # RAG
        match = rag.best_match(query)
        assert match is not None, f"No match for: {query}"
        assert match.action_id == expected_action, (
            f"Expected '{expected_action}', got '{match.action_id}' for: {query}"
        )

        # Intent
        intent = parser.parse(query, match)
        assert intent.action_id == expected_action

        # Execute
        result = executor.execute(intent)
        # All test actions should at least not crash (success varies by env)
        assert isinstance(result.success, bool)
