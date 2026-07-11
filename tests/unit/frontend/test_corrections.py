"""Unit tests for per-span human-correction persistence."""

from __future__ import annotations

from src.frontend.corrections import load_correction, save_correction


class TestSaveAndLoadCorrection:
    def test_round_trip(self, tmp_path):
        save_correction("trace-1", "span-1", "the expected answer", tmp_path)

        assert load_correction("trace-1", "span-1", tmp_path) == "the expected answer"

    def test_missing_file_returns_none(self, tmp_path):
        assert load_correction("no-such-trace", "span-1", tmp_path) is None

    def test_missing_span_in_existing_file_returns_none(self, tmp_path):
        save_correction("trace-1", "span-1", "answer", tmp_path)

        assert load_correction("trace-1", "span-2", tmp_path) is None

    def test_second_span_does_not_clobber_first(self, tmp_path):
        save_correction("trace-1", "span-1", "answer one", tmp_path)
        save_correction("trace-1", "span-2", "answer two", tmp_path)

        assert load_correction("trace-1", "span-1", tmp_path) == "answer one"
        assert load_correction("trace-1", "span-2", tmp_path) == "answer two"

    def test_overwrite_updates_existing_correction(self, tmp_path):
        save_correction("trace-1", "span-1", "first draft", tmp_path)
        save_correction("trace-1", "span-1", "corrected draft", tmp_path)

        assert load_correction("trace-1", "span-1", tmp_path) == "corrected draft"

    def test_different_traces_are_independent(self, tmp_path):
        save_correction("trace-1", "span-1", "trace one answer", tmp_path)
        save_correction("trace-2", "span-1", "trace two answer", tmp_path)

        assert load_correction("trace-1", "span-1", tmp_path) == "trace one answer"
        assert load_correction("trace-2", "span-1", tmp_path) == "trace two answer"
