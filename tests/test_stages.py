"""The parallel parse step. Offline: run_stage is replaced, no stage runs."""

from __future__ import annotations

import threading
import time

import pytest

import rag_evaluation.stages as stages
from rag_evaluation.stages import StageError, _parse_batches


class TestParseBatches:
    def test_runs_up_to_workers_at_once(self, monkeypatch):
        running, peak, lock = 0, 0, threading.Lock()

        def fake(name, *args, env=None):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with lock:
                running -= 1
            return ""

        monkeypatch.setattr(stages, "run_stage", fake)
        _parse_batches([str(i) for i in range(10)], {}, workers=4)
        assert peak == 4

    def test_every_batch_runs_even_when_one_fails(self, monkeypatch):
        seen = []

        def fake(name, *args, env=None):
            batch = args[-1]
            seen.append(batch)
            if batch == "2":
                raise StageError("boom")
            return ""

        monkeypatch.setattr(stages, "run_stage", fake)
        with pytest.raises(StageError, match="batch 2: boom"):
            _parse_batches(["1", "2", "3"], {}, workers=2)
        assert sorted(seen) == ["1", "2", "3"]
