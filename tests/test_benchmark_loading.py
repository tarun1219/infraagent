"""
Unit tests for IaCBenchmark loading, filtering, and task structure.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from iachench.benchmark import IaCBenchmark
from iachench.metrics import compute_metric, compute_pass_at_k


# ── IaCBenchmark Tests ───────────────────────────────────────────────────────

class TestIaCBenchmark:
    def setup_method(self):
        self.benchmark = IaCBenchmark()

    def test_get_tasks_returns_list(self):
        tasks = self.benchmark.get_tasks()
        assert isinstance(tasks, list)

    def test_get_tasks_limit(self):
        tasks = self.benchmark.get_tasks(limit=10)
        assert len(tasks) <= 10

    def test_task_has_required_fields(self):
        tasks = self.benchmark.get_tasks(limit=5)
        for task in tasks:
            assert "id" in task
            assert "type" in task
            assert "difficulty" in task
            assert "description" in task
            assert task["type"] in ("kubernetes", "terraform", "dockerfile")
            assert task["difficulty"] in (1, 2, 3, 4, 5)

    def test_filter_by_type(self):
        tasks = self.benchmark.get_tasks(task_type="kubernetes")
        for task in tasks:
            assert task["type"] == "kubernetes"

    def test_filter_by_difficulty(self):
        tasks = self.benchmark.get_tasks(difficulty=3)
        for task in tasks:
            assert task["difficulty"] == 3

    def test_filter_by_difficulty_range(self):
        tasks = self.benchmark.get_tasks(min_difficulty=2, max_difficulty=4)
        for task in tasks:
            assert 2 <= task["difficulty"] <= 4

    def test_task_count_approximate(self):
        # IaCBench has 300 tasks total
        tasks = self.benchmark.get_tasks()
        assert 200 <= len(tasks) <= 300

    def test_task_ids_unique(self):
        tasks = self.benchmark.get_tasks()
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"


# ── Metrics Tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_compute_metric_all_pass(self):
        results = [
            {"functional": True,  "security": True},
            {"functional": True,  "security": True},
            {"functional": True,  "security": True},
        ]
        metrics = compute_metric(results)
        assert metrics["functional_accuracy"] == pytest.approx(100.0)
        assert metrics["security_pass_rate"]  == pytest.approx(100.0)

    def test_compute_metric_all_fail(self):
        results = [
            {"functional": False, "security": False},
            {"functional": False, "security": False},
        ]
        metrics = compute_metric(results)
        assert metrics["functional_accuracy"] == pytest.approx(0.0)
        assert metrics["security_pass_rate"]  == pytest.approx(0.0)

    def test_compute_metric_mixed(self):
        results = [
            {"functional": True,  "security": True},
            {"functional": True,  "security": False},
            {"functional": False, "security": False},
            {"functional": False, "security": False},
        ]
        metrics = compute_metric(results)
        assert metrics["functional_accuracy"] == pytest.approx(50.0)
        assert metrics["security_pass_rate"]  == pytest.approx(25.0)

    def test_compute_metric_empty(self):
        metrics = compute_metric([])
        assert metrics["functional_accuracy"] == 0.0
        assert metrics["n"] == 0

    def test_compute_metric_by_difficulty(self):
        results = [
            {"functional": True,  "security": True,  "difficulty": 1},
            {"functional": True,  "security": True,  "difficulty": 1},
            {"functional": False, "security": False, "difficulty": 5},
            {"functional": False, "security": False, "difficulty": 5},
        ]
        metrics = compute_metric(results, by_difficulty=True)
        assert metrics["by_difficulty"][1]["functional_accuracy"] == pytest.approx(100.0)
        assert metrics["by_difficulty"][5]["functional_accuracy"] == pytest.approx(0.0)

    def test_pass_at_k_k1_equals_accuracy(self):
        results_per_task = [[True], [True], [False], [False]]
        p1 = compute_pass_at_k(results_per_task, k=1)
        assert p1 == pytest.approx(50.0)

    def test_pass_at_k_k3(self):
        # All tasks have at least one passing attempt
        results_per_task = [
            [True,  False, False],
            [False, True,  False],
            [False, False, True],
        ]
        p3 = compute_pass_at_k(results_per_task, k=3)
        assert p3 == pytest.approx(100.0)


# ── Task File Integrity Tests ─────────────────────────────────────────────────

class TestTaskFiles:
    def test_example_task_loadable(self):
        path = Path("examples/example_task_kubernetes.json")
        if path.exists():
            with open(path) as f:
                task = json.load(f)
            assert "id" in task
            assert "type" in task

    def test_experiment_results_loadable(self):
        path = Path("results/experiment_results.json")
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            assert "main_results" in data or "one_shot_deepseek" in data
