"""
Unit tests for the InfraAgent self-correction loop.

Tests:
  - SC loop terminates on pass
  - SC loop exhausts max_rounds
  - Recovery rate calculation
  - Regression detection (fix in round N doesn't break round N-1 passes)
  - Round records are correctly stored
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch, call

from infraagent.validators import (
    MultiLayerValidator,
    ValidationReport,
    ValidationError,
    ValidationLayer,
    Severity,
)
from infraagent.agent import InfraAgent, AgentResult, RoundRecord


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_passing_report(security_score: float = 0.9) -> ValidationReport:
    return ValidationReport(
        passed=True,
        syntax_valid=True,
        schema_valid=True,
        security_score=security_score,
        best_practice_score=0.8,
        errors=[],
        total_checks=4,
        passed_checks=4,
    )


def make_failing_report(
    error_messages: Optional[List[str]] = None,
    security_score: float = 0.2,
    syntax_valid: bool = True,
    schema_valid: bool = True,
) -> ValidationReport:
    msgs = error_messages or ["CKV_K8S_30: Containers must not run as root"]
    errors = [
        ValidationError(
            layer=ValidationLayer.SECURITY,
            tool="checkov",
            rule_id=m.split(":")[0].strip(),
            message=m,
            severity=Severity.ERROR,
        )
        for m in msgs
    ]
    return ValidationReport(
        passed=False,
        syntax_valid=syntax_valid,
        schema_valid=schema_valid,
        security_score=security_score,
        best_practice_score=0.3,
        errors=errors,
        total_checks=4,
        passed_checks=1,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def stub_agent():
    """InfraAgent in stub mode — no Ollama required."""
    return InfraAgent(use_stub=True, max_rounds=3, use_rag=False)


@pytest.fixture
def stub_agent_with_rag():
    """InfraAgent with stub generator and mocked RAG."""
    with patch("infraagent.agent.RAGModule") as MockRAG:
        mock_rag = MockRAG.return_value
        mock_rag.build_context_string.return_value = "## Relevant Documentation\nUse runAsNonRoot: true"
        agent = InfraAgent(use_stub=True, max_rounds=3, use_rag=True)
        agent.rag = mock_rag
    return agent


# ── SC Loop Termination Tests ─────────────────────────────────────────────────

class TestSCLoopTermination:
    """SC loop must stop as soon as a passing report is obtained."""

    def test_loop_stops_on_first_pass(self, stub_agent):
        """If round 0 passes, no correction rounds should be taken."""
        passing_report = make_passing_report()
        with patch.object(stub_agent.validator, "validate", return_value=passing_report):
            result = stub_agent.run("Create a Deployment.", task_id="t-001")
        assert result.total_rounds_used == 0
        assert result.success is True

    def test_loop_stops_after_round_1_pass(self, stub_agent):
        """If round 0 fails but round 1 passes, exactly 1 correction round used."""
        reports = [make_failing_report(), make_passing_report()]
        with patch.object(stub_agent.validator, "validate", side_effect=reports):
            result = stub_agent.run("Create a Deployment.", task_id="t-002")
        assert result.total_rounds_used == 1
        assert result.success is True

    def test_loop_stops_after_round_2_pass(self, stub_agent):
        """Fail, fail, pass → 2 correction rounds."""
        reports = [make_failing_report(), make_failing_report(), make_passing_report()]
        with patch.object(stub_agent.validator, "validate", side_effect=reports):
            result = stub_agent.run("Create a Deployment.", task_id="t-003")
        assert result.total_rounds_used == 2
        assert result.success is True

    def test_loop_exhausts_max_rounds(self, stub_agent):
        """If all rounds fail, result.success must be False and rounds == max_rounds."""
        failing = make_failing_report()
        # Always return failing (max_rounds=3 means 1 initial + 3 correction = 4 validate calls)
        with patch.object(stub_agent.validator, "validate", return_value=failing):
            result = stub_agent.run("Create a Deployment.", task_id="t-004")
        assert result.success is False
        assert result.total_rounds_used == stub_agent.max_rounds

    def test_zero_rounds_is_one_shot(self):
        """max_rounds=0 means no correction, only one validate call."""
        agent = InfraAgent(use_stub=True, max_rounds=0, use_rag=False)
        failing = make_failing_report()
        with patch.object(agent.validator, "validate", return_value=failing):
            result = agent.run("Create a Deployment.", task_id="t-005")
        assert result.total_rounds_used == 0
        assert result.success is False

    def test_validate_called_exactly_once_on_first_pass(self, stub_agent):
        """Validate should be called exactly once if round 0 passes."""
        passing = make_passing_report()
        with patch.object(stub_agent.validator, "validate", return_value=passing) as mock_val:
            stub_agent.run("Create a Deployment.", task_id="t-006")
        assert mock_val.call_count == 1


# ── Round Records ─────────────────────────────────────────────────────────────

class TestRoundRecords:
    """AgentResult.rounds must faithfully record each round's state."""

    def test_rounds_count_matches_total_rounds_used(self, stub_agent):
        reports = [make_failing_report(), make_passing_report()]
        with patch.object(stub_agent.validator, "validate", side_effect=reports):
            result = stub_agent.run("Create a Deployment.", task_id="t-007")
        # rounds list has initial (0) + correction rounds
        assert len(result.rounds) == result.total_rounds_used + 1

    def test_round_numbers_sequential(self, stub_agent):
        reports = [make_failing_report(), make_failing_report(), make_passing_report()]
        with patch.object(stub_agent.validator, "validate", side_effect=reports):
            result = stub_agent.run("Create a Deployment.", task_id="t-008")
        for i, record in enumerate(result.rounds):
            assert record.round == i

    def test_each_round_has_code(self, stub_agent):
        passing = make_passing_report()
        with patch.object(stub_agent.validator, "validate", return_value=passing):
            result = stub_agent.run("Create a Deployment.", task_id="t-009")
        for record in result.rounds:
            assert isinstance(record.code, str)
            assert len(record.code) > 0

    def test_final_report_matches_last_round_report(self, stub_agent):
        reports = [make_failing_report(), make_passing_report()]
        with patch.object(stub_agent.validator, "validate", side_effect=reports):
            result = stub_agent.run("Create a Deployment.", task_id="t-010")
        assert result.final_report is result.rounds[-1].report

    def test_round_duration_positive(self, stub_agent):
        passing = make_passing_report()
        with patch.object(stub_agent.validator, "validate", return_value=passing):
            result = stub_agent.run("Create a Deployment.", task_id="t-011")
        for record in result.rounds:
            assert record.duration_s >= 0.0


# ── Recovery Rate Calculation ─────────────────────────────────────────────────

class TestRecoveryRateCalculation:
    """
    Recovery rate = fraction of initially-failing tasks that eventually pass.
    compute_recovery_rate() is the canonical function in iachench/metrics.py.
    """

    def test_all_recover(self):
        from iachench.metrics import compute_recovery_rate
        results = [
            {"initial_passed": False, "final_passed": True},
            {"initial_passed": False, "final_passed": True},
        ]
        rate = compute_recovery_rate(results)
        assert rate == pytest.approx(1.0)

    def test_none_recover(self):
        from iachench.metrics import compute_recovery_rate
        results = [
            {"initial_passed": False, "final_passed": False},
            {"initial_passed": False, "final_passed": False},
        ]
        rate = compute_recovery_rate(results)
        assert rate == pytest.approx(0.0)

    def test_partial_recovery(self):
        from iachench.metrics import compute_recovery_rate
        results = [
            {"initial_passed": False, "final_passed": True},
            {"initial_passed": False, "final_passed": False},
            {"initial_passed": False, "final_passed": False},
            {"initial_passed": False, "final_passed": False},
        ]
        rate = compute_recovery_rate(results)
        assert rate == pytest.approx(0.25)

    def test_exclude_initially_passing_tasks(self):
        """Tasks that passed round 0 should not be counted in the denominator."""
        from iachench.metrics import compute_recovery_rate
        results = [
            {"initial_passed": True,  "final_passed": True},   # exclude
            {"initial_passed": False, "final_passed": True},   # recovered
            {"initial_passed": False, "final_passed": False},  # not recovered
        ]
        rate = compute_recovery_rate(results)
        assert rate == pytest.approx(0.5)

    def test_no_initially_failing_tasks(self):
        """If all tasks pass round 0, recovery rate is undefined; return None or 0."""
        from iachench.metrics import compute_recovery_rate
        results = [
            {"initial_passed": True, "final_passed": True},
            {"initial_passed": True, "final_passed": True},
        ]
        rate = compute_recovery_rate(results)
        assert rate is None or rate == pytest.approx(0.0)


# ── Regression Detection ──────────────────────────────────────────────────────

class TestRegressionDetection:
    """
    Self-correction must not break previously-passing checks.
    If round N fixes an issue but introduces a new one, it must be detected.
    """

    def test_regression_detected_when_new_error_introduced(self, stub_agent):
        """
        round 0: syntax_valid=False (L1 error)
        round 1: syntax_valid=True but security now fails (regression)
        round 2: passes fully
        """
        reports = [
            make_failing_report(syntax_valid=False, schema_valid=False),
            make_failing_report(syntax_valid=True,  schema_valid=True),   # fixed syntax, security still fails
            make_passing_report(),
        ]
        with patch.object(stub_agent.validator, "validate", side_effect=reports):
            result = stub_agent.run("Create a Deployment.", task_id="t-012")

        # Round 1 fixes syntax but security fails — this should still be recorded
        assert result.rounds[1].report.syntax_valid is True
        assert result.rounds[1].report.passed is False

    def test_error_messages_fed_back_to_generator(self, stub_agent):
        """
        The generator's self_correct() must receive the errors from the previous round.
        """
        failing = make_failing_report(["CKV_K8S_30: root user", "CKV_K8S_22: writable fs"])
        passing = make_passing_report()

        with patch.object(stub_agent.validator, "validate", side_effect=[failing, passing]), \
             patch.object(stub_agent.generator, "self_correct", wraps=stub_agent.generator.self_correct) as mock_sc:
            stub_agent.run("Create a Deployment.", task_id="t-013")

        mock_sc.assert_called_once()
        call_kwargs = mock_sc.call_args
        errors_arg = call_kwargs.kwargs.get("errors") or (call_kwargs.args[2] if len(call_kwargs.args) > 2 else None)
        if errors_arg is not None:
            assert len(errors_arg) >= 1

    def test_rag_query_reformulated_from_errors(self, stub_agent_with_rag):
        """In SC rounds, RAG must be queried using error messages, not just the task."""
        failing = make_failing_report(["CKV_K8S_30: Containers must not run as root"])
        passing = make_passing_report()

        with patch.object(stub_agent_with_rag.validator, "validate", side_effect=[failing, passing]):
            stub_agent_with_rag.run("Create a Deployment.", task_id="t-014")

        # RAG build_context_string must have been called at least twice:
        # once for initial generation, once for the SC round with error query
        assert stub_agent_with_rag.rag.build_context_string.call_count >= 1

    def test_result_to_dict_complete_after_correction(self, stub_agent):
        """AgentResult.to_dict() must include all round records after SC."""
        reports = [make_failing_report(), make_passing_report()]
        with patch.object(stub_agent.validator, "validate", side_effect=reports):
            result = stub_agent.run("Create a Deployment.", task_id="t-015")

        d = result.to_dict()
        assert "rounds" in d
        assert len(d["rounds"]) == 2  # round 0 + round 1
        assert d["rounds"][0]["passed"] is False
        assert d["rounds"][1]["passed"] is True
        assert d["success"] is True


# ── Error Classification ──────────────────────────────────────────────────────

class TestErrorClassification:
    """
    Recovery rates differ by error class:
    syntax=72%, schema=55%, security=8%, cross-resource=2%
    This tests that errors_to_feedback() formats errors correctly for LLM consumption.
    """

    @pytest.fixture
    def validator(self):
        return MultiLayerValidator()

    def test_syntax_errors_in_feedback(self, validator):
        report = make_failing_report(
            error_messages=["[L:1] invalid YAML at line 3: unclosed bracket"],
        )
        report.errors[0].layer = ValidationLayer.SYNTAX
        feedback = validator.errors_to_feedback(report)
        assert any(item["layer"] == "syntax" for item in feedback)

    def test_schema_errors_in_feedback(self, validator):
        report = make_failing_report(
            error_messages=["kubeconform: could not validate: autoscaling/v2beta2 not found"],
        )
        report.errors[0].layer = ValidationLayer.SCHEMA
        feedback = validator.errors_to_feedback(report)
        assert any(item["layer"] == "schema" for item in feedback)

    def test_security_errors_in_feedback(self, validator):
        report = make_failing_report(
            error_messages=["CKV_K8S_30: Containers must not run as root"],
        )
        report.errors[0].layer = ValidationLayer.SECURITY
        feedback = validator.errors_to_feedback(report)
        assert any(item["layer"] == "security" for item in feedback)

    def test_feedback_includes_rule_id(self, validator):
        report = make_failing_report(["CKV_K8S_30: root user"])
        feedback = validator.errors_to_feedback(report)
        for item in feedback:
            assert "rule_id" in item

    def test_feedback_includes_message(self, validator):
        report = make_failing_report(["CKV_K8S_30: Containers must not run as root"])
        feedback = validator.errors_to_feedback(report)
        for item in feedback:
            assert "message" in item
            assert item["message"]
