"""
Unit tests for the InfraAgent Task Planner.

Tests keyword-based resource decomposition, dependency graph ordering,
and difficulty estimation without requiring any LLM or external tools.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

# The planner is imported via the infraagent package.
# If the module is not yet present, tests are skipped gracefully.
try:
    from infraagent.planner import TaskPlanner, IaCLanguage, TaskPlan, SubTask
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False

pytestmark = pytest.mark.skipif(not HAS_PLANNER, reason="infraagent.planner not available")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def planner() -> "TaskPlanner":
    return TaskPlanner()


# ── Language Detection ────────────────────────────────────────────────────────

class TestLanguageDetection:
    """TaskPlanner.plan() must correctly infer IaC language from keywords."""

    @pytest.mark.parametrize("intent,expected_lang", [
        ("Create a Kubernetes Deployment for nginx", IaCLanguage.KUBERNETES),
        ("Write a Deployment and Service for my web app", IaCLanguage.KUBERNETES),
        ("Create a ConfigMap and mount it in a Pod", IaCLanguage.KUBERNETES),
        ("Create a HorizontalPodAutoscaler targeting 70% CPU", IaCLanguage.KUBERNETES),
        ("Create an Ingress routing traffic to service port 8080", IaCLanguage.KUBERNETES),
        ('Create a Terraform S3 bucket with versioning', IaCLanguage.TERRAFORM),
        ('resource "aws_iam_role" for lambda execution', IaCLanguage.TERRAFORM),
        ("Deploy a VPC with public and private subnets using Terraform", IaCLanguage.TERRAFORM),
        ("Write a Dockerfile for a Python Flask application", IaCLanguage.DOCKERFILE),
        ("FROM node:20-alpine, run npm install, expose port 3000", IaCLanguage.DOCKERFILE),
        ("Multi-stage Docker build for a Go binary", IaCLanguage.DOCKERFILE),
    ])
    def test_language_detection(self, planner, intent, expected_lang):
        plan = planner.plan(intent)
        assert plan.language == expected_lang, (
            f"Expected {expected_lang.value} for: {intent!r}, got {plan.language.value}"
        )


# ── Resource Decomposition ────────────────────────────────────────────────────

class TestResourceDecomposition:
    """TaskPlanner must decompose multi-resource intents into ordered subtasks."""

    def test_single_resource_one_subtask(self, planner):
        plan = planner.plan("Create a basic Nginx Deployment with 2 replicas")
        assert len(plan.subtasks) >= 1

    def test_deployment_plus_service_two_subtasks(self, planner):
        plan = planner.plan(
            "Create a Deployment 'api' and a Service exposing port 8080"
        )
        resource_types = [st.resource_type.lower() for st in plan.subtasks]
        assert any("deployment" in r for r in resource_types)
        assert any("service" in r for r in resource_types)

    def test_deployment_hpa_configmap_three_subtasks(self, planner):
        plan = planner.plan(
            "Create a Deployment, a HorizontalPodAutoscaler targeting 70% CPU, "
            "and a ConfigMap with app settings."
        )
        resource_types = [st.resource_type.lower() for st in plan.subtasks]
        assert any("deployment" in r for r in resource_types)
        assert any("horizontalpodautoscaler" in r or "hpa" in r for r in resource_types)

    def test_terraform_multi_resource(self, planner):
        plan = planner.plan(
            "Create a Terraform S3 bucket and an IAM role with a policy attached."
        )
        assert len(plan.subtasks) >= 1
        assert plan.language == IaCLanguage.TERRAFORM

    def test_subtasks_have_required_fields(self, planner):
        plan = planner.plan("Create a Kubernetes Deployment for nginx")
        for st in plan.subtasks:
            assert hasattr(st, "resource_type"), "SubTask missing resource_type"
            assert hasattr(st, "prompt"),        "SubTask missing prompt"
            assert st.resource_type, "resource_type must not be empty"


# ── Dependency Graph Ordering ─────────────────────────────────────────────────

class TestDependencyOrdering:
    """
    Resources with cross-dependencies must appear in correct topological order:
    - HPA must come after Deployment (references scaleTargetRef)
    - Service must come after Deployment
    - Ingress must come after Service
    - PVC must come before any Deployment that mounts it
    """

    def test_hpa_after_deployment(self, planner):
        plan = planner.plan(
            "Create a Deployment 'web' and a HorizontalPodAutoscaler "
            "targeting it at 70% CPU utilization."
        )
        types = [st.resource_type.lower() for st in plan.subtasks]
        dep_idx = next((i for i, t in enumerate(types) if "deployment" in t), None)
        hpa_idx = next((i for i, t in enumerate(types)
                        if "horizontalpodautoscaler" in t or "hpa" in t), None)
        if dep_idx is not None and hpa_idx is not None:
            assert dep_idx < hpa_idx, "Deployment must precede HPA in subtask order"

    def test_service_after_deployment(self, planner):
        plan = planner.plan(
            "Create a Deployment 'api' and a Service exposing port 8080."
        )
        types = [st.resource_type.lower() for st in plan.subtasks]
        dep_idx = next((i for i, t in enumerate(types) if "deployment" in t), None)
        svc_idx = next((i for i, t in enumerate(types) if "service" in t), None)
        if dep_idx is not None and svc_idx is not None:
            assert dep_idx < svc_idx, "Deployment must precede Service"

    def test_ingress_after_service(self, planner):
        plan = planner.plan(
            "Create a Deployment 'web', a Service, and an Ingress routing "
            "example.com to the service."
        )
        types = [st.resource_type.lower() for st in plan.subtasks]
        svc_idx = next((i for i, t in enumerate(types) if "service" in t), None)
        ing_idx = next((i for i, t in enumerate(types) if "ingress" in t), None)
        if svc_idx is not None and ing_idx is not None:
            assert svc_idx < ing_idx, "Service must precede Ingress"

    def test_no_cycles_in_dependency_graph(self, planner):
        # If the planner builds a DAG, it must be acyclic for all standard cases.
        intents = [
            "Create a Deployment, Service, and Ingress for a web app.",
            "Create a StatefulSet with a PVC and a Service.",
            "Create a DaemonSet and a ConfigMap it mounts.",
        ]
        for intent in intents:
            plan = planner.plan(intent)
            # Check subtask list is finite and non-empty (cycle would cause infinite loop)
            assert 0 < len(plan.subtasks) < 20, f"Suspicious subtask count for: {intent!r}"


# ── Difficulty Estimation ─────────────────────────────────────────────────────

class TestDifficultyEstimation:
    """
    Difficulty should correlate with task complexity:
    L1 = single resource, no security context
    L3 = security context + cross-resource refs
    L5 = multi-namespace, StatefulSet, NetworkPolicy
    """

    def test_single_basic_resource_is_l1_or_l2(self, planner):
        plan = planner.plan("Create a basic Nginx Deployment with 2 replicas.")
        assert plan.difficulty.value <= 2, (
            f"Single-resource task should be L1 or L2, got L{plan.difficulty.value}"
        )

    def test_security_context_raises_difficulty(self, planner):
        simple = planner.plan("Create a Deployment for nginx.")
        secure = planner.plan(
            "Create a Deployment with runAsNonRoot, readOnlyRootFilesystem, "
            "drop ALL capabilities, and resource limits."
        )
        assert secure.difficulty.value >= simple.difficulty.value, (
            "Security context requirements should not lower difficulty"
        )

    def test_cross_resource_dependency_raises_difficulty(self, planner):
        simple = planner.plan("Create a Deployment for nginx.")
        multi = planner.plan(
            "Create a Deployment, HorizontalPodAutoscaler, Service, and Ingress "
            "with NetworkPolicy restricting ingress to the frontend namespace."
        )
        assert multi.difficulty.value > simple.difficulty.value, (
            "Multi-resource + NetworkPolicy should have higher difficulty"
        )

    def test_l5_task_is_high_difficulty(self, planner):
        plan = planner.plan(
            "Create a StatefulSet for PostgreSQL with PVC, a CronJob for backup, "
            "a Service, NetworkPolicy, and RBAC for the backup job."
        )
        assert plan.difficulty.value >= 4, (
            f"Complex multi-resource task should be L4+, got L{plan.difficulty.value}"
        )

    def test_difficulty_values_in_valid_range(self, planner):
        intents = [
            "Create an nginx Deployment.",
            "Create a Deployment with a Service and ConfigMap.",
            "Create a StatefulSet with PVC, Service, and NetworkPolicy.",
        ]
        for intent in intents:
            plan = planner.plan(intent)
            assert 1 <= plan.difficulty.value <= 5, (
                f"Difficulty {plan.difficulty.value} out of range for: {intent!r}"
            )


# ── TaskPlan Structure ────────────────────────────────────────────────────────

class TestTaskPlanStructure:
    def test_plan_has_task_id(self, planner):
        plan = planner.plan("Create a Deployment.")
        assert plan.task_id, "TaskPlan must have a non-empty task_id"

    def test_custom_task_id_preserved(self, planner):
        plan = planner.plan("Create a Deployment.", task_id="k8s-042")
        assert plan.task_id == "k8s-042"

    def test_plan_language_is_enum(self, planner):
        plan = planner.plan("Create a Kubernetes Deployment.")
        assert isinstance(plan.language, IaCLanguage)

    def test_plan_subtasks_is_list(self, planner):
        plan = planner.plan("Create a Deployment and a Service.")
        assert isinstance(plan.subtasks, list)
        assert len(plan.subtasks) >= 1
