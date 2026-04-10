"""
Task Planner for InfraAgent.

Performs keyword-based intent analysis:
  - IaC language detection (Kubernetes / Terraform / Dockerfile)
  - Multi-resource decomposition into ordered subtasks
  - Dependency graph ordering (Deployment → HPA, Service → Ingress, …)
  - Difficulty estimation (L1–L5)
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class IaCLanguage(str, Enum):
    KUBERNETES = "kubernetes"
    TERRAFORM  = "terraform"
    DOCKERFILE = "dockerfile"


class DifficultyLevel(int, Enum):
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


@dataclass
class SubTask:
    resource_type: str
    prompt:        str
    depends_on:    List[str] = field(default_factory=list)
    # Extra fields expected by generator.py
    constraints:   List[str] = field(default_factory=list)
    language:      "IaCLanguage" = field(default=None, init=False)  # set by TaskPlan


@dataclass
class TaskPlan:
    task_id:           str
    language:          IaCLanguage
    difficulty:        DifficultyLevel
    subtasks:          List[SubTask]
    raw_intent:        str
    # Aliases expected by generator.py
    original_intent:   str = field(default="", init=False)
    integration_notes: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.original_intent   = self.raw_intent
        self.integration_notes = ""
        # Propagate language to subtasks so generator.py can access subtask.language
        for st in self.subtasks:
            st.language = self.language


# ── Keyword tables ────────────────────────────────────────────────────────────

_K8S_KEYWORDS = {
    "kubernetes", "deployment", "service", "pod", "configmap", "secret",
    "ingress", "namespace", "statefulset", "daemonset", "job", "cronjob",
    "horizontalpodautoscaler", "hpa", "networkpolicy", "persistentvolumeclaim",
    "pvc", "rbac", "clusterrole", "rolebinding", "serviceaccount",
    "kubectl", "k8s", "kube",
}
_TF_KEYWORDS = {
    "terraform", "aws_", "resource \"", "provider", "module",
    "s3", "iam", "vpc", "subnet", "rds", "lambda", "eks", "ecr",
    "localstack", "hcl", "tfstate",
}
_DOCKERFILE_KEYWORDS = {
    "dockerfile", "from ", "docker", "run ", "copy ", "expose", "entrypoint",
    "multi-stage", "distroless", "useradd", "groupadd",
}

# Resources and their canonical names + dependency ordering
_K8S_RESOURCES_ORDER = [
    "configmap", "secret", "persistentvolumeclaim", "serviceaccount",
    "clusterrole", "rolebinding", "deployment", "statefulset", "daemonset",
    "job", "cronjob", "service", "horizontalpodautoscaler", "ingress",
    "networkpolicy",
]
_TF_RESOURCES_ORDER = [
    "vpc", "subnet", "security_group", "iam_role", "iam_policy",
    "s3_bucket", "rds", "lambda", "eks",
]
_DIFFICULTY_SECURITY_KEYWORDS = {
    "runasnonroot", "readonlyrootfilesystem", "drop all", "capabilities",
    "securitycontext", "seccomp", "pss", "restricted", "allowprivilegeescalation",
    "non-root", "nonroot",
}
_DIFFICULTY_COMPLIANCE_KEYWORDS = {
    "admission", "podsecurity", "kms", "encryption", "distroless",
    "least privilege", "wildcard", "compliance", "cis",
}


class TaskPlanner:
    """Decomposes a natural-language IaC intent into an ordered TaskPlan."""

    def plan(self, intent: str, task_id: Optional[str] = None) -> TaskPlan:
        lower = intent.lower()
        language   = self._detect_language(lower)
        subtasks   = self._decompose(intent, lower, language)
        difficulty = self._estimate_difficulty(lower, subtasks)
        tid        = task_id or f"{language.value[:2]}-{uuid.uuid4().hex[:6]}"
        return TaskPlan(
            task_id=tid,
            language=language,
            difficulty=difficulty,
            subtasks=subtasks,
            raw_intent=intent,
        )

    # ── Language detection ────────────────────────────────────────────────────

    def _detect_language(self, lower: str) -> IaCLanguage:
        tf_score  = sum(1 for kw in _TF_KEYWORDS      if kw in lower)
        k8s_score = sum(1 for kw in _K8S_KEYWORDS     if kw in lower)
        df_score  = sum(1 for kw in _DOCKERFILE_KEYWORDS if kw in lower)

        # Dockerfile wins on explicit FROM/Dockerfile mentions
        if "dockerfile" in lower or ("from " in lower and "docker" in lower):
            return IaCLanguage.DOCKERFILE
        # Terraform wins on aws_/terraform/resource " pattern
        if tf_score > k8s_score and tf_score > df_score:
            return IaCLanguage.TERRAFORM
        if df_score > k8s_score:
            return IaCLanguage.DOCKERFILE
        return IaCLanguage.KUBERNETES  # default

    # ── Resource decomposition ────────────────────────────────────────────────

    def _decompose(self, intent: str, lower: str, language: IaCLanguage) -> List[SubTask]:
        if language == IaCLanguage.KUBERNETES:
            return self._decompose_k8s(intent, lower)
        if language == IaCLanguage.TERRAFORM:
            return self._decompose_tf(intent, lower)
        return [SubTask(resource_type="Dockerfile", prompt=intent)]

    def _decompose_k8s(self, intent: str, lower: str) -> List[SubTask]:
        found: List[str] = []
        for resource in _K8S_RESOURCES_ORDER:
            # also check abbreviations
            aliases = {"horizontalpodautoscaler": ["hpa", "horizontalpodautoscaler"]}
            checks = aliases.get(resource, [resource])
            if any(c in lower for c in checks):
                found.append(resource)

        if not found:
            found = ["deployment"]

        # Dependency ordering — already encoded in _K8S_RESOURCES_ORDER index
        found_sorted = sorted(found, key=lambda r: _K8S_RESOURCES_ORDER.index(r)
                              if r in _K8S_RESOURCES_ORDER else 99)

        subtasks = []
        for i, r in enumerate(found_sorted):
            deps = found_sorted[:i]
            subtasks.append(SubTask(
                resource_type=r.capitalize(),
                prompt=f"From the following intent, generate the {r} resource only:\n{intent}",
                depends_on=deps,
            ))
        return subtasks

    def _decompose_tf(self, intent: str, lower: str) -> List[SubTask]:
        found: List[str] = []
        for resource in _TF_RESOURCES_ORDER:
            if resource.replace("_", " ") in lower or resource in lower:
                found.append(resource)
        if not found:
            # Infer from AWS keywords
            if "s3" in lower or "bucket" in lower:
                found.append("s3_bucket")
            elif "iam" in lower or "role" in lower:
                found.append("iam_role")
            else:
                found.append("aws_resource")

        subtasks = []
        for i, r in enumerate(found):
            deps = found[:i]
            subtasks.append(SubTask(
                resource_type=r,
                prompt=f"Generate the Terraform {r} resource for:\n{intent}",
                depends_on=deps,
            ))
        return subtasks

    # ── Difficulty estimation ─────────────────────────────────────────────────

    def _estimate_difficulty(self, lower: str, subtasks: List[SubTask]) -> DifficultyLevel:
        score = 1

        # Base: number of resources
        n = len(subtasks)
        if n >= 4:
            score += 2
        elif n >= 2:
            score += 1

        # Security context keywords → +1
        if any(kw in lower for kw in _DIFFICULTY_SECURITY_KEYWORDS):
            score += 1

        # Compliance / encryption / admission keywords → +1
        if any(kw in lower for kw in _DIFFICULTY_COMPLIANCE_KEYWORDS):
            score += 1

        # Cross-resource stateful patterns → +1
        cross_resource = {"networkpolicy", "statefulset", "persistentvolumeclaim",
                          "cronjob", "rbac", "rolebinding", "clusterrole"}
        resource_types = {st.resource_type.lower() for st in subtasks}
        if resource_types & cross_resource:
            score += 1

        return DifficultyLevel(min(score, 5))
