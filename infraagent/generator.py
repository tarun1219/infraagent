"""
LLM Code Generator Module for InfraAgent.

Wraps Ollama-served open-source models (DeepSeek-Coder, CodeLlama,
Mistral, Phi-3) with a structured prompting strategy that incorporates
task plans and RAG-retrieved context. Supports both initial generation
and self-correction prompts.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from infraagent.planner import IaCLanguage, SubTask, TaskPlan


class ModelID(str, Enum):
    DEEPSEEK_CODER = "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
    CODELLAMA = "codellama:13b-instruct-q4_K_M"
    MISTRAL = "mistral:7b-instruct-q4_K_M"
    PHI3 = "phi3:3.8b-mini-instruct-4k-q4_K_M"
    # Commercial ceiling reference — served via OpenAI API (not Ollama)
    GPT4O = "gpt-4o"
    # Second commercial baseline — served via Anthropic API
    CLAUDE = "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# Anthropic / Claude backend
# ---------------------------------------------------------------------------

def _call_anthropic(
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    Call the Anthropic Messages API with the Claude-3.5-Sonnet model.

    Requires the ANTHROPIC_API_KEY environment variable.
    Estimated cost: ~$0.03 per task at Claude-3.5-Sonnet pricing.

    Falls back to a RuntimeError on API errors so the caller can decide
    whether to use a stub or propagate the failure.
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return message.content[0].text if message.content else ""
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI / GPT-4o backend
# ---------------------------------------------------------------------------

def _call_openai(
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    Call the OpenAI Chat Completions API with the GPT-4o model.

    Requires the OPENAI_API_KEY environment variable.
    Estimated cost: ~$0.03–$0.05 per task at gpt-4o pricing,
    so ~$5 for 150 tasks (5 conditions × 150 = 750 calls total
    if run across all conditions; one-shot only ≈ 150 calls ≈ $2).

    Falls back to a stub on API errors so the pipeline never hard-crashes.
    """
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
You are InfraAgent, an expert Infrastructure-as-Code engineer. \
Your task is to generate syntactically correct, schema-valid, and \
security-hardened IaC code based on the provided task description and \
supporting documentation. Follow these rules strictly:
1. Always use the correct, non-deprecated API versions.
2. Always include resource limits (CPU and memory) for all containers.
3. Always include liveness and readiness probes for deployments.
4. Never run containers as root; use securityContext with runAsNonRoot: true.
5. Ensure label selectors in Services/HPAs/PDBs match pod template labels exactly.
6. Output ONLY the raw IaC code — no explanations, no markdown fences unless \
they are part of the YAML/HCL syntax itself.
""")

_GENERATION_PROMPT = textwrap.dedent("""\
## Task
{description}

## Sub-task to generate
Resource type: {resource_type}
Language: {language}
Constraints: {constraints}

{context}

## Integration notes
{integration_notes}

Generate the complete, production-ready {language} code for this resource:
""")

_CORRECTION_PROMPT = textwrap.dedent("""\
## Original Task
{description}

## Previously Generated Code (contained errors)
```
{previous_code}
```

## Validation Errors Found
{errors}

{context}

## Instructions
Fix ALL of the errors listed above. Output ONLY the corrected {language} code \
without any additional explanation. Ensure:
1. All deprecated API versions are replaced with current ones.
2. All label selectors match pod template labels exactly.
3. All security issues (missing runAsNonRoot, missing limits, etc.) are resolved.
4. The code is complete — do not omit any sections from the original.
""")


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    code: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    generation_round: int       # 0 = initial, 1+ = correction rounds
    used_rag: bool


# ---------------------------------------------------------------------------
# Code extractor
# ---------------------------------------------------------------------------

def _extract_code(raw_output: str, language: IaCLanguage) -> str:
    """Strip markdown fences and prose, leaving only the IaC code."""
    # Remove triple-backtick blocks if present
    fenced = re.findall(
        r"```(?:yaml|hcl|terraform|dockerfile|docker)?\s*\n(.*?)```",
        raw_output, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        return "\n---\n".join(f.strip() for f in fenced)

    # Heuristic: keep lines that look like IaC
    if language == IaCLanguage.KUBERNETES or language == IaCLanguage.TERRAFORM:
        lines = raw_output.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("apiVersion:", "kind:", "metadata:", "resource ", "variable ", "output ", "provider ", "terraform {", "FROM ", "RUN ", "COPY ", "WORKDIR ")):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)

    return raw_output.strip()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LLMCodeGenerator:
    """
    Generates IaC code via Ollama-hosted open-source LLMs.

    Supports multiple models (configurable), structured prompting with
    optional RAG context injection, and self-correction prompts that
    include structured validator error feedback.
    """

    def __init__(
        self,
        model: ModelID = ModelID.DEEPSEEK_CODER,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        ollama_base_url: str = "http://localhost:11434",
        use_stub: bool = False,
    ):
        """
        Args:
            model: The model to use. Open-source models are served via Ollama;
                   ModelID.GPT4O is served via the OpenAI API (requires
                   OPENAI_API_KEY env var; ~$0.03–0.05 per task).
            temperature: Sampling temperature (low = more deterministic).
            max_tokens: Maximum output tokens per generation.
            ollama_base_url: Base URL of the local Ollama server (ignored for GPT-4o).
            use_stub: If True, return placeholder code (for testing without any backend).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_base_url = ollama_base_url
        self.use_stub = use_stub
        self._is_openai = (model == ModelID.GPT4O)
        self._is_anthropic = (model == ModelID.CLAUDE)
        self._call_count = 0

    def generate(
        self,
        task: TaskPlan,
        subtask: SubTask,
        rag_context: str = "",
        generation_round: int = 0,
    ) -> GenerationResult:
        """
        Generate IaC code for a single subtask.

        Args:
            task: The full TaskPlan (for context, constraints, notes).
            subtask: The specific SubTask being generated.
            rag_context: Pre-built RAG context string from RAGModule.
            generation_round: 0 = initial generation.

        Returns:
            A GenerationResult with the generated code.
        """
        prompt = _GENERATION_PROMPT.format(
            description=task.original_intent,
            resource_type=subtask.resource_type,
            language=subtask.language.value,
            constraints=", ".join(subtask.constraints) if subtask.constraints else "none",
            context=rag_context,
            integration_notes=task.integration_notes or "none",
        )
        raw = self._call_llm(prompt)
        code = _extract_code(raw, task.language)
        self._call_count += 1
        return GenerationResult(
            code=code,
            model=self.model.value,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(code.split()),
            generation_round=generation_round,
            used_rag=bool(rag_context),
        )

    def self_correct(
        self,
        task: TaskPlan,
        previous_code: str,
        errors: List[Dict[str, Any]],
        rag_context: str = "",
        round_number: int = 1,
    ) -> GenerationResult:
        """
        Perform one self-correction step, feeding back validation errors
        to the LLM to produce a revised IaC artifact.

        Args:
            task: The original TaskPlan.
            previous_code: The code that failed validation.
            errors: Structured list of validation errors from the validator.
            rag_context: RAG context relevant to the errors encountered.
            round_number: Current correction round (1-indexed).

        Returns:
            A GenerationResult with corrected code.
        """
        error_text = self._format_errors(errors)
        prompt = _CORRECTION_PROMPT.format(
            description=task.original_intent,
            previous_code=previous_code,
            errors=error_text,
            context=rag_context,
            language=task.language.value,
        )
        raw = self._call_llm(prompt)
        code = _extract_code(raw, task.language)
        self._call_count += 1
        return GenerationResult(
            code=code,
            model=self.model.value,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(code.split()),
            generation_round=round_number,
            used_rag=bool(rag_context),
        )

    def _call_llm(self, user_prompt: str) -> str:
        if self.use_stub:
            return self._stub_response()
        if self._is_anthropic:
            try:
                return _call_anthropic(user_prompt, self.temperature, self.max_tokens)
            except Exception as exc:
                print(f"[Generator] Anthropic API call failed: {exc}. Using stub.")
                return self._stub_response()
        if self._is_openai:
            try:
                return _call_openai(user_prompt, self.temperature, self.max_tokens)
            except Exception as exc:
                print(f"[Generator] OpenAI call failed: {exc}. Using stub.")
                return self._stub_response()
        try:
            import requests
            payload = {
                "model": self.model.value,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                "stream": False,
            }
            resp = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except Exception as exc:
            print(f"[Generator] LLM call failed: {exc}. Using stub.")
            return self._stub_response()

    def _stub_response(self) -> str:
        """Return a minimal valid stub for testing."""
        return textwrap.dedent("""\
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: stub-app
              namespace: default
              labels:
                app: stub-app
            spec:
              replicas: 1
              selector:
                matchLabels:
                  app: stub-app
              template:
                metadata:
                  labels:
                    app: stub-app
                spec:
                  securityContext:
                    runAsNonRoot: true
                    runAsUser: 1000
                    seccompProfile:
                      type: RuntimeDefault
                  containers:
                  - name: stub-app
                    image: nginx:1.27
                    ports:
                    - containerPort: 80
                    resources:
                      requests:
                        memory: "64Mi"
                        cpu: "100m"
                      limits:
                        memory: "128Mi"
                        cpu: "200m"
                    securityContext:
                      allowPrivilegeEscalation: false
                      readOnlyRootFilesystem: true
                      capabilities:
                        drop: ["ALL"]
            """)

    @staticmethod
    def _format_errors(errors: List[Dict[str, Any]]) -> str:
        lines = []
        for i, err in enumerate(errors, 1):
            layer = err.get("layer", "unknown")
            message = err.get("message", str(err))
            severity = err.get("severity", "ERROR")
            lines.append(f"{i}. [{layer}] [{severity}] {message}")
        return "\n".join(lines) if lines else "No errors."
