"""
InfraAgent: Main Agent Loop.

Orchestrates the full pipeline:
  1. Task Planner decomposes user intent
  2. RAG Module retrieves relevant documentation
  3. LLM Generator produces initial IaC code
  4. Multi-Layer Validator checks correctness & security
  5. Self-Correction Loop (up to max_rounds) feeds errors
     back to the LLM until all validations pass or rounds exhaust

Returns a structured AgentResult capturing all intermediate states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from infraagent.generator import GenerationResult, LLMCodeGenerator, ModelID
from infraagent.planner import IaCLanguage, TaskPlan, TaskPlanner
from infraagent.rag_module import RAGModule
from infraagent.validator import MultiLayerValidator, ValidationReport


@dataclass
class RoundRecord:
    """Records the state at each generation/correction round."""
    round: int          # 0 = initial generation
    code: str
    report: ValidationReport
    duration_s: float


@dataclass
class AgentResult:
    """Full result of one InfraAgent run on a single task."""
    task_id: str
    task_intent: str
    language: str
    difficulty: int
    model: str
    used_rag: bool
    max_rounds: int

    rounds: List[RoundRecord] = field(default_factory=list)
    final_code: str = ""
    final_report: Optional[ValidationReport] = None
    success: bool = False
    total_rounds_used: int = 0
    total_duration_s: float = 0.0

    # Convenience accessors
    @property
    def syntax_valid(self) -> bool:
        return self.final_report.syntax_valid if self.final_report else False

    @property
    def schema_valid(self) -> bool:
        return self.final_report.schema_valid if self.final_report else False

    @property
    def security_score(self) -> float:
        return self.final_report.security_score if self.final_report else 0.0

    @property
    def best_practice_score(self) -> float:
        return self.final_report.best_practice_score if self.final_report else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_intent": self.task_intent,
            "language": self.language,
            "difficulty": self.difficulty,
            "model": self.model,
            "used_rag": self.used_rag,
            "max_rounds": self.max_rounds,
            "success": self.success,
            "total_rounds_used": self.total_rounds_used,
            "total_duration_s": self.total_duration_s,
            "syntax_valid": self.syntax_valid,
            "schema_valid": self.schema_valid,
            "dry_run_server_valid": self.final_report.dry_run_server_valid if self.final_report else None,
            "dry_run_server_errors": self.final_report.dry_run_server_errors if self.final_report else [],
            "security_score": self.security_score,
            "best_practice_score": self.best_practice_score,
            "overall_score": self.final_report.overall_score if self.final_report else 0.0,
            "rounds": [
                {
                    "round": r.round,
                    "passed": r.report.passed,
                    "syntax_valid": r.report.syntax_valid,
                    "schema_valid": r.report.schema_valid,
                    "security_score": r.report.security_score,
                    "best_practice_score": r.report.best_practice_score,
                    "overall_score": r.report.overall_score,
                    "n_errors": len(r.report.errors),
                    "duration_s": r.duration_s,
                }
                for r in self.rounds
            ],
        }


class InfraAgent:
    """
    End-to-end InfraAgent pipeline.

    Args:
        model:        The Ollama LLM model to use.
        max_rounds:   Maximum self-correction iterations (0 = one-shot only).
        use_rag:      Whether to augment prompts with RAG retrieval.
        rag_top_k:    Number of RAG passages to retrieve per query.
        use_stub:     Run in stub mode (no Ollama required).
        verbose:      Print progress to stdout.
    """

    def __init__(
        self,
        model: ModelID = ModelID.DEEPSEEK_CODER,
        max_rounds: int = 3,
        use_rag: bool = True,
        rag_top_k: int = 5,
        rag_persist_dir: str = "./rag_corpus",
        use_stub: bool = False,
        verbose: bool = True,
    ):
        self.planner = TaskPlanner()
        self.rag = RAGModule(
            persist_dir=rag_persist_dir,
            top_k=rag_top_k,
            use_stub=True,  # always use stub for RAG in offline mode
        )
        self.generator = LLMCodeGenerator(
            model=model,
            use_stub=use_stub,
        )
        self.validator = MultiLayerValidator()
        self.max_rounds = max_rounds
        self.use_rag = use_rag
        self.verbose = verbose
        self._model_name = model.value

    def run(self, intent: str, task_id: Optional[str] = None) -> AgentResult:
        """
        Execute the full InfraAgent pipeline on a natural language intent.

        Args:
            intent:   Natural language infrastructure request.
            task_id:  Optional task identifier (auto-generated if None).

        Returns:
            AgentResult with final code, validation report, and round history.
        """
        start_all = time.perf_counter()

        # --- Step 1: Plan ---
        plan: TaskPlan = self.planner.plan(intent, task_id=task_id)
        if self.verbose:
            print(f"[InfraAgent] Task: {plan.task_id} | "
                  f"Lang: {plan.language.value} | "
                  f"Difficulty: L{plan.difficulty.value} | "
                  f"Subtasks: {len(plan.subtasks)}")

        result = AgentResult(
            task_id=plan.task_id,
            task_intent=intent,
            language=plan.language.value,
            difficulty=plan.difficulty.value,
            model=self._model_name,
            used_rag=self.use_rag,
            max_rounds=self.max_rounds,
        )

        # --- Step 2: Generate all subtasks ---
        rag_context = ""
        if self.use_rag:
            rag_context = self.rag.build_context_string(
                query=intent + " " + " ".join(
                    st.resource_type for st in plan.subtasks
                ),
                language_filter=plan.language.value,
            )

        code_parts: List[str] = []
        for subtask in plan.subtasks:
            t0 = time.perf_counter()
            gen: GenerationResult = self.generator.generate(
                task=plan,
                subtask=subtask,
                rag_context=rag_context,
                generation_round=0,
            )
            code_parts.append(gen.code)

        combined_code = "\n---\n".join(code_parts)

        # --- Step 3: Validate initial generation ---
        t0 = time.perf_counter()
        report = self.validator.validate(combined_code, plan.language.value)
        duration = time.perf_counter() - t0

        result.rounds.append(RoundRecord(
            round=0,
            code=combined_code,
            report=report,
            duration_s=duration,
        ))

        if self.verbose:
            self._log_round(0, report)

        # --- Step 4: Self-correction loop ---
        current_code = combined_code
        for rnd in range(1, self.max_rounds + 1):
            if report.passed:
                break

            # Retrieve error-targeted RAG context
            error_query = " ".join(e.message for e in report.errors[:5])
            if self.use_rag and error_query:
                rag_context = self.rag.build_context_string(
                    query=error_query,
                    language_filter=plan.language.value,
                )

            feedback = self.validator.errors_to_feedback(report)

            t0 = time.perf_counter()
            gen = self.generator.self_correct(
                task=plan,
                previous_code=current_code,
                errors=feedback,
                rag_context=rag_context,
                round_number=rnd,
            )
            corrected_code = gen.code

            report = self.validator.validate(corrected_code, plan.language.value)
            duration = time.perf_counter() - t0

            result.rounds.append(RoundRecord(
                round=rnd,
                code=corrected_code,
                report=report,
                duration_s=duration,
            ))
            current_code = corrected_code

            if self.verbose:
                self._log_round(rnd, report)

            if report.passed:
                break

        # --- Finalize ---
        result.final_code = current_code
        result.final_report = report
        result.success = report.passed
        result.total_rounds_used = len(result.rounds) - 1  # correction rounds
        result.total_duration_s = time.perf_counter() - start_all

        if self.verbose:
            status = "PASSED" if result.success else "FAILED"
            print(f"[InfraAgent] {status} after {result.total_rounds_used} "
                  f"correction round(s) | "
                  f"{result.total_duration_s:.1f}s total")

        return result

    def _log_round(self, rnd: int, report: ValidationReport):
        status = "PASS" if report.passed else "FAIL"
        print(
            f"  Round {rnd}: [{status}] "
            f"syntax={report.syntax_valid} "
            f"schema={report.schema_valid} "
            f"security={report.security_score:.2f} "
            f"bp={report.best_practice_score:.2f} "
            f"errors={len(report.errors)}"
        )
