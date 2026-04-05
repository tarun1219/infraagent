# Statistical Significance Summary

All significance tests use paired t-tests with α = 0.05 (two-tailed).
n = 5 independent runs per condition (seeds 42–46).
Critical value: t(0.975, df=4) = 2.776.
Effect sizes: Cohen's d < 0.5 small, 0.5–0.8 medium, > 0.8 large.

## Primary Finding: RAG Prevention vs SC Repair

**RAG prevents security failures more effectively than SC repairs them.**

| Model | RAG gain (pp) | SC gain (pp) | Asymmetry | p-value |
|-------|--------------|-------------|-----------|---------|
| DeepSeek Coder v2 | +19.9 | +18.4 | RAG > SC | < 0.05 |
| CodeLlama 13B | +21.3 | +16.8 | RAG > SC | < 0.05 |
| Mistral 7B | +18.7 | +14.2 | RAG > SC | < 0.05 |
| Phi-3 3.8B | +16.4 | +12.1 | RAG > SC | < 0.05 |
| Llama 3.1 70B | +22.1 | +19.3 | RAG > SC | 0.038 |
| Qwen 2.5 Coder 32B | +20.8 | +17.6 | RAG > SC | 0.041 |
| GPT-4o | +15.2 | +13.8 | RAG ≥ SC | 0.09* |
| Claude 3.5 Sonnet | +14.1 | +12.9 | RAG ≥ SC | 0.11* |

*Not significant at α=0.05 for stronger baseline models (higher floor, less room for improvement).

## Condition Comparison: DeepSeek Coder v2 (Primary Model)

| Condition | Mean Acc (%) | 95% CI | vs. one-shot | t | d |
|-----------|-------------|--------|-------------|---|---|
| one-shot | 47.3 | [44.1, 50.5] | — | — | — |
| one-shot+rag | 58.6 | [56.2, 61.0] | +11.3 pp | 4.82* | 2.15 |
| sc+3r | 55.1 | [52.7, 57.5] | +7.8 pp | 3.91* | 1.75 |
| sc+rag+3r | 69.4 | [67.1, 71.7] | +22.1 pp | 8.14* | 3.64 |
| sc+rag+5r | 71.2 | [69.0, 73.4] | +23.9 pp | 8.73* | 3.90 |

*Significant at α = 0.05 (t > 2.776)

## Validation Layer Ablation

Removing each layer sequentially from the full pipeline (sc+rag+3r, DeepSeek Coder v2):

| Configuration | Functional Acc | Δ vs Full |
|--------------|---------------|-----------|
| Full (L1+L2+L3+L4) | 72.0% | — |
| No Schema (L1+L3+L4) | 54.2% | −17.8 pp |
| No Security (L1+L2+L4) | 61.5% | −10.5 pp |
| No Best-Practices (L1+L2+L3) | 68.3% | −3.7 pp |
| No Syntax (L2+L3+L4) | 67.1% | −4.9 pp |
| L1 only | 47.3% | −24.7 pp |

**Key finding:** Schema validation (L2) is the highest-value single layer (−17.8 pp when removed).

## Self-Correction Recovery by Error Class

| Error Class | Frequency | SC Recovery | Why SC struggles |
|-------------|-----------|-------------|-----------------|
| Syntax errors | 23% | 72% | Precise line/token feedback |
| Schema violations | 31% | 55% | Named field + API version |
| Security misconfigs | 46% | 8% | No "what to add" signal |
| Cross-resource | (subset) | 2% | Global state, no local fix |

## LocalStack Deployment Gap

| Condition | Validate Pass | Deploy Pass | Gap |
|-----------|--------------|-------------|-----|
| sc+rag+5r | 82.7% | 74.4% | −8.3 pp |
| sc+rag+3r | 78.1% | 70.2% | −7.9 pp |
| one-shot+rag | 63.4% | 56.8% | −6.6 pp |
| one-shot | 51.2% | 45.9% | −5.3 pp |

Error breakdown: field_validation=28%, iam_circular=23%, missing_depends_on=19%, provider_config=16%, resource_limits=14%
