---
name: Bug report
about: Report a bug in IaCBench, InfraAgent, or the evaluation scripts
title: "[BUG] "
labels: bug
assignees: ''
---

## Bug Description

A clear description of what the bug is.

## Reproduction Steps

```bash
# Minimal command to reproduce
python scripts/run_experiments.py --condition sc+rag+3r --model deepseek --runs 1
```

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include the full error traceback if applicable.

```
Paste error output here
```

## Environment

- OS: [e.g. Ubuntu 22.04, macOS 14.2]
- Python version: [e.g. 3.11.5]
- InfraAgent version / commit: [e.g. `git rev-parse HEAD`]
- Ollama version (if applicable): [e.g. 0.1.27]
- Model (if applicable): [e.g. deepseek-coder-v2:16b-lite-instruct-q4_K_M]

## Component

- [ ] `iachench/` — benchmark tasks or validators
- [ ] `infraagent/` — agent or RAG pipeline
- [ ] `scripts/` — evaluation or figure scripts
- [ ] Paper figures / statistics
- [ ] Documentation

## Additional Context

Any other context about the problem here (e.g. related issues, PRs, papers).
