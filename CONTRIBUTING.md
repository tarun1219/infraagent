# Contributing to InfraAgent

## Reporting Issues

- Use GitHub Issues
- Include: OS, Python version, model name, task ID, full error message

## Adding Benchmark Tasks

Tasks live in `iachench/tasks/`. Format specification: [iachench/tasks/README.md](iachench/tasks/README.md).

New tasks require:
- `prompt`: natural language description
- `expected_resources`: list of resource types
- `validation_criteria`: threshold-based pass/fail rules
- `expected_output`: reference correct configuration
- `failure_modes`: common mistakes for this task type

All new tasks must pass `make test` before submission.

## Pull Requests

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-change`
3. Run tests: `make test`
4. Run lint: `make lint`
5. Submit PR with a description of the change and motivation

## Code Style

- **Formatting**: Black (`make format`)
- **Type hints**: required for all public functions
- **Docstrings**: required for all modules and public classes
- **Line length**: 100 characters max

## Adding a New Model

1. Add model ID to `MODELS` dict in `scripts/run_experiments.py`
2. Add display name to `MODEL_LABELS` in `scripts/generate_figures.py`
3. Add routing in `infraagent/generator.py`
4. Update `requirements.txt` if new SDK needed
