.PHONY: install test lint format reproduce quickstart clean help

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	flake8 infraagent/ iachench/ scripts/ --max-line-length=100
	black --check infraagent/ iachench/ scripts/

format:
	black infraagent/ iachench/ scripts/

reproduce:
	bash scripts/reproduce_paper.sh

quickstart:
	python scripts/run_experiments.py --condition sc+rag+3r --model deepseek --num_tasks 10

figures:
	python scripts/plot_figures.py --output paper_figures/

stats:
	python scripts/compute_statistics.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

help:
	@echo "InfraAgent — available make targets:"
	@echo "  make install    Install Python dependencies"
	@echo "  make test       Run unit tests"
	@echo "  make lint       Check code style"
	@echo "  make format     Auto-format code with black"
	@echo "  make reproduce  Reproduce all paper results (~12 GPU hours)"
	@echo "  make quickstart Run 10 sample tasks (quick sanity check)"
	@echo "  make figures    Regenerate all 14 paper figures"
	@echo "  make stats      Compute statistical significance tables"
	@echo "  make clean      Remove compiled Python files"
