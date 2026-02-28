.PHONY: help setup test lint format ingest evaluate demo qdrant qdrant-stop clean

# Default target
help:
	@echo ""
	@echo "  RAG Financial Docs — available commands"
	@echo ""
	@echo "  make setup       Install all dependencies (including dev)"
	@echo "  make qdrant      Start Qdrant vector DB via Docker"
	@echo "  make qdrant-stop Stop Qdrant"
	@echo "  make data        Generate synthetic documents"
	@echo "  make ingest      Run the full ingestion pipeline"
	@echo "  make evaluate    Run the evaluation suite"
	@echo "  make demo        Launch Streamlit UI"
	@echo "  make test        Run test suite"
	@echo "  make lint        Check code with ruff"
	@echo "  make format      Auto-format code with ruff"
	@echo "  make clean       Remove generated artifacts"
	@echo ""

setup:
	pip install -e ".[dev]"
	pre-commit install

qdrant:
	docker-compose up -d qdrant
	@echo "Qdrant running at http://localhost:6333"
	@echo "Dashboard: http://localhost:6333/dashboard"

qdrant-stop:
	docker-compose stop qdrant

data:
	python scripts/generate_synthetic_data.py

ingest:
	python scripts/ingest.py

evaluate:
	python scripts/evaluate.py

demo:
	streamlit run src/app/streamlit_app.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf data/processed/*
