.PHONY: install lint test eval models api console up down

install:   ## Install dev dependencies (CPU PyTorch)
	pip install -r requirements-dev.txt
lint:
	ruff check .
test:
	pytest -q
eval:      ## Retrain all models from scratch and apply the quality gate (~12 min on 2 CPU cores)
	python -m indl.eval.run --gate
models:    ## Same, and write the production weights to models/
	python -m indl.eval.run --gate --save
api:
	uvicorn indl.api:app --reload
console:
	streamlit run streamlit_app.py
up:
	cp -n .env.example .env || true; docker compose up --build -d
down:
	docker compose down
