# Dreceres de l'equip. `make help` per veure-les totes.
PY ?= python

.PHONY: help setup setup-dl data baselines test lint app clean

help:  ## Mostra aquesta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Instal·la dependències i el paquet dreamteam (dins del .venv actiu)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

setup-dl:  ## Dependències de deep learning i descàrrega de CLIP/MiniLM a la caché
	$(PY) -m pip install -r requirements-dl.txt
	$(PY) -m dreamteam.embeddings

data:  ## Baixa les dades públiques dels reptes anteriors a data/past/
	bash scripts/download_past_data.sh

baselines:  ## Executa tots els baselines (comprova que l'entorn funciona de punta a punta)
	$(PY) reptes/2025_schneider_explicabilitat/baseline.py
	$(PY) reptes/2022_accenture_supply_chain/baseline.py
	$(PY) reptes/2024_mango_atributs/baseline.py
	$(PY) reptes/2021_reviews_nlp/baseline.py
	$(PY) reptes/2025_mango_demanda/simulate_data.py
	$(PY) reptes/2025_mango_demanda/pipeline.py
	$(PY) reptes/2022_qualcomm_power_switches/solver.py

test:  ## Tests de la llibreria
	$(PY) -m pytest -q

lint:  ## Revisió d'estil amb ruff
	ruff check src tests reptes app

app:  ## Demo de Streamlit
	streamlit run app/streamlit_demo.py

clean:  ## Esborra sortides generades (submissions, caché, __pycache__)
	rm -rf submissions cache catboost_info .pytest_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
