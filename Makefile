.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	python app.py

test:
	python -m pytest tests/ -v

download-model:
	python scripts/download_model.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f evaluations.db
