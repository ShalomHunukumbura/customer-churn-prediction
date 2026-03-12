VENV=.venv
ACT=. $(VENV)/bin/activate

setup:
	python3 -m venv $(VENV)
	$(ACT) && pip install -r requirements.txt

data:
	$(ACT) && python -m src.data.generate_dataset
	$(ACT) && python -m src.eda

train:
	$(ACT) && python -m src.train

test:
	$(ACT) && pytest -q

serve:
	$(ACT) && uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

all: data train test
