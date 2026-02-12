setup: .venv pre-commit-install

update_venv:
	. .venv/bin/activate
	uv pip freeze > requirements.txt

pre-commit-check:
	. .venv/bin/activate; pre-commit run --all-files

pre-commit-install: .pre-commit-config.yaml
	@echo Installing pre-commit hooks
	. .venv/bin/activate; pre-commit install; pre-commit autoupdate
	make pre-commit-check

.venv: .venv/touchfile

.venv/touchfile: requirements.txt
	@echo Creating virtual environment at .venv
	test -d .venv || uv venv
	. .venv/bin/activate; uv pip install -Ur requirements.txt
	touch .venv/touchfile
