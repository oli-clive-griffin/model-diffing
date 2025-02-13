.PHONY: install
install:
	pip install -e .

.PHONY: install-dev
install-dev:
	pip install -e .[dev]
	pre-commit install

.PHONY: type
type:
	SKIP=no-commit-to-branch pre-commit run -a pyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	SKIP=no-commit-to-branch pre-commit run -a ruff-lint
	SKIP=no-commit-to-branch pre-commit run -a ruff-format

.PHONY: check
check:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

.PHONY: test
test:
	python -m pytest

.PHONY: test-all
test-all:
	python -m pytest --runslow

.PHONY: sync-up
sync-up:
	rsync -avzP \
		--filter=':- .gitignore' \
		--exclude='.git/' \
		. $(LOCATION)

.PHONY: sync-down
sync-down:
	rsync -avzP \
		--filter=':- .gitignore' \
		--exclude='.git/' \
		$(LOCATION) .

# Default values
LOCATION ?= user@10.10.10.10:/path/to/destination

.PHONY: setup-vast
setup-vast: # this is so shit but it works for now. Tired of trying to get uv, pyproject.toml, and pip to work together
	pip install transformer_lens pydantic wandb fire tqdm einops transformers datasets plotly pandas matplotlib