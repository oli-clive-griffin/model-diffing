.PHONY: install
install:
	uv pip install -e .

.PHONY: install-dev
install-dev:
	uv pip install -e .[dev]
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

.PHONY: setup-machine
setup-machine:
	pip install uv && \
	uv venv --python 3.11 && \
	. .venv/bin/activate && \
	uv pip install -e .
	echo "setup complete, now run 'source .venv/bin/activate'"