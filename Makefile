
PYTHON ?= python3
POETRY ?= poetry


.PHONY: all
all: format_check type_check lint_check test


.PHONY: test
test:
	$(PYTHON) -m unittest discover -s tests


.PHONY: format_check
format_check:
	$(POETRY) run yapf --recursive --diff basic_iter tests


.PHONY: type_check
type_check:
	$(POETRY) run mypy --strict basic_iter


.PHONY: lint_check
lint_check:
	$(POETRY) run pylint --recursive=yes basic_iter tests


.PHONY: doc
doc:
	sphinx-apidoc -f -o docs/source basic_iter
	$(MAKE) html -C docs


.PHONY: format
format:
	$(POETRY) run yapf --recursive --in-place basic_iter tests


.PHONY: clean
clean:
	$(RM) -r docs/build

