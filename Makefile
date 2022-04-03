
PYTHON ?= python3
POETRY ?= poetry


.PHONY: all
all: format_check type_check test


.PHONY: test
test:
	$(PYTHON) -m unittest discover -s tests


.PHONY: format_check
format_check:
	$(POETRY) run black --check src tests


.PHONY: type_check
type_check:
	$(POETRY) run mypy --strict src


.PHONY: doc
doc:
	sphinx-apidoc -f -o docs/source src
	$(MAKE) html -C docs


.PHONY: clean
clean:
	$(RM) -r docs/build

