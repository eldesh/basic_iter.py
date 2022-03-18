# Basic Iterator Operators for Python

This package provides basic functions on iterators.


## Generate documents

```sh
basic_iter$ sphinx-apidoc -o docs/source src
basic_iter$ make html -C docs
```


## Format Checking

For format checking by black:

```sh
basic_iter$ poetry run black --check src tests
```


## Type Checking

For type checking by mypy:

```sh
basic_iter$ poetry run mypy --strict src
```


## Unit Test

For executing unit tests:

```sh
basic_iter$ python3 -m unittest discover -s tests
```


