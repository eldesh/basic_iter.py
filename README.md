# Basic Iterator Operators for Python

This package provides basic functions on iterators.


## Generate documents

To generate documents:

```sh
basic_iter$ make doc
```

This will generate documentation under `./docs/build/html`.


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


