# Contributing to torchjd

This document explains how to contribute to torchjd. We try to keep the quality of the codebase as
high as possible. Even if this slows down development in the short term, it helps a lot in the long
term. To make the code understandable by everyone, we try to keep it simple, and to stick as much as
possible to the single-responsibility principle. Please use issues to communicate with maintainers
before implementing major changes to torchjd.

## Installation

1) Pre-requisites: To work with torchjd, you need python to be installed. In the following, we
   suggest to use python 3.12.3, but you can work with any python version supported by torchjd. We
   use [pyenv](https://github.com/pyenv/pyenv) to install python and
   [pdm](https://pdm-project.org/en/latest/) to manage dependencies. While the desired python
   version can also be installed without pyenv, the installation of torchjd for development purposes
   requires pdm. To install it, follow their
   [installation steps](https://pdm-project.org/en/latest/#installation).

2) Create a virtual environment and install the project in it. From the root of torchjd, run:
   ```bash
   pdm venv create 3.12.3  # Requires python 3.12.3 to be installed
   pdm use .venv/bin/python
   pdm install --frozen-lockfile
   pdm run pre-commit install
   ```

> [!TIP]
> The python version that you should specify in your IDE is `<path-to-torchjd>/.venv/bin/python`.

## Running tests
   - To verify that your installation was successful, and that all unit tests pass, run:
     ```bash
     pdm run pytest tests/unit
     ```

   - If you have access to a cuda-enabled GPU, you should also check that the unit tests pass on it:
     ```bash
     CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTEST_TORCH_DEVICE=cuda pdm run pytest tests/unit
     ```

   - To check that the usage examples from docstrings and `.rst` files are correct, we test their
   behavior in `tests/doc`. To run these tests, do:
     ```bash
     pdm run pytest tests/doc
     ```

## Building the documentation locally
   - From the `docs` folder, run:
     ```bash
     pdm run make html
     ```
   - You can then open `docs/build/html/index.html` with a web browser.
   - Sometimes, you need to manually delete the built documentation before generating it. To do
   this, from the `docs` folder, run:
     ```bash
     pdm run make clean
     ```

## Development guidelines

Most source python files in torchjd have a corresponding `.rst` in `docs/source`. Please make sure
to add such a documentation entry whenever you add a new public module. In most cases, public
classes should contain a usage example in their docstring.
We ask contributors to implement the unit
tests necessary to check the correctness of their implementations. Besides, whenever usage examples
are provided, we require the example's code to be tested in `tests/doc`.
Lastly, make sure that new modules are imported by the `__init__.py` file of the package they are
located into. This makes them easier to import for the user.
