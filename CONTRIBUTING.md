# Contributing to torchjd

The following steps outline how to contribute to torchjd.
We worked on recent versions of Ubuntu while developing torchjd. The commands proposed in this
section should be adapted if you use another operating system.

## Development guidelines
Please use issues to communicate with maintainers before implementing major changes to torchjd.
We try to keep the quality of the codebase as high as possible. Even if this slows down development
in the short term, it helps a lot in the long term. To make the code understandable by everyone, we
try to keep it simple, and to stick as much as possible to the single-responsibility principle.

Most source python files in torchjd have a corresponding `.rst` in `docs/source`. Please make sure
to add such a documentation entry whenever you add a new public module. In most cases, public
classes should contain a usage example in their docstring.
We ask contributors to implement the unit
tests necessary to check the correctness of their implementations. Besides, whenever usage examples
are provided, we require the example's code to be tested in `tests/doc`.
Lastly, make sure that new modules are imported by the `__init__.py` file of the package they are
located into. This makes them easier to import.

## Installation
1) Pre-requisites: We use [pyenv](https://github.com/pyenv/pyenv) to manage python versions and
   [pdm](https://pdm-project.org/en/latest/) to manage dependencies. While the desired python
   version can be installed manually rather than using pyenv, the installation of torchjd for
   development purposes requires pdm.
   - Install pyenv by following [their instructions](https://github.com/pyenv/pyenv#installation)
   - Install libraries that are required to install python with pyenv:
     ```bash
     sudo apt-get install build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev
     ```
   - Install python 3.12.4 (or another supported version) using pyenv:
      ```bash
      pyenv install 3.12.4
      ```
   - Download and install pdm:
     ```bash
     curl -sSL https://pdm.fming.dev/install-pdm.py | python3 -
     ```
   - Make pdm accessible from your `PATH`. If not already present, add the following line to your
   `.bashrc` file:
     ```bash
     export PATH="$PATH:$HOME/.local/bin"
     ```

2) Create a virtual environment and install the project in it. From the root of torchjd, run:
   ```bash
   pdm venv create 3.12.4
   pdm venv use .venv/bin/python
   pdm install --frozen-lockfile
   ```

3) Install pre-commit hooks, from the root of torchjd.
   ```bash
   pdm run pre-commit install
   ```

> [!TIP]
> The python version that you should specify in your IDE is `<path-to-torchjd>/.venv/bin/python`.

## Running tests
   - To verify that your installation was successful, and that alls unit tests pass, run:
     ```bash
     pdm run pytest tests/unit
     ```

   - To check that the usage examples from docstrings and `.rst` files are correct, we test their
   behavior in `tests/doc`. To run these tests, use:
     ```bash
     pdm run pytest tests/doc
     ```

## Building the documentation locally
   - From the `docs` folder, run:
     ```bash
     pdm run make html
     ```
   - You can then open with a web browser `docs/build/html/index.html`
   - Sometimes, you need to manually delete the built documentation before regenerating it. To do
   this, from the `docs` folder, run:
     ```bash
     pdm run make clean
     ```
