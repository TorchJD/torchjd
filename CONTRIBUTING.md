# Contributing to TorchJD

This document explains how to contribute to TorchJD. Please use issues or discussions to communicate
with maintainers before implementing major changes.

## Installation

1) Pre-requisites: To work with TorchJD, you need python to be installed. In the following, we
   suggest to use python 3.12.3, but you can work with any python version supported by `torchjd`. We
   use [pyenv](https://github.com/pyenv/pyenv) to install python and
   [pdm](https://pdm-project.org/en/latest/) to manage dependencies. While the desired python
   version can also be installed without pyenv, the installation of `torchjd` for development
   purposes requires `pdm`. To install it, follow their
   [installation steps](https://pdm-project.org/en/latest/#installation).

2) Create a virtual environment and install the project in it. From the root of `torchjd`, run:
   ```bash
   pdm venv create 3.12.3  # Requires python 3.12.3 to be installed
   pdm use -i .venv/bin/python
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

  - To compute the code coverage locally, you should run the unit tests and the doc tests together,
  with the `--cov` flag:
    ```bash
    pdm run pytest tests/unit tests/doc --cov=src
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

The following guidelines should help preserve a good code quality in TorchJD. Contributions that do
not respect these guidelines will still be greatly appreciated but will require more work from
maintainers to be merged.

### Documentation

Most source python files in TorchJD have a corresponding `.rst` in `docs/source`. Please make sure
to add such a documentation entry whenever you add a new public module. In most cases, public
classes should contain a usage example in their docstring. We also ask contributors to add an entry
in the `[Unreleased]` section of the changelog whenever they make a change that may affect users (we
do not report internal changes). If this section does not exist yet (right after a release), you
should create it.

### Testing

We ask contributors to implement the unit tests necessary to check the correctness of their
implementations. Besides, whenever usage examples are provided, we require the example's code to be
tested in `tests/doc`. We require a very high code coverage for newly introduced sources (~95-100%).

### Coding

We try to keep the quality of the codebase as high as possible. Even if this slows down development
in the short term, it helps a lot in the long term. To make the code easy to understand and to
maintain, we try to keep it simple, and to stick as much as possible to the
[SOLID principles](https://en.wikipedia.org/wiki/SOLID). Try to preserve the existing coding style
of the library when adding new sources. Also, please make sure that new modules are imported by the
`__init__.py` file of the package they are located into. This makes them easier to import for the
user.

## Release

To release a new `torchjd` version, you have to:
- Make sure that all tests, including those on cuda, pass (for this, you need access to a machine
  that has a cuda-enabled GPU).
- Make sure that all important changes since the last release have been reported in the
  `[Unreleased]`
  section at the top of the changelog.
- Add a `[X.Y.Z]` section with the current date in the changelog just below the `[Unreleased]`
  header.
- Change the version in `pyproject.toml`.
- Make a pull request with those changes and merge it.
- Make a draft of the release on GitHub (click on `Releases`, then `Draft a new release`, then fill
  the details.
- Publish the release (click on  `Publish release`). This should trigger the deployment of the new
  version on PyPI and the building and deployment of the documentation on github-pages.
- Check that the new version is correctly deployed to PyPI, that it is installable and that it
  works.
- Check that the documentation has been correctly deployed.
