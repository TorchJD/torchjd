The following steps outline how to contribute to torchjd. Additionally, make sure to use issues to communicate with us.

1) Clone the repository

2) Install Python 3.10.13. We use pyenv to manage python versions:
    - Install pyenv: https://github.com/pyenv/pyenv#installation
    - [Ubuntu] Install libraries that are required to install python with pyenv:
      ```bash
      sudo apt-get install build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev
      ```
    - Install a python 3.10.13 version using pyenv:
      ```bash
      pyenv install 3.10.13
      ```
    - Automatically activate this python version when you are inside of this repo (command to run
      from the root of torchjd):
      ```bash
      pyenv local 3.10.13
      ```

3) Install `pdm`:
    - Download and install pdm:
      ```bash
      curl -sSL https://pdm.fming.dev/install-pdm.py | python3 -
      ```
    - Make pdm accessible from your `PATH`. In your `.bashrc` file, add:
      ```bash
      export PATH="$PATH:$HOME/.local/bin"
      ```

4) Inside the project root folder, install the dependencies:
   ```bash
   pdm install --frozen-lockfile
   ```
   It should create a virtual environment in a `.venv` folder.
   > ⚠️ If it does not create this `.venv` folder, you can try to run `pdm venv create`, followed by
   `pdm use .venv/bin/python`, and install the project by re-running `pdm install
   --frozen-lockfile`.

   > 💡 The python version that you should specify in your IDE is
   `path_to_torchjd/.venv/bin/python`.

5) Install pre-commit by running:
   ```bash
   pre-commit install
   ```
   from the root of torchjd. This will register some hooks that git will execute before each commit. These hooks help to maintain the repository, for instance by checking that each file ends with a newline.

6) Make changes to the library. Do not forget to update the `.rst` files of the `docs/source/` folder if needed.

7) Make sure that all tests are still passing. To run the tests, simply go to the root of torchjd and run:
   ```bash
   pdm run pytest --doctest-plus --doctest-rst
   ```
   It will automatically locate all tests and run them. The `--doctest-plus --doctest-rst` flags will
   additionally make it verify that all usage examples present in docstrings produce the advertised
   output.
8) If you want to update the versions of the dependencies in the environment, while still respecting the constraints written in `pyproject.toml`, you have to run:
   ```bash
   pdm update
   ```
   and add the updated `pdm.lock` file to your commit. These locked dependencies are only used to run the tests and will not affect users. Only the version constraints specified in `pyproject.toml` will affect them.
