from importlib.util import find_spec


class _OptionalDepsNotInstalledError(ModuleNotFoundError):
    pass


def check_dependencies_are_installed(dependency_names: list[str]) -> None:
    # Check that the dependencies are installed
    if any(find_spec(name) is None for name in dependency_names):
        raise _OptionalDepsNotInstalledError()
