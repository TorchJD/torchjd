from importlib.util import find_spec


class OptionalDepsNotInstalledError(ModuleNotFoundError):
    pass


def check_dependencies_are_installed(dependency_names: list[str]) -> None:
    """
    Check that the required list of dependencies are installed.

    This can be useful for Aggregators whose dependencies are optional when installing torchjd.
    """

    if any(find_spec(name) is None for name in dependency_names):
        raise OptionalDepsNotInstalledError()
