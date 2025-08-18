import pytest


# deprecated since 2025-08-18
def test_deprecate_import_backward_from_torchjd():
    with pytest.deprecated_call():
        from torchjd import backward  # noqa: F401


# deprecated since 2025-08-18
def test_deprecate_import_mtl_backward_from_torchjd():
    with pytest.deprecated_call():
        from torchjd import mtl_backward  # noqa: F401
