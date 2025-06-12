"""Central registry of sparse conversions and helpers.

For now this file simply re-exports :func:`to_coalesced_coo`, but keeps
the door open for future registration logic.
"""

from __future__ import annotations

from ._utils import to_coalesced_coo

__all__ = ["to_coalesced_coo"]
