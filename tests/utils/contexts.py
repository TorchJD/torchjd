from contextlib import AbstractContextManager
from typing import TypeAlias

ExceptionContext: TypeAlias = AbstractContextManager[Exception | None]
