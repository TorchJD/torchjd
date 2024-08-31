from typing import ContextManager, TypeAlias

ExceptionContext: TypeAlias = ContextManager[Exception | None]
