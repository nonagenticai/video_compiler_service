"""Database connection module with connection pooling and initialization.

Thin re-export from w2v_common. All connection management, engine creation,
and health-check logic lives in the shared library; this module preserves
the ``src.database.connection`` import path.
"""

from w2v_common.database.connection import (
    close_engine as close_engine,
)
from w2v_common.database.connection import (
    get_engine as get_engine,
)
from w2v_common.database.connection import (
    health_check as health_check,
)

__all__ = ["close_engine", "get_engine", "health_check"]
