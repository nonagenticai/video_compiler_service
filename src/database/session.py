"""Database session management module.

Thin re-export from w2v_common. Session factory, FastAPI dependency, and
context-manager helpers live in the shared library; this module preserves
the ``src.database.session`` import path.
"""

from w2v_common.database.session import (
    close_session_maker as close_session_maker,
)
from w2v_common.database.session import (
    get_db as get_db,
)
from w2v_common.database.session import (
    get_db_session as get_db_session,
)
from w2v_common.database.session import (
    get_session_maker as get_session_maker,
)

__all__ = ["close_session_maker", "get_db", "get_db_session", "get_session_maker"]
