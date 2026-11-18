"""Alembic migration environment with async engine support."""

import asyncio
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5477/video_compiler_service",  # pragma: allowlist secret
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = DATABASE_URL
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 10},
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    except ProgrammingError as exc:
        # Postgres 15+ strips CREATE from `public` for non-owners.
        # Translate the opaque SQLAlchemy traceback into an actionable message
        # so operators don't have to grep the CI log for the root cause.
        msg = str(exc).lower()
        if "permission denied for schema public" in msg:
            print(
                "\n"
                "ERROR: migration could not create objects in schema 'public'.\n"
                "\n"
                "Postgres 15+ no longer grants CREATE on schema 'public' to\n"
                "non-owners by default. Either:\n"
                "  1. Set the CI secret PG_BOOTSTRAP_URL to a superuser DSN so\n"
                "     the migrate initContainer can grant permissions automatically, or\n"
                "  2. Run this once on the target database as a superuser:\n"
                "        GRANT ALL ON SCHEMA public TO <migration-user>;\n"
                "        GRANT ALL ON ALL TABLES IN SCHEMA public TO <migration-user>;\n"
                "        GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO <migration-user>;\n"
                "\n",
                file=sys.stderr,
            )
        raise
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
