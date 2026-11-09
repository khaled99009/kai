import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from kai.db.base import Base
from kai.db.target_policy import (
    escape_configparser_percent,
    require_explicit_database_url,
)

config = context.config

# Alembic must never guess a target. An absent DATABASE_URL is refused here --
# before an engine is created or a connection is attempted -- instead of falling
# back to the application settings, whose default is the development database.
# Development and production targets remain supported when supplied explicitly;
# the test-database allowlist belongs to the migration test runner, not here.
try:
    database_url = require_explicit_database_url(os.environ.get("DATABASE_URL"))
except ValueError as exc:
    raise SystemExit(f"alembic: {exc}") from None

# ConfigParser treats "%" as interpolation; escape it so a password containing
# "%" survives set_main_option/get_main_option unchanged. The URL is never
# printed by this module.
config.set_main_option("sqlalchemy.url", escape_configparser_percent(database_url))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _register_models() -> None:
    """Import every model module so ``Base.metadata`` is complete.

    Deliberately imported *after* the explicit-target check: importing some
    packages (for example ``kai.runtime``) runs module-level code that builds the
    application's global engine. Refusing an absent target must happen before any
    engine exists, even an unused one.
    """
    from kai.alerts import models as alert_models  # noqa: F401
    from kai.approvals import models as approval_models  # noqa: F401
    from kai.db import models  # noqa: F401
    from kai.memory import models as memory_models  # noqa: F401
    from kai.planner import models as planner_models  # noqa: F401
    from kai.runtime import effects as runtime_effects  # noqa: F401
    from kai.runtime import models as runtime_models  # noqa: F401
    from kai.tasks import models as task_models  # noqa: F401


_register_models()

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
