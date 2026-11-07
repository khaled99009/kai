"""Orchestrate one isolated, per-run Alembic migration test database.

This module is deliberately the only place that holds administrator credentials.
It proves the target PostgreSQL instance's identity, creates one uniquely named
database owned by the restricted ``kai_test`` role, runs the migration test
group against that database, verifies the normal test database did not move, and
optionally removes only the database it created.

No function here runs at import time.  The unit tests inject fake admin,
identity, and Docker objects, so they never touch PostgreSQL or Docker.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol

import psycopg
from psycopg import sql
from sqlalchemy.engine.url import make_url

from kai.db.migration_subprocess import (
    AlembicCommandError,
    AlembicTimeout,
    build_child_env,
    run_alembic_checked,
    sanitize_text,
)
from kai.db.target_policy import (
    MIGRATION_DB_NAME_PREFIX,
    MIGRATION_DB_NAME_RE,
    SAFE_TEST_DB_ROLE,
    build_migration_database_url,
    migration_run_target_error,
    migration_target_error,
)

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_DOCKER_ENV_NAMES: Final = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
)
#: The Docker CLI is always addressed with this explicit, project-supplied
#: context.  Nothing is inferred from DOCKER_HOST, DOCKER_CONTEXT, or a default
#: context name, and no ambient Docker routing variable is inherited.
_DOCKER_CONTEXT_ENV: Final = "KAI_TEST_DOCKER_CONTEXT"
_DOCKER_CONFIG_ENV: Final = "KAI_TEST_DOCKER_CONFIG"
_PG_CONTROL_RE: Final = re.compile(r"Database system identifier:\s*(\d+)")
_REVISION_RE: Final = re.compile(r"^([0-9a-f]{12,})(?:\s|$)")
_EXPECTED_TABLES: Final = (
    "alembic_version",
    "tenants",
    "users",
    "tasks",
    "task_plans",
    "memory_entries",
    "runtime_executions",
    "approvals",
    "alerts",
)
_CONTAINER_ID_RE: Final = re.compile(r"^[0-9a-f]{12,64}$")


class IdentityError(RuntimeError):
    """The trusted Docker anchor or PostgreSQL system identity did not match."""


class ProvisioningError(RuntimeError):
    """A safe, isolated migration database could not be provisioned."""


class MigrationOrchestrationError(RuntimeError):
    """The migration group did not finish on a trustworthy database."""


@dataclass(frozen=True)
class OrchestratorConfig:
    runtime_url: str
    admin_url: str
    admin_user: str
    container_id: str | None
    identity_file: Path | None
    anchor_kind: str
    expected_project: str
    expected_service: str
    expected_config_file: Path
    docker_context: str | None = None
    docker_config_dir: Path | None = None
    alembic_timeout: float = 120.0
    pytest_timeout: float = 600.0
    keep_database: bool = False


@dataclass(frozen=True)
class IdentityReference:
    container_id: str
    system_identifier: str
    source: str
    created_at: str | None = None


@dataclass(frozen=True)
class NormalBaseline:
    revision: str | None
    tables: tuple[str, ...]


class DockerOps(Protocol):
    """The narrow Docker surface the orchestrator needs; easy to fake in tests."""

    def inspect(self, container_id: str) -> Mapping[str, object]: ...

    def compose_ps(
        self, project: str, service: str, compose_file: Path
    ) -> str: ...

    def exec(
        self, container_id: str, *args: str
    ) -> subprocess.CompletedProcess[str]: ...

    def daemon_id(self) -> str: ...


class AdminOps(Protocol):
    """The narrow admin surface the orchestrator needs; easy to fake in tests."""

    def __enter__(self) -> AdminOps: ...

    def __exit__(self, *exc: object) -> None: ...

    def check_privileges(self, *, runtime_role: str, expected_admin: str) -> None: ...

    def system_identifier(self) -> str: ...

    def database_exists(self, database_name: str) -> bool: ...

    def create_database(self, database_name: str, owner: str) -> None: ...

    def configure_database(self, database_name: str, owner: str, run_id: str) -> None: ...

    def verify_database(self, database_name: str, owner: str, run_id: str) -> None: ...

    def active_connections(self, database_name: str) -> int: ...

    def drop_own_database(self, database_name: str, owner: str, run_id: str) -> None: ...


class PsycopgAdmin:
    """Synchronous administrator connection used only by the orchestrator."""

    def __init__(self, admin_url: str, *, connect_timeout: int = 10) -> None:
        self._url = make_url(admin_url)
        if self._url.get_backend_name() != "postgresql":
            raise ProvisioningError("administrator target is not PostgreSQL")
        self._connect_timeout = connect_timeout
        self._conn: psycopg.Connection[tuple[object, ...]] | None = None

    def _kwargs(self, database: str | None = None) -> dict[str, object]:
        return {
            "host": self._url.host,
            "port": self._url.port,
            "user": self._url.username,
            "password": self._url.password,
            "dbname": database or self._url.database,
            "connect_timeout": self._connect_timeout,
            "autocommit": True,
        }

    def _connection(self) -> psycopg.Connection[tuple[object, ...]]:
        if self._conn is None:
            raise ProvisioningError("administrator connection is not open")
        return self._conn

    def __enter__(self) -> PsycopgAdmin:
        self._conn = psycopg.connect(**self._kwargs())  # type: ignore[arg-type]
        return self

    def __exit__(self, *exc: object) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _execute(
        self,
        query: sql.SQL | sql.Composed | str,
        params: tuple[object, ...] = (),
    ) -> tuple[object, ...] | None:
        with self._connection().cursor() as cur:
            cur.execute(query, params or None)
            if cur.description is None:
                return None
            return cur.fetchone()

    def check_privileges(self, *, runtime_role: str, expected_admin: str) -> None:
        row = self._execute(
            "SELECT current_user, rolsuper, rolcreatedb, rolcreaterole, rolreplication"
            " FROM pg_roles WHERE rolname = current_user"
        )
        if row is None:
            raise ProvisioningError("cannot read the administrator role")
        current_user, rolsuper, rolcreatedb, _, _ = row
        if current_user != expected_admin:
            raise ProvisioningError("administrator session is not the expected role")
        if not rolsuper and not rolcreatedb:
            raise ProvisioningError("administrator lacks CREATEDB")

        membership = self._execute(
            "SELECT pg_has_role(current_user, %s, 'MEMBER')", (runtime_role,)
        )
        if not rolsuper and not (membership and bool(membership[0])):
            raise ProvisioningError(
                "administrator cannot assign ownership to the runtime role"
            )

        can_identity = self._execute(
            "SELECT has_function_privilege(current_user,"
            " 'pg_control_system()', 'EXECUTE')"
        )
        if not (can_identity and bool(can_identity[0])):
            raise ProvisioningError(
                "administrator cannot read the PostgreSQL system identifier"
            )

        can_stats = self._execute(
            "SELECT rolsuper OR pg_has_role(current_user, 'pg_read_all_stats',"
            " 'MEMBER') FROM pg_roles WHERE rolname = current_user"
        )
        if not (can_stats and bool(can_stats[0])):
            raise ProvisioningError(
                "administrator cannot prove there are no active database connections"
            )

    def system_identifier(self) -> str:
        row = self._execute("SELECT system_identifier FROM pg_control_system()")
        if row is None:
            raise IdentityError("PostgreSQL returned no system identifier")
        return str(row[0])

    def database_exists(self, database_name: str) -> bool:
        row = self._execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)
        )
        return row is not None

    def create_database(self, database_name: str, owner: str) -> None:
        self._execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(database_name), sql.Identifier(owner)
            )
        )

    def _owner_connection(
        self, database_name: str, owner: str
    ) -> psycopg.Connection[tuple[object, ...]]:
        conn = psycopg.connect(**self._kwargs(database_name))  # type: ignore[arg-type]
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(owner)))
        except BaseException:
            conn.close()
            raise
        return conn

    def _database_admin_connection(
        self, database_name: str
    ) -> psycopg.Connection[tuple[object, ...]]:
        return psycopg.connect(**self._kwargs(database_name))  # type: ignore[arg-type]

    def configure_database(self, database_name: str, owner: str, run_id: str) -> None:
        # Re-assert database ownership on the administrator connection.  This
        # is intentional even though CREATE DATABASE receives OWNER: PostgreSQL
        # installations with role membership/default-privilege differences can
        # otherwise retain the bootstrap role as datdba.
        with self._connection().cursor() as cur:
            cur.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(database_name), sql.Identifier(owner)
                )
            )

        # PostgreSQL 15+ initializes ``public`` with the pseudo-role
        # ``pg_database_owner``.  Altering it after SET ROLE can therefore
        # leave the actual namespace owner unchanged.  Use the bootstrap admin
        # directly on the migration database so the ownership invariant is
        # applied explicitly and consistently across PostgreSQL versions.
        conn = self._database_admin_connection(database_name)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("ALTER SCHEMA public OWNER TO {}").format(
                        sql.Identifier(owner)
                    )
                )
                cur.execute(
                    sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(
                        sql.Identifier(owner)
                    )
                )
        finally:
            conn.close()

        with self._connection().cursor() as cur:
            cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(owner)))
            try:
                cur.execute(
                    sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                        sql.Identifier(database_name),
                        sql.Literal(f"kai-test-migrations run={run_id}"),
                    )
                )
            finally:
                cur.execute("RESET ROLE")

    def verify_database(self, database_name: str, owner: str, run_id: str) -> None:
        row = self._execute(
            "SELECT pg_get_userbyid(datdba),"
            " pg_get_userbyid((SELECT nspowner FROM pg_namespace"
            "                 WHERE nspname = 'public')),"
            " shobj_description(oid, 'pg_database')"
            " FROM pg_database WHERE datname = %s",
            (database_name,),
        )
        if row is None:
            raise ProvisioningError("migration database was not created")
        db_owner, schema_owner, comment = row
        # PostgreSQL 15+ reports the default public schema owner as the
        # pseudo-role ``pg_database_owner``.  Database ownership still binds
        # that role to the database owner, so it is valid when datdba is the
        # restricted runtime role we created for this run.
        schema_owned_by_runtime = schema_owner in {owner, "pg_database_owner"}
        if db_owner != owner or not schema_owned_by_runtime:
            raise ProvisioningError(
                "migration database ownership is not the runtime role"
                f" (database={db_owner!r}, schema={schema_owner!r},"
                f" expected={owner!r})"
            )
        if not isinstance(comment, str) or f"run={run_id}" not in comment:
            raise ProvisioningError("migration database provenance marker is missing")

    def active_connections(self, database_name: str) -> int:
        row = self._execute(
            "SELECT count(*) FROM pg_stat_activity"
            " WHERE datname = %s AND pid <> pg_backend_pid()",
            (database_name,),
        )
        return int(str(row[0])) if row is not None else 0

    def drop_own_database(self, database_name: str, owner: str, run_id: str) -> None:
        if MIGRATION_DB_NAME_RE.match(database_name) is None:
            raise ProvisioningError("refusing to drop a non-migration database")
        if not database_name.startswith(MIGRATION_DB_NAME_PREFIX):
            raise ProvisioningError("refusing to drop a database outside this run")

        self.verify_database(database_name, owner, run_id)
        if self.active_connections(database_name) != 0:
            raise ProvisioningError(
                "refusing to drop a database with active connections"
            )
        with self._connection().cursor() as cur:
            cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(owner)))
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database_name)))


def load_config(env: Mapping[str, str] | None = None) -> OrchestratorConfig:
    """Read the orchestrator inputs without opening a connection."""
    values = os.environ if env is None else env
    runtime_url = values.get("DATABASE_URL")
    if not runtime_url:
        raise ProvisioningError("DATABASE_URL is required")
    error = migration_target_error(runtime_url)
    if error is not None:
        raise ProvisioningError(error)
    runtime = make_url(runtime_url)
    if runtime.database != "kai_test":
        raise ProvisioningError("normal test database must be exactly 'kai_test'")
    if values.get("KAI_TEST_MIGRATIONS_DATABASE_URL"):
        raise ProvisioningError(
            "KAI_TEST_MIGRATIONS_DATABASE_URL must be generated per run, not pre-set"
        )
    if values.get("KAI_TEST_ALLOW_DB_TRUNCATE"):
        raise ProvisioningError(
            "KAI_TEST_ALLOW_DB_TRUNCATE must not be set for the migration group"
        )

    admin_user = values.get("KAI_TEST_POSTGRES_ADMIN_USER", "kai_test_admin")
    admin_url = values.get("KAI_TEST_ADMIN_DATABASE_URL")
    if admin_url:
        try:
            admin_parsed = make_url(admin_url)
        except Exception as exc:
            raise ProvisioningError(
                "KAI_TEST_ADMIN_DATABASE_URL is not a valid PostgreSQL URL"
            ) from exc
        if admin_parsed.get_backend_name() != "postgresql":
            raise ProvisioningError(
                "KAI_TEST_ADMIN_DATABASE_URL must name PostgreSQL"
            )
        if admin_parsed.username:
            admin_user = admin_parsed.username
        if not admin_parsed.password:
            raise ProvisioningError(
                "KAI_TEST_ADMIN_DATABASE_URL must include the administrator password"
            )
    else:
        admin_password = values.get("KAI_TEST_POSTGRES_ADMIN_PASSWORD")
        admin_password_file = values.get("KAI_TEST_POSTGRES_ADMIN_PASSWORD_FILE")
        if not admin_password and admin_password_file:
            try:
                admin_password = Path(admin_password_file).read_text(
                    encoding="utf-8"
                ).strip()
            except OSError as exc:
                raise ProvisioningError(
                    "cannot read the administrator password file"
                ) from exc
        if not admin_password:
            raise ProvisioningError("administrator password is required")

        admin_db = values.get("KAI_TEST_POSTGRES_ADMIN_DB", "postgres")
        admin_url = (
            runtime.set(
                username=admin_user,
                password=admin_password,
                database=admin_db,
            )
            .render_as_string(hide_password=False)
        )

    identity_value = values.get("KAI_TEST_PG_IDENTITY_FILE")
    identity_file = Path(identity_value) if identity_value else None
    container_id = values.get("KAI_TEST_PG_CONTAINER_ID") or None
    anchor_kind = values.get("KAI_TEST_PG_ANCHOR_KIND", "ci" if container_id else "compose")

    docker_context = values.get(_DOCKER_CONTEXT_ENV, "").strip()
    if not docker_context:
        raise ProvisioningError(
            f"{_DOCKER_CONTEXT_ENV} is required; refusing to guess a Docker"
            " context from ambient environment variables"
        )
    docker_config_value = values.get(_DOCKER_CONFIG_ENV)
    docker_config_dir = (
        Path(docker_config_value).resolve() if docker_config_value else None
    )

    return OrchestratorConfig(
        runtime_url=runtime_url,
        admin_url=admin_url,
        admin_user=admin_user,
        container_id=container_id,
        identity_file=identity_file,
        anchor_kind=anchor_kind,
        expected_project=values.get("KAI_TEST_PG_PROJECT", "kai-test-db"),
        expected_service=values.get("KAI_TEST_PG_SERVICE", "db-test"),
        expected_config_file=Path(
            values.get(
                "KAI_TEST_PG_COMPOSE_FILE",
                str(_REPO_ROOT / "docker-compose.test.yml"),
            )
        ),
        docker_context=docker_context,
        docker_config_dir=docker_config_dir,
        keep_database=values.get("KAI_TEST_KEEP_MIGRATION_DB", "").lower()
        in {"1", "true", "yes", "on"},
    )


def _docker_env(docker_config_dir: Path | None = None) -> dict[str, str]:
    """Return only the OS settings the Docker CLI needs.

    Docker routing is never inherited.  ``DOCKER_HOST``, ``DOCKER_CONTEXT``,
    ``DOCKER_CONFIG``, ``DOCKER_TLS_VERIFY``, and ``DOCKER_CERT_PATH`` are
    deliberately absent.  Pinned commands receive the per-run private config
    directory; only the one-time snapshot receives the user-supplied source
    config.  The orchestrator process holds the administrator password; the
    Docker CLI has no reason to inherit it (or any future secret).
    """
    allowed = {name.upper() for name in _DOCKER_ENV_NAMES}
    env = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in allowed
    }
    if docker_config_dir is not None:
        env["DOCKER_CONFIG"] = str(docker_config_dir)
    # pg_controldata's field names must not be translated by the host locale.
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


@dataclass(frozen=True)
class _PinnedDockerContext:
    """A private, immutable snapshot of one selected Docker context."""

    name: str
    config_dir: Path


def _read_context_snapshot(
    config: OrchestratorConfig, context: str
) -> Mapping[str, object]:
    """Read the selected context exactly once, before pinning it."""
    try:
        result = subprocess.run(
            ["docker", "--context", context, "context", "inspect", context],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=_docker_env(config.docker_config_dir),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IdentityError(
            "cannot run the Docker CLI to snapshot the selected context"
        ) from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise IdentityError(
            f"cannot snapshot Docker context {context!r}; check"
            f" {_DOCKER_CONTEXT_ENV} and {_DOCKER_CONFIG_ENV}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise IdentityError("Docker context inspect returned invalid JSON") from exc
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise IdentityError("Docker context inspect returned no single context")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise IdentityError("Docker context inspect returned no context")
    return payload


def _pin_docker_context(config: OrchestratorConfig) -> _PinnedDockerContext:
    """Copy the selected context into a private config used for the whole run.

    Resolving a context name on every command lets a concurrent edit of the
    user's Docker config redirect later commands to another endpoint.  The
    snapshot copies the endpoint metadata and any TLS material once, then every
    command resolves the private context from the temporary config directory.
    """
    context = (config.docker_context or "").strip()
    if not context:
        raise IdentityError(
            f"{_DOCKER_CONTEXT_ENV} is required; refusing to guess a Docker"
            " context from ambient environment variables"
        )

    source = _read_context_snapshot(config, context)
    endpoints = source.get("Endpoints")
    if not isinstance(endpoints, dict):
        raise IdentityError("Docker context inspect returned no endpoints")
    docker_endpoint = endpoints.get("docker")
    if not isinstance(docker_endpoint, dict):
        raise IdentityError("Docker context has no docker endpoint")
    host = docker_endpoint.get("Host")
    if not isinstance(host, str) or not host.strip():
        raise IdentityError("Docker context has no usable docker endpoint")

    metadata = source.get("Metadata", {})
    if not isinstance(metadata, dict):
        raise IdentityError("Docker context inspect returned invalid metadata")
    # ``docker context inspect`` reports TLS material as
    # ``map[endpointName][]fileName`` (docker/cli ``store.EndpointFiles``), built
    # by listing ``contexts/tls/<store id>/<endpoint>/``.  It is used only as the
    # "this context has TLS material to snapshot" signal; the copy itself uses
    # the reported ``Storage.TLSPath`` root, which preserves the per-endpoint
    # directory layout the CLI expects.
    tls_material = source.get("TLSMaterial", {})
    if not isinstance(tls_material, dict):
        raise IdentityError("Docker context inspect returned invalid TLS material")

    storage = source.get("Storage")
    tls_source: Path | None = None
    if isinstance(storage, dict):
        raw_tls_path = storage.get("TLSPath")
        if isinstance(raw_tls_path, str) and raw_tls_path.strip():
            tls_source = Path(raw_tls_path)
    if tls_material:
        if tls_source is None:
            raise IdentityError("Docker context reports TLS material without a path")
        if not tls_source.is_dir():
            raise IdentityError("Docker context TLS material path is not a directory")

    pinned_name = f"kai-pinned-{secrets.token_hex(8)}"
    try:
        config_dir = Path(tempfile.mkdtemp(prefix="kai-docker-context-"))
    except OSError as exc:
        raise IdentityError("cannot create the per-run Docker context") from exc
    atexit.register(shutil.rmtree, config_dir, ignore_errors=True)

    # The store id is the lower-case hex SHA-256 of the context name
    # (docker/cli ``store.contextdirOf`` -> ``digest.FromString(name).Encoded()``
    # with the canonical SHA-256 algorithm).  The same id names the metadata
    # directory and the TLS directory of one context.
    name_hash = hashlib.sha256(pinned_name.encode("utf-8")).hexdigest()
    meta_dir = config_dir / "contexts" / "meta" / name_hash
    try:
        meta_dir.mkdir(parents=True, exist_ok=True)
        # ``meta.json`` holds exactly the Docker CLI context-store metadata
        # contract (``store.Metadata``: name, metadata, endpoints).  The TLS
        # listing is deliberately *not* written here: it exists only in
        # ``docker context inspect`` output, the CLI ignores any key it does not
        # know, and it drops such a key when it rewrites the file.  The CLI
        # resolves TLS by listing ``contexts/tls/<store id>/<endpoint>/``, so the
        # copied tree below is what carries the material, not this metadata.
        (meta_dir / "meta.json").write_text(
            json.dumps(
                {
                    "Name": pinned_name,
                    "Metadata": metadata,
                    "Endpoints": endpoints,
                }
            ),
            encoding="utf-8",
        )
        if tls_material and tls_source is not None:
            shutil.copytree(
                tls_source,
                config_dir / "contexts" / "tls" / name_hash,
                dirs_exist_ok=True,
            )
    except (OSError, shutil.Error) as exc:
        raise IdentityError("cannot snapshot the selected Docker context") from exc

    return _PinnedDockerContext(name=pinned_name, config_dir=config_dir)


class DockerCli:
    """One Docker CLI session pinned to a private context snapshot and daemon."""

    def __init__(self, config: OrchestratorConfig) -> None:
        pinned = _pin_docker_context(config)
        self._context = pinned.name
        self._config_dir = pinned.config_dir
        self._daemon_id: str | None = None

    def _command(self, *args: str) -> list[str]:
        return ["docker", "--context", self._context, *args]

    def _query_daemon_id(self) -> str:
        try:
            result = subprocess.run(
                self._command("info", "--format", "{{.ID}}"),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
                env=_docker_env(self._config_dir),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise IdentityError(
                "cannot run the Docker CLI to verify the selected daemon"
            ) from exc
        daemon_id = result.stdout.strip()
        if result.returncode != 0 or not daemon_id:
            raise IdentityError(
                "cannot verify the selected Docker daemon; check"
                f" {_DOCKER_CONTEXT_ENV} and {_DOCKER_CONFIG_ENV}"
            )
        return daemon_id

    def _verify_daemon(self) -> str:
        actual = self._query_daemon_id()
        if self._daemon_id is None:
            self._daemon_id = actual
        elif actual != self._daemon_id:
            raise IdentityError(
                "Docker daemon changed during the migration run; refusing to continue"
            )
        return self._daemon_id

    def _run(
        self, *args: str, timeout: float = 30
    ) -> subprocess.CompletedProcess[str]:
        self._verify_daemon()
        try:
            return subprocess.run(
                self._command(*args),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env=_docker_env(self._config_dir),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise IdentityError("cannot run the Docker CLI") from exc

    def daemon_id(self) -> str:
        return self._verify_daemon()

    def inspect(self, container_id: str) -> Mapping[str, object]:
        result = self._run("inspect", container_id)
        if result.returncode != 0:
            raise IdentityError("cannot inspect the trusted PostgreSQL container")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise IdentityError("Docker inspect returned invalid JSON") from exc
        if (
            not isinstance(payload, list)
            or not payload
            or not isinstance(payload[0], dict)
        ):
            raise IdentityError("Docker inspect returned no container")
        return payload[0]

    def compose_ps(
        self, project: str, service: str, compose_file: Path
    ) -> str:
        result = self._run(
            "compose",
            "-f",
            str(compose_file),
            "-p",
            project,
            "ps",
            "-q",
            service,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise IdentityError("cannot find the Compose test database container")
        return result.stdout.strip().splitlines()[0]

    def exec(
        self, container_id: str, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return self._run("exec", container_id, *args)


def _discover_compose_container(
    docker: DockerOps, project: str, service: str, compose_file: Path
) -> str:
    return docker.compose_ps(project, service, compose_file)


def _port_matches(info: Mapping[str, object], runtime_url: str) -> bool:
    runtime_port = make_url(runtime_url).port
    network = info.get("NetworkSettings")
    if not isinstance(network, dict):
        return False
    ports = network.get("Ports")
    if not isinstance(ports, dict):
        return False
    bindings = ports.get("5432/tcp")
    if not isinstance(bindings, list):
        return False
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        host_port = binding.get("HostPort")
        host_ip = binding.get("HostIp")
        if str(host_port) == str(runtime_port) and host_ip in {
            "127.0.0.1",
            "::1",
            "0.0.0.0",
            "::",
        }:
            return True
    return False


def _verify_docker_anchor(
    info: Mapping[str, object], config: OrchestratorConfig, container_id: str
) -> None:
    actual_id = info.get("Id")
    if not isinstance(actual_id, str) or not actual_id.startswith(container_id):
        raise IdentityError("Docker container id does not match the trusted anchor")
    if not _port_matches(info, config.runtime_url):
        raise IdentityError("Docker port mapping does not reach the intended target")

    image = info.get("Config")
    image_name = image.get("Image") if isinstance(image, dict) else None
    if not isinstance(image_name, str) or "postgres:17" not in image_name:
        raise IdentityError("Docker container is not the expected PostgreSQL image")

    mounts = info.get("Mounts")
    if not isinstance(mounts, list):
        raise IdentityError("Docker mounts are unavailable")
    data_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict)
        and mount.get("Destination") == "/var/lib/postgresql/data"
    ]
    if not data_mounts:
        raise IdentityError("test database data volume is not mounted")

    if config.anchor_kind == "ci":
        return

    labels = image.get("Labels") if isinstance(image, dict) else None
    if not isinstance(labels, dict):
        raise IdentityError("Compose container labels are missing")
    if labels.get("com.docker.compose.project") != config.expected_project:
        raise IdentityError("Compose project label does not match")
    if labels.get("com.docker.compose.service") != config.expected_service:
        raise IdentityError("Compose service label does not match")

    config_files = labels.get("com.docker.compose.project.config_files")
    if isinstance(config_files, str):
        candidates = {
            Path(part).resolve()
            for part in config_files.split(",")
            if part.strip()
        }
        if config.expected_config_file.resolve() not in candidates:
            raise IdentityError("Compose config file does not match this repository")

    volume_name = data_mounts[0].get("Name")
    if not isinstance(volume_name, str) or "kai_test_pgdata" not in volume_name:
        raise IdentityError("test database volume is not kai_test_pgdata")
    if "kai_pgdata" in volume_name:
        raise IdentityError("refusing the development database volume")


def _read_system_identifier_from_container(
    docker: DockerOps, container_id: str
) -> str:
    commands = (
        ("pg_controldata",),
        ("pg_controldata", "-D", "/var/lib/postgresql/data"),
    )
    for command in commands:
        result = docker.exec(container_id, *command)
        if result.returncode != 0:
            continue
        match = _PG_CONTROL_RE.search(result.stdout + "\n" + result.stderr)
        if match:
            return match.group(1)
    raise IdentityError("cannot read the PostgreSQL system identifier from Docker")


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    normalized = re.sub(r"(\.[0-9]{6})[0-9]+", r"\1", normalized)
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _read_identity_file(
    path: Path, expected_container_id: str | None
) -> IdentityReference:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityError("cannot read the trusted PostgreSQL identity file") from exc
    if not isinstance(payload, dict):
        raise IdentityError("trusted PostgreSQL identity file has no object")
    container_id = payload.get("container_id")
    system_identifier = payload.get("system_identifier")
    created_at = payload.get("created_at")
    if not isinstance(container_id, str) or not _CONTAINER_ID_RE.match(container_id):
        raise IdentityError("trusted identity file has no valid container id")
    if expected_container_id and container_id != expected_container_id:
        raise IdentityError("trusted identity file belongs to another container")
    if not isinstance(system_identifier, str) or not system_identifier.isdigit():
        raise IdentityError("trusted identity file has no system identifier")
    if not isinstance(created_at, str):
        raise IdentityError("trusted identity file has no creation timestamp")
    try:
        _parse_timestamp(created_at)
    except ValueError as exc:
        raise IdentityError(
            "trusted identity file has an invalid creation timestamp"
        ) from exc
    return IdentityReference(
        container_id=container_id,
        system_identifier=system_identifier,
        source="identity-file",
        created_at=created_at,
    )


def _verify_identity_freshness(
    info: Mapping[str, object], reference: IdentityReference
) -> None:
    """Reject an identity file that predates the container it names."""
    if reference.created_at is None:
        return
    state = info.get("State")
    started_at = state.get("StartedAt") if isinstance(state, dict) else None
    if not isinstance(started_at, str):
        started_at = info.get("Created") if isinstance(info.get("Created"), str) else None
    if not isinstance(started_at, str):
        raise IdentityError("Docker container start time is unavailable")
    try:
        created = _parse_timestamp(reference.created_at)
        started = _parse_timestamp(started_at)
    except ValueError as exc:
        raise IdentityError("Docker container/identity timestamps are invalid") from exc
    if created < started:
        raise IdentityError("trusted identity file predates the container start")


def _verify_identity_reference(
    config: OrchestratorConfig,
    reference: IdentityReference,
    docker: DockerOps,
) -> None:
    """Bind one identity reference to the live data directory it names.

    This is the only place that accepts a reference, whether it came from the
    identity file or from a caller.  It re-inspects the named container, verifies
    the Docker anchor and freshness, then reads ``pg_controldata`` from that
    same container and requires exact equality with the reference value.
    """
    if config.anchor_kind not in {"ci", "compose"}:
        raise IdentityError("unsupported Docker anchor kind")
    if _CONTAINER_ID_RE.match(reference.container_id) is None:
        raise IdentityError("trusted identity names no valid Docker container id")
    if not reference.system_identifier.isdigit():
        raise IdentityError("trusted identity has no valid system identifier")

    info = docker.inspect(reference.container_id)
    _verify_docker_anchor(info, config, reference.container_id)
    _verify_identity_freshness(info, reference)
    actual = _read_system_identifier_from_container(docker, reference.container_id)
    if actual != reference.system_identifier:
        raise IdentityError(
            "container system identifier does not match the trusted identity"
            " reference"
        )


def resolve_identity(
    config: OrchestratorConfig, *, docker: DockerOps | None = None
) -> IdentityReference:
    """Resolve the expected system identifier from Docker, never from the URL."""
    if config.anchor_kind not in {"ci", "compose"}:
        raise IdentityError("unsupported Docker anchor kind")
    docker_ops = docker or DockerCli(config)
    if config.identity_file is not None:
        reference = _read_identity_file(config.identity_file, config.container_id)
        _verify_identity_reference(config, reference, docker_ops)
        return reference

    if config.anchor_kind == "ci" and not config.container_id:
        raise IdentityError("CI Docker anchor requires a container id")
    container_id = config.container_id or _discover_compose_container(
        docker_ops,
        config.expected_project,
        config.expected_service,
        config.expected_config_file,
    )
    if _CONTAINER_ID_RE.match(container_id) is None:
        raise IdentityError("Docker anchor names no valid container id")
    info = docker_ops.inspect(container_id)
    _verify_docker_anchor(info, config, container_id)
    return IdentityReference(
        container_id=container_id,
        system_identifier=_read_system_identifier_from_container(
            docker_ops, container_id
        ),
        source="docker-pgcontroldata",
    )


def _revision_from_current(stdout: str) -> str | None:
    for line in stdout.splitlines():
        match = _REVISION_RE.match(line.strip())
        if match:
            return match.group(1)
    return None


def _head_revision() -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(Config(str(_REPO_ROOT / "alembic.ini")))
    head = script.get_current_head()
    if head is None:
        raise MigrationOrchestrationError("Alembic repository has no head revision")
    return head


def _connect_runtime(runtime_url: str) -> psycopg.Connection[tuple[object, ...]]:
    url = make_url(runtime_url)
    return psycopg.connect(
        host=url.host,
        port=url.port,
        user=url.username,
        password=url.password,
        dbname=url.database,
        connect_timeout=10,
        autocommit=True,
    )


def read_normal_baseline(runtime_url: str) -> NormalBaseline:
    """Read only the normal test database's revision and public table names."""
    conn = _connect_runtime(runtime_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version_num FROM alembic_version")
            revision_row = cur.fetchone()
            cur.execute(
                "SELECT tablename FROM pg_tables"
                " WHERE schemaname = 'public' ORDER BY tablename"
            )
            tables = tuple(str(row[0]) for row in cur.fetchall())
        revision = str(revision_row[0]) if revision_row else None
        return NormalBaseline(revision=revision, tables=tables)
    finally:
        conn.close()


def _read_tables(database_url: str) -> tuple[str, ...]:
    conn = _connect_runtime(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables"
                " WHERE schemaname = 'public' ORDER BY tablename"
            )
            return tuple(str(row[0]) for row in cur.fetchall())
    finally:
        conn.close()


def _verify_migration_schema(
    migration_url: str,
    expected_head: str,
    *,
    alembic: Callable[..., object] = run_alembic_checked,
    table_reader: Callable[[str], tuple[str, ...]] = _read_tables,
) -> None:
    result = alembic("current", target_url=migration_url)
    stdout = getattr(result, "stdout", "")
    if _revision_from_current(stdout) != expected_head:
        raise MigrationOrchestrationError(
            "alembic_version does not match the repository head"
        )
    tables = set(table_reader(migration_url))
    missing = sorted(set(_EXPECTED_TABLES) - tables)
    if missing:
        raise MigrationOrchestrationError(
            "migration schema is missing expected tables: " + ", ".join(missing)
        )


def _run_pytest(
    migration_url: str, run_id: str, timeout: float
) -> subprocess.CompletedProcess[str]:
    env = build_child_env(
        migration_url, include_migration_context=True, run_id=run_id
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-m", "db_migration"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout,
            cwd=str(_REPO_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        raise MigrationOrchestrationError("migration pytest group timed out") from exc
    if result.returncode != 0:
        password = None
        with contextlib.suppress(Exception):
            password = make_url(migration_url).password
        detail = sanitize_text(
            result.stderr or result.stdout or "", migration_url, password
        )
        detail = detail.strip()
        if len(detail) > 800:
            detail = detail[:800] + "..."
        suffix = f": {detail}" if detail else ""
        raise MigrationOrchestrationError(
            f"migration pytest group failed with exit code {result.returncode}{suffix}"
        )
    return result


def run_migration_suite(
    config: OrchestratorConfig,
    *,
    identity: IdentityReference | None = None,
    admin: AdminOps | None = None,
    docker: DockerOps | None = None,
    alembic: Callable[..., object] = run_alembic_checked,
    pytest_runner: Callable[[str, str, float], object] = _run_pytest,
    baseline_reader: Callable[[str], NormalBaseline] = read_normal_baseline,
    table_reader: Callable[[str], tuple[str, ...]] = _read_tables,
) -> int:
    """Provision, run, verify, and optionally clean one migration database."""
    docker_ops = docker or DockerCli(config)
    if identity is None:
        reference = resolve_identity(config, docker=docker_ops)
    else:
        # A caller-supplied reference is not trusted by itself.  It must still
        # match the data directory of the live container it names.
        reference = identity
        _verify_identity_reference(config, reference, docker_ops)
    expected_head = _head_revision()
    before = baseline_reader(config.runtime_url)
    if before.revision != expected_head:
        raise MigrationOrchestrationError(
            "normal test database is not at the repository head"
        )

    admin_context = admin or PsycopgAdmin(config.admin_url)
    with admin_context as admin_ops:
        admin_ops.check_privileges(
            runtime_role=SAFE_TEST_DB_ROLE, expected_admin=config.admin_user
        )
        actual_identifier = admin_ops.system_identifier()
        if actual_identifier != reference.system_identifier:
            raise IdentityError(
                "PostgreSQL system identifier does not match the trusted Docker anchor"
            )

        run_id = ""
        database_name = ""
        for _ in range(3):
            candidate_id = secrets.token_hex(8)
            candidate_name = f"{MIGRATION_DB_NAME_PREFIX}{candidate_id}"
            if MIGRATION_DB_NAME_RE.match(candidate_name) is None:
                continue
            if not admin_ops.database_exists(candidate_name):
                run_id = candidate_id
                database_name = candidate_name
                break
        if not database_name:
            raise ProvisioningError(
                "could not allocate an unused migration database name"
            )

        admin_ops.create_database(database_name, SAFE_TEST_DB_ROLE)
        try:
            admin_ops.configure_database(database_name, SAFE_TEST_DB_ROLE, run_id)
            admin_ops.verify_database(database_name, SAFE_TEST_DB_ROLE, run_id)

            migration_url = build_migration_database_url(
                config.runtime_url, database_name
            )
            error = migration_run_target_error(migration_url, run_id)
            if error is not None:
                raise ProvisioningError(error)

            alembic("upgrade", "head", target_url=migration_url)
            _verify_migration_schema(
                migration_url,
                expected_head,
                alembic=alembic,
                table_reader=table_reader,
            )

            pytest_runner(migration_url, run_id, config.pytest_timeout)

            _verify_migration_schema(
                migration_url,
                expected_head,
                alembic=alembic,
                table_reader=table_reader,
            )
            after = baseline_reader(config.runtime_url)
            if after != before:
                raise MigrationOrchestrationError(
                    "normal test database changed during the migration group"
                )

            if not config.keep_database:
                try:
                    admin_ops.drop_own_database(
                        database_name, SAFE_TEST_DB_ROLE, run_id
                    )
                except Exception as exc:
                    detail = sanitize_text(
                        str(exc), config.admin_url, config.runtime_url
                    )
                    print(
                        "warning: migration database cleanup failed; "
                        f"left {database_name} for manual cleanup: {detail}"
                    )
            else:
                print(f"migration database kept: {database_name}")
            return 0
        except (AlembicTimeout, AlembicCommandError, MigrationOrchestrationError):
            print(f"migration database left for investigation: {database_name}")
            raise
        except Exception as exc:
            print(f"migration database left for investigation: {database_name}")
            detail = sanitize_text(
                f"{type(exc).__name__}: {exc}",
                config.admin_url,
                config.runtime_url,
            ).strip()
            if len(detail) > 800:
                detail = detail[:800] + "..."
            suffix = f": {detail}" if detail else ""
            raise MigrationOrchestrationError(
                "migration orchestration failed; database left for investigation"
                + suffix
            ) from exc
