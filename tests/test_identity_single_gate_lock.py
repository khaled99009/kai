"""The single-gate lock: one writer for ``identities``, in the order it is written.

Two earlier rounds measured this property and stopped at "there is one writer and it is
guarded":

* ``test_identity_approval_binding_design.py`` scans every line of ``src/`` and
  ``alembic/`` with ``re.search(r"\\.identities\\s*=(?!=)", line)`` and compares offsets in
  the *text* of ``_drive``.
* ``test_gmail_authority_review_design.py`` repeats the text-offset comparison.

Both are real guards, and both stay. This round adds the two things they cannot reach,
and they are the two things the safety of the property actually depends on.

**The old scan is one regular expression over assignment syntax.** It cannot see
``row.identities.append(...)``, ``row.identities[0] = x``, ``row.identities += [...]``,
``del row.identities``, ``setattr(row, "identities", ...)``,
``update(RuntimeExecution).values(identities=...)``, ``query.update({...})``,
``bulk_update_mappings``, a second ``RuntimeExecution(identities=...)`` on another path,
or a raw ``"UPDATE runtime_executions SET identities = ..."``. Every one of those is a
writer.

**The old ordering check compares string offsets.** ``drive.index(call) <
drive.index(failure) < drive.index(write)`` proves the *text* is in that order, which for
this shape of code is close to equivalent to the control flow. It does not prove the
write *executes* after the failure gate; it proves a ``return`` appears earlier in the
file.

Both claims were checked rather than argued, by breaking ``_drive`` on purpose and
running the old suite against it:

* a ``row.identities.append(...)`` line added beside the writer - the old suite passed
  **12/12** with a live second writer in the file, while
  :func:`test_identities_has_exactly_one_writer_in_src_and_alembic` failed on it;
* the writer hoisted above the ``try`` - the old ordering test still passed, because the
  text order of the remaining statements was unchanged, while
  :func:`test_drive_resolves_then_writes_the_identity` recorded a write *before* the call;
* the ``except RuntimeFatal`` handler stopped returning - both suites caught this one, so
  the new suite is strictly stronger rather than the only thing standing there.

So this round closes both holes without touching a line of production code. The scan
becomes an AST walk over a documented set of write forms, fail-closed on any site it
cannot classify. The ordering becomes an *observed execution order*: the real ``_drive``
runs against a real PostgreSQL row, the real ``IdentityResolver`` and the real contacts
API decide the outcome, a SQLAlchemy ``set`` listener records every write to the column
together with the source line that performed it, and the failure path is exercised for
real - so "no write happened" is a measured result rather than an absence of evidence.

Nothing here is a policy. The vocabulary, the projection, ``rule_identity``, the
approval binding and the fulfillment semantics are untouched, and the last test in this
file asserts exactly that, so a future round cannot smuggle a semantic change in behind
a technique-only one.

**This round is technique-only: zero intended behavioural change.**
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import event

from kai.identity import IdentityStatus
from kai.memory.models import MemoryType
from kai.memory.service import create_memory
from kai.planner.service import (
    PlannerService,
    PlanNotAllowedError,
    PlanNotFoundError,
)
from kai.runtime.models import RuntimeExecution
from kai.runtime.schemas import MeetingWindow
from kai.runtime.service import (
    _POSITIVE_IDENTITY,
    AgentRuntime,
    UnderstandingOutput,
)
from kai.tasks.models import Task
from kai.tasks.service import LIVE_SEALABLE_PROFILES
from kai.tasks.state import InvalidTransition
from kai.tools.gateway import _IDENTITY_TO_POLICY
from tool_harness import (
    GoogleBackend,
    contact_payload,
    insert_account,
    make_gateway,
    make_task,
    make_tenant_user,
    session_scope,
)

REPO: Final = Path(__file__).resolve().parents[1]
SERVICE: Final = "src/kai/runtime/service.py"
MODELS: Final = "src/kai/runtime/models.py"
API: Final = "src/kai/api/runtime.py"
SCAN_ROOTS: Final = (REPO / "src", REPO / "alembic")
COLUMN: Final = "identities"

#: The head of the *deliverable* migration chain: the merge revision shipped by
#: this branch. All revisions currently present are intentional and therefore
#: there is no local-only allow-list to hide an undeclared head.
DELIVERY_HEAD: Final = "7573ac048c34"
_LOCAL_ONLY_REVISIONS: Final = frozenset()

#: The merged chain includes both the intent-attempt and Google pending branches.
DELIVERY_CHAIN_LENGTH: Final = 17


def _delivery_chain() -> tuple[str, ...]:
    """Return the revisions of the deliverable chain, newest first.

    Counting ``*.py`` files cannot tell the deliverable chain from unrelated local
    work, and it does not prove the files form a chain at all. This walks the chain
    from ``base`` to :data:`DELIVERY_HEAD` through Alembic's own revision map, then
    requires every revision on disk to be either on that chain or declared
    local-only. An *unexpected* migration therefore still fails the guard, whether
    it joins the chain (the length changes) or dangles off it (it is off-chain and
    undeclared).
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(Config(str(REPO / "alembic.ini")))
    chain = tuple(
        rev.revision for rev in script.walk_revisions("base", DELIVERY_HEAD)
    )
    every_revision = {
        rev.revision for rev in script.walk_revisions("base", "heads")
    }
    undeclared = every_revision - set(chain) - _LOCAL_ONLY_REVISIONS
    assert undeclared == set(), (
        f"migration(s) off the deliverable chain and not declared local-only: {sorted(undeclared)}"
    )
    undeclared_heads = set(script.get_heads()) - {DELIVERY_HEAD} - _LOCAL_ONLY_REVISIONS
    assert undeclared_heads == set(), f"undeclared migration head(s): {sorted(undeclared_heads)}"
    return chain

GUEST: Final = "Alice"
ALICE: Final = "alice@example.com"

#: The one statement that may persist a resolved identity. Referenced by shape, never by
#: line number, so a legitimate move inside ``_drive`` is not mistaken for a second writer.
WRITER: Final = "row.identities = identities"
#: The one statement that may hand the column a value before the gate has ever run.
SEED: Final = "identities=[]"


# ---------------------------------------------------------------------------
# Locating things. Every expectation is anchored to source text or to the shape of the
# source, never to a hardcoded line number, so this file does not rot when the file it
# watches moves.
# ---------------------------------------------------------------------------


def _lines(relative: str, fragment: str) -> list[int]:
    text = (REPO / relative).read_text(encoding="utf-8")
    return [n for n, line in enumerate(text.splitlines(), 1) if fragment in line]


def _line(relative: str, fragment: str) -> int:
    found = _lines(relative, fragment)
    assert len(found) == 1, f"expected one {fragment!r} in {relative}, found {found}"
    return found[0]


def _text_at(relative: str, line: int) -> str:
    return (REPO / relative).read_text(encoding="utf-8").splitlines()[line - 1].strip()


def _enclosing_function(relative: str, line: int) -> str:
    """The function or method a line sits inside.

    Lets the allow-list below say *which* reader each site belongs to, so it asserts the
    site is in a reader and not merely that some line number still exists.
    """
    name = "<module>"
    for number, text in enumerate((REPO / relative).read_text(encoding="utf-8").splitlines(), 1):
        if number > line:
            break
        stripped = text.lstrip()
        if stripped.startswith(("def ", "async def ")):
            name = stripped.split("def ", 1)[1].split("(", 1)[0].strip()
    return name


def _service_tree() -> ast.Module:
    return ast.parse((REPO / SERVICE).read_text(encoding="utf-8"), filename=SERVICE)


def _drive_ast() -> ast.AsyncFunctionDef:
    """``_drive`` taken from the whole-file parse, so every ``lineno`` is a file line.

    Parsing ``inspect.getsource(_drive)`` instead would restart the numbering at the
    method's own first line, and the events recorded at run time are attributed to real
    file lines - so the two would not be comparable.
    """
    for node in ast.walk(_service_tree()):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == AgentRuntime.__name__
            and any(
                isinstance(child, ast.AsyncFunctionDef) and child.name == "_drive"
                for child in node.body
            )
        ):
            for child in node.body:
                if isinstance(child, ast.AsyncFunctionDef) and child.name == "_drive":
                    return child
    raise AssertionError("no AgentRuntime._drive in the service module")


def _fails(node: ast.AST) -> list[ast.Return]:
    """Every ``return await self._fail(...)`` in the given subtree.

    The call sits under an ``Await``, so the value is unwrapped before it is inspected.
    """
    found = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Return):
            continue
        value = child.value
        if isinstance(value, ast.Await):
            value = value.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        if isinstance(func, ast.Attribute) and func.attr == "_fail":
            found.append(child)
    return found


def _identity_gate_line() -> int:
    """The line of the ``_fail`` return inside the ``except RuntimeFatal`` handler.

    Located structurally rather than by searching for text. Several other calls to
    ``_fail`` sit in ``_drive``, and some other handlers catch ``RuntimeFatal`` without
    delegating to it. The gate is the one handler that both catches ``RuntimeFatal`` and
    returns through ``_fail``, and requiring exactly one such handler keeps the two
    conditions from being met by different code.
    """
    handlers = []
    for node in ast.walk(_drive_ast()):
        if not isinstance(node, ast.ExceptHandler):
            continue
        named = node.type.id if isinstance(node.type, ast.Name) else None
        if named == "RuntimeFatal" and _fails(node):
            handlers.append(node)
    assert len(handlers) == 1, f"expected one RuntimeFatal gate delegating to _fail: {handlers}"
    returns = _fails(handlers[0])
    assert len(returns) == 1, f"expected one _fail in the RuntimeFatal handler: {returns}"
    return returns[0].lineno  # type: ignore[return-value]


def _gates_after_write() -> list[int]:
    """Every failure gate that sits after the write, in source order.

    There is more than one, because planning can refuse in several ways. The tests assert
    membership rather than a single value: which of them a given run reaches depends on
    what the planner raises, and pretending to know would pin the test to today's
    behaviour rather than to the property.
    """
    writer = _line(SERVICE, WRITER)
    return sorted(r.lineno for r in _fails(_drive_ast()) if r.lineno > writer)


def _resolve_call_line() -> int:
    for node in ast.walk(_drive_ast()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_resolve_guest_identities"
        ):
            return node.lineno  # type: ignore[return-value]
    raise AssertionError("no call to _resolve_guest_identities in _drive")


# ---------------------------------------------------------------------------
# Part 1 - what counts as a writer, and the exhaustive search for one
# ---------------------------------------------------------------------------


#: Methods that mutate a list or dict in place. Calling one on ``.identities`` changes
#: the persisted value exactly as an assignment does, and none contains ``.identities =``.
_IN_PLACE_MUTATORS: Final = frozenset(
    {
        "append", "extend", "insert", "update", "setdefault", "pop", "popitem",
        "clear", "sort", "remove", "reverse", "__setitem__", "__delitem__",
    }
)

#: Calls that persist an argument. A keyword or dict key named ``identities`` reaching
#: one of these writes the column even though no attribute is ever assigned.
_DML_SINKS: Final = frozenset(
    {
        "execute", "exec_driver_sql", "execute_sql", "merge", "add", "add_all",
        "bulk_insert_mappings", "bulk_update_mappings", "bulk_save",
        "values", "update", "set", "insert",
    }
)


def _is_column(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == COLUMN


def _string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _callee(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _dict_has_column(node: ast.AST) -> bool:
    """Does this argument hold a dict keyed by the column, directly or inside a list?

    ``stmt.update({"identities": v})`` is the obvious shape, but
    ``bulk_update_mappings(Model, [{"identities": v}])`` buries the same write one level
    down in a list of mappings - the shape a backfill would actually use. The negative
    control caught that omission, so the walk goes through sequences too.
    """
    if isinstance(node, ast.Dict):
        return any(_is_column(k) for k in node.keys)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_dict_has_column(item) for item in node.elts)
    return False


def _reaches_column(node: ast.AST) -> bool:
    """Does writing into this expression reach the column?

    ``row.identities``, ``row.identities[0]`` and ``row.identities[0:2]`` all qualify: the
    attribute is on the spine, so the value the ORM flushes is what changed.
    """
    current: ast.AST = node
    while True:
        if isinstance(current, ast.Attribute):
            if current.attr == COLUMN:
                return True
            current = current.value
        elif isinstance(current, ast.Subscript):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        else:
            return False


def classify_writes(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Every syntactic form in a module that can persist the ``identities`` column.

    Returns ``(lineno, form, source)`` sorted by line. This vocabulary *is* the
    definition of "writer" for this repository, and it is deliberately wider than what
    the code contains: the purpose of the lock is to catch the form that does not exist
    yet. :func:`test_the_scan_detects_every_form_it_claims_to_cover` proves each arm
    fires, so the width is real rather than asserted.
    """
    found: list[tuple[int, str, str]] = []

    def record(node: ast.AST, form: str) -> None:
        found.append((node.lineno, form, ast.unparse(node)))  # type: ignore[attr-defined]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_reaches_column(t) for t in targets):
                record(node, f"attribute-{type(node).__name__.lower()}")
        elif isinstance(node, ast.Delete):
            if any(_reaches_column(t) for t in node.targets):
                record(node, "attribute-delete")
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _IN_PLACE_MUTATORS
                and _reaches_column(node.func.value)
            ):
                record(node, f"in-place-{node.func.attr}")
            callee = _callee(node)
            if callee in _DML_SINKS and any(_dict_has_column(a) for a in node.args):
                # Both spellings reach a sink: ``stmt.update({...})`` and
                # ``bulk_update_mappings(Model, [{...}])``.
                record(node, f"dml-dict-{callee}")
            if callee == "setattr" and len(node.args) >= 2 and _is_column(node.args[1]):
                record(node, "setattr")
            if callee in _DML_SINKS:
                if any(kw.arg == COLUMN for kw in node.keywords):
                    record(node, f"dml-keyword-{callee}")
                for arg in node.args:
                    literal = _string(arg)
                    if literal and COLUMN in literal.lower():
                        record(node, f"dml-literal-{callee}")
    return sorted(found)


def scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        assert root.is_dir(), f"scan root missing, so the scan would prove nothing: {root}"
        files.extend(sorted(root.rglob("*.py")))
    assert files, "the scan read no files, so it proved nothing"
    return files


def repo_writers() -> list[str]:
    hits: list[str] = []
    for path in scanned_files():
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, form, src in classify_writes(tree):
            hits.append(f"{rel}:{line} [{form}] {src}")
    return hits


def test_identities_has_exactly_one_writer_in_src_and_alembic() -> None:
    """The single-writer invariant, by AST exhaustion rather than by grep.

    A **writer** is any syntactic site that can change the persisted value of
    ``runtime_executions.identities``; :func:`classify_writes` enumerates them. The code
    has exactly one, and it is the guarded one.
    """
    files = scanned_files()
    # If the roots ever stop covering the package and the migration history, a writer
    # elsewhere would be invisible. This keeps the scan honest about its own reach.
    assert any(p.name == "service.py" for p in files)
    assert any("versions" in p.parts for p in files), "alembic/versions was not scanned"

    writer_line = _line(SERVICE, WRITER)
    assert repo_writers() == [f"{SERVICE}:{writer_line} [attribute-assign] {WRITER}"], (
        f"expected exactly one writer, found: {repo_writers()}"
    )

    # The writer is the one after the guarded call and after the gate that returns early.
    drive = _drive_ast()
    assignments = [
        node
        for node in ast.walk(drive)
        if isinstance(node, ast.Assign) and any(_reaches_column(t) for t in node.targets)
    ]
    assert len(assignments) == 1, f"expected one write in _drive, found {assignments}"
    assert ast.unparse(assignments[0]) == WRITER
    assert _resolve_call_line() < assignments[0].lineno
    assert _identity_gate_line() < assignments[0].lineno

    # The gate it depends on is executable code, not a comment.
    resolver = textwrap.dedent(inspect.getsource(AgentRuntime._resolve_guest_identities))
    assert "if resolved.status not in _POSITIVE_IDENTITY:" in resolver
    assert 'raise RuntimeFatal(f"attendee identity not established: {guest}")' in resolver
    assert frozenset(
        {
            IdentityStatus.CONTACT_MATCH,
            IdentityStatus.PROVIDER_VERIFIED,
            IdentityStatus.CHANNEL_VERIFIED,
        }
    ) == _POSITIVE_IDENTITY


def test_the_only_construction_of_the_row_seeds_an_empty_list() -> None:
    """The row starts empty, so a failed grounding leaves nothing behind for a reader.

    ``RuntimeExecution(...)`` is built in exactly one place and passes ``identities=[]``.
    That literal matters for the same reason the writer does: a non-empty seed would hand
    the readers an identity no gate ever approved. So the seed is pinned *by value*, not
    merely counted, and the column is pinned as non-nullable so nothing can leave it
    undefined for a reader to interpret.
    """
    constructions: list[int] = []
    seeds: list[str] = []
    for path in scanned_files():
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee(node)
            if name is None or RuntimeExecution.__name__ not in name:
                continue
            constructions.append(node.lineno)
            assert rel == SERVICE, f"the row is also constructed in {rel}:{node.lineno}"
            for kw in node.keywords:
                if kw.arg == COLUMN:
                    # The value is the invariant. The keyword's own line number is not,
                    # and pinning it would only encode how the call happens to wrap.
                    seeds.append(ast.unparse(kw.value))

    assert len(constructions) == 1, f"the row must be built in one place, got {constructions}"
    assert _text_at(SERVICE, constructions[0]).endswith(f"{RuntimeExecution.__name__}(")
    assert seeds == ["[]"], f"the seed must be the empty list, found {seeds}"

    column = RuntimeExecution.__table__.c.identities
    assert column.nullable is False
    # SQLAlchemy wraps the declared default in a context-aware callable, so the
    # declaration itself is what gets pinned: non-nullable, defaulting to an empty list.
    declaration = _lines(MODELS, f"{COLUMN}: Mapped[list] = mapped_column(")
    assert len(declaration) == 1, f"expected one declaration of the column: {declaration}"
    assert _text_at(MODELS, declaration[0]).endswith("nullable=False, default=list)")


def test_no_dml_or_setattr_path_can_write_the_column() -> None:
    """No raw SQL, bulk update, or ``setattr`` route exists - and the column says so.

    The AST scan already covers these arms. This test states the negative result
    separately, so a regression in either the scan or the code surfaces in the one that
    is not expected to fail. The migrations are checked too: the column is declared once
    and backfilled never, so no deployed database can gain a writer by being upgraded.
    """
    offenders: list[str] = []
    for path in scanned_files():
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee(node)
            if callee == "setattr" and len(node.args) >= 2 and _is_column(node.args[1]):
                offenders.append(f"{rel}:{node.lineno} setattr")
            if callee in _DML_SINKS:
                if any(kw.arg == COLUMN for kw in node.keywords):
                    offenders.append(f"{rel}:{node.lineno} dml-keyword {callee}")
                if any(_dict_has_column(a) for a in node.args):
                    offenders.append(f"{rel}:{node.lineno} dml-dict {callee}")
                for arg in node.args:
                    literal = _string(arg)
                    if literal and COLUMN in literal.lower():
                        offenders.append(f"{rel}:{node.lineno} dml-literal {literal[:48]!r}")
    assert offenders == [], f"a DML path names the column: {offenders}"

    migrations = sorted((REPO / "alembic" / "versions").glob("*.py"))
    # The guard is the deliverable chain, not a file count: 15 revisions from base
    # to ``a7c9e1f3b5d2`` (the O1-B INTENT attempt record). Declaring the length
    # means an *unintended* migration cannot appear unnoticed -- a new one has to
    # be declared here -- while local, unapproved work (``b2f7c1d9e4a6``, deferred
    # Google linking) is named rather than counted, so the guard passes in a clean
    # delivery tree where that file is absent.
    chain = _delivery_chain()
    assert len(chain) == DELIVERY_CHAIN_LENGTH, f"unexpected deliverable chain: {chain}"
    assert chain[0] == DELIVERY_HEAD, f"deliverable chain head is not declared: {chain[0]}"
    declaring = [p for p in migrations if "'identities'," in p.read_text(encoding="utf-8")]
    assert len(declaring) == 1, f"expected one migration to declare the column: {declaring}"
    text = declaring[0].read_text(encoding="utf-8")
    for verb in ("INSERT", "UPDATE", "op.bulk_insert", "bulk_update"):
        assert verb not in text, f"the migration is not a pure declaration: found {verb}"
    for path in migrations:
        body = path.read_text(encoding="utf-8")
        assert "identities" not in body or path in declaring, (
            f"{path.name} touches the column but does not declare it"
        )

    # The one server-side default is the empty array, so a row inserted outside the ORM
    # still starts with nothing in it. At the database level too, the gate stays the only
    # way in.
    assert "server_default=sa.text(\"'[]'::jsonb\")" in text, (
        "the column must default to an empty array at the database level"
    )
    assert RuntimeExecution.__table__.c.identities.server_default is None, (
        "the model relies on the ORM default; the migration supplies the SQL one"
    )


def test_every_other_identities_site_is_a_read_projection() -> None:
    """The fail-closed half: a site the scan cannot classify fails the build.

    An AST cannot resolve what an arbitrary callee does with a keyword argument, so it
    cannot tell a write from a read in ``SomeCall(identities=x)``. Rather than guess, the
    burden of proof is inverted: **every** ``identities=`` keyword or dict-key site in the
    repository must be listed below, and each entry names the function it sits in. A new
    one - the shape a seed fixture or a repair path would take - fails here, and adding
    it means classifying it out loud in review.

    Two read the value back off the row, which is the strongest form of the claim: a
    reader cannot invent an identity the writer did not persist. The other three are
    handed the writer's own local, and the ordering assertion below checks that local
    really is the writer's.
    """
    classified: dict[str, str] = {}
    for path in scanned_files():
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # The keyword's own line, not the call's: a call spans several lines, and the
            # site under review is the argument.
            for kw in node.keywords:
                if kw.arg == COLUMN:
                    classified[f"{rel}:{kw.lineno}"] = _enclosing_function(rel, kw.lineno)
            for arg in node.args:
                if _dict_has_column(arg):
                    classified[f"{rel}:{node.lineno}"] = _enclosing_function(rel, node.lineno)

    row_reads = _lines(SERVICE, f"{COLUMN}=list(row.{COLUMN}),")
    threaded = _lines(SERVICE, f"{COLUMN}={COLUMN},")
    api_reads = _lines(API, f"redact_identities(list(row.{COLUMN}))")
    seed_line = _line(SERVICE, SEED)

    assert (len(row_reads), len(threaded), len(api_reads)) == (1, 3, 1), (
        f"unexpected projection shape: {row_reads} {threaded} {api_reads}"
    )
    expected = {
        f"{SERVICE}:{row_reads[0]}": "_settle_verification",
        f"{SERVICE}:{threaded[0]}": "_execute",
        f"{SERVICE}:{threaded[1]}": "_fulfillment_outcome",
        f"{SERVICE}:{threaded[2]}": "_fulfillment_outcome",
        f"{SERVICE}:{seed_line}": "submit",
        f"{API}:{api_reads[0]}": _enclosing_function(API, api_reads[0]),
    }

    assert set(classified) == set(expected), (
        "the set of identities sites changed. New: "
        f"{sorted(set(classified) - set(expected))}. "
        f"Gone: {sorted(set(expected) - set(classified))}"
    )
    # Each site is in the reader it is supposed to be in. This is the check that would
    # fail if one of them ever migrated into something that writes.
    for site, function in expected.items():
        assert classified[site] == function, f"{site} is inside {classified[site]}, not {function}"

    # The API reads the row and redacts on the way out; the redaction is a projection
    # concern, not a trust one, and no reader inspects status.
    assert "redact_identities" in (REPO / API).read_text(encoding="utf-8")

    # The threaded local is the writer's own result, so the criteria see that same list
    # rather than a recomputation of it.
    assert _resolve_call_line() < _line(SERVICE, WRITER) < _line(SERVICE, "self._intents_for(")


def test_the_scan_detects_every_form_it_claims_to_cover() -> None:
    """Negative control: the scanner must fail on writers that do not exist yet.

    A guard that cannot fail is documentation, not enforcement. Each case is a plausible
    future writer - a seed, a repair, a backfill, a bulk update, a subscript edit, a
    ``del`` - and the assertion is that the scan classifies it. Without this,
    "exactly one writer" could be satisfied by a scanner that matches almost nothing.
    """
    # A list of pairs, not a dict: two cases legitimately share a form, and a dict would
    # silently drop one of them.
    must_detect: list[tuple[str, str]] = [
        ("attribute-assign", "row.identities = identities"),
        ("attribute-augassign", "row.identities += [extra]"),
        ("attribute-annassign", "row.identities: list = other"),
        ("attribute-delete", "del row.identities"),
        ("attribute-assign", "row.identities[0] = item"),
        ("attribute-assign", "row.identities[0:2] = items"),
        ("in-place-append", "row.identities.append(item)"),
        ("in-place-extend", "row.identities.extend(items)"),
        ("in-place-update", "row.identities.update({'a': 1})"),
        ("in-place-pop", "row.identities.pop(0)"),
        ("in-place-clear", "row.identities.clear()"),
        ("setattr", "setattr(row, 'identities', identities)"),
        ("dml-keyword-values", "stmt.values(identities=identities)"),
        ("dml-keyword-update", "stmt.update(identities=identities)"),
        ("dml-dict-update", "stmt.update({'identities': identities})"),
        (
            "dml-dict-bulk_update_mappings",
            "bulk_update_mappings(RuntimeExecution, [{'identities': v}])",
        ),
        (
            "dml-literal-execute",
            "conn.execute('UPDATE runtime_executions SET identities = :v')",
        ),
    ]
    assert len(must_detect) == 17
    for expected_form, snippet in must_detect:
        forms = {form for _, form, _ in classify_writes(ast.parse(snippet))}
        assert expected_form in forms, f"{snippet!r} gave {forms}, expected {expected_form}"

    # And it must stay quiet on reads, or the uniqueness assertion would be worthless.
    for read_only in (
        "value = row.identities",
        "snapshot = list(row.identities)",
        "print(row.identities[0])",
        "if row.identities: pass",
        "n = len(row.identities)",
        "text = f'{row.identities}'",
        "for item in row.identities: print(item)",
    ):
        assert classify_writes(ast.parse(read_only)) == [], f"false positive: {read_only!r}"


# ---------------------------------------------------------------------------
# Part 2 - the order, observed rather than read off the source
# ---------------------------------------------------------------------------


class _UnreachableModel:
    """A planner model that these tests must never reach.

    ``_drive`` is stopped either by the identity gate or by the absence of a plan, and
    both happen before any tool runs. Reaching plan generation would mean the test had
    travelled further into the pipeline than it meant to, so it fails loudly instead.
    """

    model_id = "unreachable-v0"

    async def generate_plan(self, *, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise AssertionError("these tests must not reach plan generation")


def _scenario() -> dict[str, Any]:
    start = date.today() + timedelta(days=3)
    return UnderstandingOutput(
        guests=[GUEST],
        window=MeetingWindow(
            start_date=start,
            end_date=start + timedelta(days=1),
            earliest_time="09:00",
            title="KAI",
        ),
    ).model_dump(mode="json")


def _calling_site() -> str:
    """The innermost frame in ``runtime/service.py``, as ``file:line``.

    SQLAlchemy's ``set`` event fires from inside the descriptor, so the handler's own
    frame says nothing. Walking out to the first ``runtime/service.py`` frame attributes
    each event to the statement that actually performed it, which is what ties an observed
    run back to the single static site. Writes from outside the runtime - the
    construction, the commit, the load - are attributed to nothing, and the tests require
    every event they examine to have come from inside.
    """
    frame = inspect.currentframe()
    while frame is not None:
        name = frame.f_code.co_filename.replace("\\", "/")
        if name.endswith("kai/runtime/service.py"):
            return f"service.py:{frame.f_lineno}"
        frame = frame.f_back
    return "<outside runtime/service.py>"


@contextmanager
def _observe() -> Iterator[list[tuple[str, str, Any]]]:
    """Record the call, the failure gates and the identity write, in execution order.

    Three independent instruments, all attached to the running pipeline:

    * a SQLAlchemy ``set`` listener on the column, which fires for *any* write by *any*
      code path, so a write that bypassed the single static site would still be seen;
    * a wrapper on ``_resolve_guest_identities`` that records the call and then delegates
      to the real method, so grounding itself is never faked;
    * a wrapper on ``_fail`` that records which gate fired before delegating to the real
      transition.

    Each event carries the source line that produced it, so every assertion below is an
    assertion about *this* run of the real ``_drive``.
    """
    events: list[tuple[str, str, Any]] = []
    original_resolve = AgentRuntime._resolve_guest_identities
    original_fail = AgentRuntime.__dict__["_fail"].__func__

    async def spy_resolve(self: AgentRuntime, *args: Any, **kwargs: Any) -> Any:
        events.append(("call", _calling_site(), None))
        return await original_resolve(self, *args, **kwargs)

    async def spy_fail(cls: Any, session: Any, task: Any, reason: str, **kwargs: Any) -> Any:
        events.append(("failure", _calling_site(), reason))
        return await original_fail(cls, session, task, reason, **kwargs)

    def on_set(target: Any, value: Any, oldvalue: Any, initiator: Any) -> Any:
        events.append(("write", _calling_site(), value))
        return value

    event.listen(RuntimeExecution.identities, "set", on_set, retval=True)
    AgentRuntime._resolve_guest_identities = spy_resolve  # type: ignore[method-assign]
    AgentRuntime._fail = classmethod(spy_fail)  # type: ignore[method-assign]
    try:
        yield events
    finally:
        event.remove(RuntimeExecution.identities, "set", on_set)
        AgentRuntime._resolve_guest_identities = original_resolve  # type: ignore[method-assign]
        AgentRuntime._fail = classmethod(original_fail)  # type: ignore[method-assign]


async def _drive_in_session(
    session: Any, events: list[tuple[str, str, Any]], *, resolvable: bool
) -> tuple[Any, RuntimeExecution, Any, httpx.AsyncClient]:
    """Drive the real ``_drive`` once, in a session the caller owns.

    ``resolvable`` decides one thing and one thing only: whether the guest can be
    grounded. Both runs share this code, this resolver and this listener, so a difference
    in the events is attributable to the gate rather than to the harness.

    Grounding needs two production facts to line up. A name alone is never a positive
    identity, so the address book must be reachable by email - which the runtime obtains
    from a ``PERSON`` memory entry, treated as a hint and then re-checked by the resolver
    against live evidence. For the failing run both are withheld, so the resolver returns
    ``UNKNOWN`` on its own judgement and the real guard raises.

    The task is left in ``UNDERSTANDING`` with the scenario already on the row, so Stage 1
    is skipped and identity is the first thing that runs (the only state in which an empty
    identity set is still legitimate). The plan stage is replaced by a governed planning
    refusal, so the run stops *after* the write and at a failure gate that is recorded as
    its own event, never mistaken for the identity gate. The row carries this worker's
    claim token, so the fences pass.
    """
    owner = f"owner-{uuid4().hex[:10]}@corp.example"
    tenant, user = await make_tenant_user(owner)
    await insert_account(
        tenant_id=tenant.id, user_id=user.id, email=owner, sub=f"sub-{uuid4().hex[:8]}"
    )
    task = await make_task(
        tenant_id=tenant.id, user_id=user.id, state="UNDERSTANDING", risk_policy="MEDIUM"
    )
    backend = GoogleBackend()
    if resolvable:
        await create_memory(
            session,
            tenant_id=tenant.id,
            user_id=user.id,
            memory_type=MemoryType.PERSON,
            content=f"{GUEST} صديقي وبريده {ALICE}",
        )
        backend.contacts_connections = [
            contact_payload(resource_name="people/1", display_name=GUEST, emails=(ALICE,))
        ]
    row = RuntimeExecution(
        task_id=task.id,
        tenant_id=tenant.id,
        user_id=user.id,
        timezone="UTC",
        scenario=_scenario(),
        identities=[],
        evidence=[],
        claim_token=uuid4(),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    # The construction, the commit and the load each set the attribute. Discard them, so
    # the events left are the ones the pipeline produced and nothing else.
    events.clear()

    transport = httpx.AsyncClient(
        transport=httpx.MockTransport(backend.handler),
        base_url="http://google.api.test",
        timeout=15.0,
    )
    try:
        gateway, _ = make_gateway()
        runtime = AgentRuntime(gateway=gateway, model=_UnreachableModel())
        # The identity write must still be observable in a state where an empty
        # set is legitimate, and the run must still stop after it. A governed
        # planning refusal does that without a plan, a model call, or a state in
        # which the lifecycle guard would refuse before identity runs.
        with patch.object(
            PlannerService,
            "create_plan",
            side_effect=PlanNotAllowedError(
                "observation stops after the identity stage"
            ),
        ):
            outcome = await runtime._drive(
                session, task=task, row=row, transport=transport, claim=row.claim_token
            )
        # ``make_task`` hands back an instance from its own session, and the copy in hand
        # is stale once ``_fail`` has committed the transition. Re-reading it through this
        # session means the tests assert persisted state rather than a local leftover.
        persisted = await session.get(Task, task.id)
        assert persisted is not None
        return outcome, row, persisted, transport
    except BaseException:
        await transport.aclose()
        raise


DriveResult = tuple[Any, RuntimeExecution, Any, list[tuple[str, str, Any]]]


async def _drive_once(*, resolvable: bool) -> DriveResult:
    """One observed drive, with the harness torn down afterwards."""
    with _observe() as events:
        async with session_scope() as session:
            outcome, row, task, transport = await _drive_in_session(
                session, events, resolvable=resolvable
            )
            try:
                return outcome, row, task, list(events)
            finally:
                await transport.aclose()


async def test_drive_resolves_then_writes_the_identity() -> None:
    """The write happens, it happens once, and it happens after the call.

    Every value in the log was produced by production code: the real resolver read the
    real contacts API and returned ``CONTACT_MATCH``, and the write is the single static
    statement. The assertion is on the sequence *and* on the source line each event came
    from, so it fails if the write is hoisted above the call, if a second write appears,
    or if the write leaves ``_drive``.

    The trailing failure is the governed planning refusal, in a later stage. It is
    asserted to sit at a *different* line than the identity gate, which is what stops the
    two from being conflated in this log.
    """
    outcome, row, _, events = await _drive_once(resolvable=True)

    assert [kind for kind, _, _ in events] == ["call", "write", "failure"], (
        f"expected call, then write, then the later gate: {events}"
    )
    assert all(site.startswith("service.py:") for _, site, _ in events), (
        f"an event came from outside the runtime, so the log is not all pipeline: {events}"
    )
    call_site, write_site, _ = (site for _, site, _ in events)
    assert call_site == f"service.py:{_resolve_call_line()}"
    assert write_site == f"service.py:{_line(SERVICE, WRITER)}"

    # The value is the resolver's own, persisted by the one writer.
    assert events[1][2] == [
        {"name": GUEST, "email": ALICE, "status": "CONTACT_MATCH", "confidence": 0.5}
    ]
    assert row.identities == events[1][2]

    # The identity gate did not fire. The failure that did is one of the planning gates,
    # every one of which sits after the write and none of which is the identity gate.
    assert int(events[2][1].split(":")[1]) in _gates_after_write(), (
        f"the later failure did not come from a post-write gate: {events}"
    )
    assert _identity_gate_line() < _line(SERVICE, WRITER)
    assert _identity_gate_line() not in _gates_after_write()
    assert "plan" in events[2][2].lower()
    assert outcome.task_state == "FAILED"


async def test_a_failed_grounding_never_reaches_the_identity_write() -> None:
    """The property that matters: when the gate fires, the write does not happen.

    This is the other half of ``call < failure < write``, and the half a text-offset
    comparison cannot establish - it shows the ``return`` precedes the write in the file,
    not that the write never *executes*. Here the resolver returns ``UNKNOWN`` on its own
    judgement, the real guard raises, the real ``_fail`` transitions the task, and the
    same column listener that sees the write in the previous test sees none at all.

    That pairing is what makes the absence evidence. The listener is demonstrably capable
    of reporting a write to this column from this pipeline; it reports none because
    nothing wrote.
    """
    outcome, row, task, events = await _drive_once(resolvable=False)

    assert [kind for kind, _, _ in events] == ["call", "failure"], (
        f"expected the call then the identity gate, got {events}"
    )
    assert not [e for e in events if e[0] == "write"], (
        f"the identity column was written despite a failed grounding: {events}"
    )

    # The failure came from the identity gate, at the identity gate's own line, carrying
    # the production message.
    assert events[1][1] == f"service.py:{_identity_gate_line()}"
    assert events[1][2] == f"attendee identity not established: {GUEST}"
    assert _identity_gate_line() < _line(SERVICE, WRITER)

    # Nothing to inherit, and the task is failed rather than half-provisioned.
    assert row.identities == []
    assert task.state == "FAILED"
    assert outcome.task_state == "FAILED"


async def test_a_sealed_row_is_neither_re_grounded_nor_written_again() -> None:
    """The gate is also why a re-drive cannot accumulate a second write.

    ``_drive`` grounds only while the column is empty, so a row that already holds a
    sealed identity is reused verbatim: no call to the resolver, no write, nothing
    appended. This is the same single-gate property seen from the replay side, and it is
    why "which identity rows exist" has one answer per attempt rather than one per retry.

    The run still ends at the later gate, so the assertion is about the absence of the
    call and the write, not about the log being empty.
    """
    # Re-drive the same row with the listener reattached, so a second grounding or a
    # second write would be observed. It has to be the *same* session: the row is
    # attached to it, and a fresh one would find the instance detached.
    with _observe() as replay:
        async with session_scope() as session:
            outcome, row, task, transport = await _drive_in_session(
                session, replay, resolvable=True
            )
            try:
                sealed = list(row.identities)
                assert sealed, "the first drive must have sealed an identity"
                assert [k for k, _, _ in replay].count("write") == 1

                # The same session, the same row, a second claim: the sealed column is
                # reused rather than re-derived. The task is already FAILED, so the second
                # attempt has no plan and cannot transition again - that exception is the
                # expected shape, and the events are what this test is about.
                replay.clear()
                gateway, _ = make_gateway()
                runtime = AgentRuntime(gateway=gateway, model=_UnreachableModel())
                with pytest.raises((PlanNotFoundError, InvalidTransition)):
                    await runtime._drive(
                        session, task=task, row=row, transport=transport, claim=row.claim_token
                    )
                kinds = [kind for kind, _, _ in replay]
                assert "write" not in kinds, f"a sealed row was written again: {replay}"
                assert "call" not in kinds, f"a sealed row was re-grounded: {replay}"
                assert row.identities == sealed, "the sealed identity was disturbed"
                assert outcome.task_state == "FAILED"
            finally:
                await transport.aclose()


# ---------------------------------------------------------------------------
# Part 3 - the round is technique-only, and this is what makes that checkable
# ---------------------------------------------------------------------------


def test_this_round_changed_no_semantics() -> None:
    """The frozen list, asserted rather than promised.

    Every boundary this round was told not to touch, pinned at its current value. A future
    edit to this file accompanied by a change to any of these is not technique-only and
    does not belong in a round like this one.
    """
    from kai.identity.service import _has_conflict
    from kai.policy.models import IdentityStatus as PolicyIdentityStatus

    # Vocabulary: five states, none added, none renamed.
    assert {s.value for s in IdentityStatus} == {
        "CONTACT_MATCH", "PROVIDER_VERIFIED", "CHANNEL_VERIFIED", "UNKNOWN", "CONFLICTING",
    }
    assert {s.value for s in _POSITIVE_IDENTITY} == {
        "CONTACT_MATCH", "PROVIDER_VERIFIED", "CHANNEL_VERIFIED",
    }

    # Projection: total onto the policy's own states, and unchanged.
    assert {k.value: v.value for k, v in _IDENTITY_TO_POLICY.items()} == {
        "CONTACT_MATCH": "VERIFIED",
        "CHANNEL_VERIFIED": "VERIFIED",
        "PROVIDER_VERIFIED": "VERIFIED",
        "UNKNOWN": "UNKNOWN",
        "CONFLICTING": "CONFLICTING",
    }
    assert all(isinstance(v, PolicyIdentityStatus) for v in _IDENTITY_TO_POLICY.values())

    # Gmail stays unlicensed for live sealing; the local, provider-free
    # task_preparation profile is the only other live-sealable question.
    assert frozenset({"calendar_meeting", "task_preparation"}) == LIVE_SEALABLE_PROFILES

    # The identity round itself added no migration. Later rounds added
    # ``d4e5f6a7b8c9_add_revoked_tokens``, ``a1b2c3d4e5f7_add_alerts`` and
    # ``e6f7a8b9c0d1_add_alert_outbox``, and this round adds
    # ``a7c9e1f3b5d2_add_intent_attempts`` (the O1-B INTENT attempt record); none
    # of them touches an identity column. The chain is asserted by
    # :func:`_delivery_chain` rather than by counting files, so the guard holds in
    # a clean delivery tree and still rejects an unexpected migration.
    chain = _delivery_chain()
    assert len(chain) == DELIVERY_CHAIN_LENGTH, f"unexpected deliverable chain: {chain}"
    assert chain[0] == DELIVERY_HEAD, f"deliverable chain head is not declared: {chain[0]}"

    # The conflict rule is the same object, untouched: the round changed no semantics, so
    # it is referenced rather than restated.
    assert callable(_has_conflict)
    assert Path(inspect.getsourcefile(_has_conflict) or "").as_posix().endswith(
        "kai/identity/service.py"
    )

    # No reader learned to check status. The writers own the property, by design.
    provenance = (REPO / "tests" / "test_intent_provenance.py").read_text(encoding="utf-8")
    assert "test_how_an_identity_was_established_is_not_part_of_the_criteria" in provenance


def test_the_single_writer_claim_is_not_a_property_of_this_test_file() -> None:
    """The lock must hold for the repository, not for the file that asserts it.

    A scan that only looked at itself would pass forever. The scan is rooted at ``src/``
    and ``alembic/`` and reads every ``.py`` beneath them, so this test restates the
    property from outside: the file asserting it is not itself in scope, the scan reaches
    the whole package, and the single writer it finds is a statement on the class the
    runtime actually instantiates - with no stage before ``_drive`` touching the column.
    """
    files = scanned_files()
    assert Path(__file__).name not in {p.name for p in files}, (
        "the scan must not include the test file, or the property would be self-referential"
    )
    assert len(files) > 50, f"the scan only reached {len(files)} files, which is implausible"

    source = textwrap.dedent(inspect.getsource(AgentRuntime))
    assert WRITER in source
    before_drive = source.split("async def _drive", 1)[0]
    assert ".identities" not in before_drive, "no stage before _drive may touch the column"
