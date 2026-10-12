"""Persistence: SQLite for the session registry, Parquet+DuckDB for the facts.

Split deliberately, matching the two data shapes:

* Sessions, experiments and adapter health are low-volume, transactional, and
  read while a collector writes -> SQLite in WAL mode (concurrent readers, one
  writer) is the right tool.
* Events and telemetry samples are high-volume, append-only, and analysed in
  bulk -> Parquet on disk (source of truth) queried through DuckDB views.

The API opens DuckDB *read-only over the Parquet globs* in an in-memory catalog,
so a live collector appending Parquet and the API reading never fight over a
single-writer lock — the same trick that kept Fathom's API and ingest apart.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import Event, Sample

SCHEMA_VERSION = 1

# Explicit Arrow schemas so every part file matches and a column that is null in
# one session does not drift the type of the union.
_EVENTS_SCHEMA = pa.schema([
    ("schema_version", pa.int32()), ("event_id", pa.string()), ("session_id", pa.string()),
    ("kind", pa.string()), ("source", pa.string()), ("provenance", pa.string()),
    ("ts_wall_ns", pa.int64()), ("ts_mono_ns", pa.int64()),
    ("engine", pa.string()), ("model_id", pa.string()),
    ("request_id", pa.string()), ("step_idx", pa.int64()), ("token_idx", pa.int64()),
    ("component", pa.string()), ("layer_idx", pa.int64()), ("op_id", pa.string()),
    ("start_ns", pa.int64()), ("end_ns", pa.int64()), ("duration_us", pa.float64()),
    ("unit", pa.string()), ("device", pa.string()), ("pid", pa.int64()),
    ("bytes_moved", pa.int64()), ("bytes_alloc", pa.int64()),
    ("tensor_json", pa.string()), ("value", pa.float64()), ("confidence", pa.float64()),
    ("native_ref", pa.string()), ("clock_note", pa.string()), ("reason", pa.string()),
    ("extra_json", pa.string()),
])
_EVENT_COLS = [f.name for f in _EVENTS_SCHEMA]

_SAMPLES_SCHEMA = pa.schema([
    ("schema_version", pa.int32()), ("session_id", pa.string()),
    ("ts_wall_ns", pa.int64()), ("ts_mono_ns", pa.int64()),
    ("source", pa.string()), ("provenance", pa.string()),
    ("gpu_util_pct", pa.float64()), ("mem_used_pct", pa.float64()),
    ("vram_used_bytes", pa.int64()), ("vram_total_bytes", pa.int64()),
    ("power_w", pa.float64()), ("temp_c", pa.float64()),
    ("sclk_mhz", pa.float64()), ("mclk_mhz", pa.float64()), ("encode_util_pct", pa.float64()),
    ("cpu_util_pct", pa.float64()), ("load1", pa.float64()), ("load5", pa.float64()),
    ("load15", pa.float64()), ("ram_used_bytes", pa.int64()), ("ram_total_bytes", pa.int64()),
    ("proc_rss_bytes", pa.int64()), ("active_requests", pa.int64()), ("queue_depth", pa.int64()),
    ("tokens_per_s", pa.float64()), ("per_core_json", pa.string()), ("extra_json", pa.string()),
    ("reason", pa.string()),
])
_SAMPLE_COLS = [f.name for f in _SAMPLES_SCHEMA]

_ARCH_SCHEMA = pa.schema([
    ("session_id", pa.string()), ("model_id", pa.string()),
    ("graph_json", pa.string()), ("provenance", pa.string()),
    ("source", pa.string()), ("updated_ns", pa.int64()),
])


def _j(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v, sort_keys=True)


class TraceStore:
    """Parquet source of truth + DuckDB read layer."""

    def __init__(self, root: Path, read_only: bool = False):
        self.root = Path(root)
        self.pq_root = self.root / "parquet"
        self.pq_root.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self._tls = threading.local()
        self.con = duckdb.connect(":memory:")
        self._register()

    def _dir(self, table: str, session_id: str) -> Path:
        d = self.pq_root / table / f"session_id={session_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _glob(self, table: str) -> str:
        return str(self.pq_root / table / "session_id=*" / "*.parquet")

    @staticmethod
    def _duck_type(arrow_type: pa.DataType) -> str:
        return {
            pa.int32(): "INTEGER", pa.int64(): "BIGINT",
            pa.float64(): "DOUBLE", pa.string(): "VARCHAR",
        }.get(arrow_type, "VARCHAR")

    def _empty_view(self, view: str, schema: pa.Schema) -> None:
        sel = ", ".join(
            f"CAST(NULL AS {self._duck_type(f.type)}) AS {f.name}" for f in schema)
        self.con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT {sel} LIMIT 0")

    @staticmethod
    def _readable(paths: list) -> list:
        """Keep only intact Parquet files.

        A process killed mid-flush can leave a truncated part file. One such
        file must not poison the whole catalog, so validate the magic header
        and footer before handing the list to DuckDB.
        """
        good = []
        for p in paths:
            try:
                with open(p, "rb") as fh:
                    head = fh.read(4)
                    fh.seek(-4, 2)
                    foot = fh.read(4)
                if head == b"PAR1" and foot == b"PAR1":
                    good.append(str(p))
            except OSError:
                continue
        return good

    def _register(self) -> None:
        import glob as _glob
        for table, schema, view in (
            ("events", _EVENTS_SCHEMA, "pq_events"),
            ("samples", _SAMPLES_SCHEMA, "pq_samples"),
            ("architecture", _ARCH_SCHEMA, "pq_architecture"),
        ):
            cols = [f.name for f in schema]
            files = self._readable(sorted(_glob.glob(self._glob(table))))
            if files:
                listed = ", ".join(f"'{f}'" for f in files)
                self.con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS SELECT {', '.join(cols)} "
                    f"FROM read_parquet([{listed}], union_by_name=true)")
            else:
                self._empty_view(view, schema)

    def _append(self, table: str, schema: pa.Schema, rows: list[dict]) -> int:
        if not rows:
            return 0
        session_id = rows[0]["session_id"]
        path = self._dir(table, session_id) / "part-0.parquet"
        new = pa.Table.from_pylist(rows, schema=schema)
        if path.exists():
            old = pq.read_table(path)
            new = pa.concat_tables([old, new])
        pq.write_table(new, path)
        self._register()
        return len(rows)

    # ---------- writes ----------

    def append_events(self, events: Iterable[Event]) -> int:
        rows = []
        for e in events:
            d = e.model_dump(mode="json")
            row = {c: d.get(c) for c in _EVENT_COLS}
            row["tensor_json"] = _j(d.get("tensor"))
            row["extra_json"] = _j(d.get("extra"))
            rows.append(row)
        return self._append("events", _EVENTS_SCHEMA, rows)

    def append_samples(self, samples: Iterable[Sample]) -> int:
        rows = []
        for s in samples:
            d = s.model_dump(mode="json")
            row = {c: d.get(c) for c in _SAMPLE_COLS}
            row["per_core_json"] = _j(d.get("per_core"))
            row["extra_json"] = _j(d.get("extra"))
            rows.append(row)
        return self._append("samples", _SAMPLES_SCHEMA, rows)

    def set_architecture(self, session_id: str, model_id: str, graph: dict,
                         provenance: str, source: str) -> None:
        row = {
            "session_id": session_id, "model_id": model_id,
            "graph_json": json.dumps(graph, sort_keys=True),
            "provenance": provenance, "source": source,
            "updated_ns": time.time_ns(),
        }
        path = self._dir("architecture", session_id) / "part-0.parquet"
        pq.write_table(pa.Table.from_pylist([row], schema=_ARCH_SCHEMA), path)
        self._register()

    def reset_session(self, session_id: str) -> None:
        for table in ("events", "samples", "architecture"):
            d = self.pq_root / table / f"session_id={session_id}"
            if d.exists():
                for f in d.glob("*.parquet"):
                    f.unlink()
                try:
                    d.rmdir()
                except OSError:
                    pass
        self._register()

    # ---------- reads ----------

    def _cur(self):
        cur = getattr(self._tls, "cur", None)
        if cur is None:
            cur = self.con.cursor()
            self._tls.cur = cur
        return cur

    def query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cur = self._cur().execute(sql, params or [])
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def refresh(self) -> None:
        """Re-scan the Parquet globs so a reader sees files written after startup."""
        self._register()

    def close(self) -> None:
        self.con.close()


class SessionStore:
    """SQLite registry: sessions, experiments, adapter health snapshots."""

    def __init__(self, root: Path):
        self.path = Path(root) / "inferscope.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self) -> None:
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY, kind TEXT, engine TEXT, model_id TEXT,
              started_ns INTEGER, ended_ns INTEGER, config_json TEXT,
              capability_json TEXT, source TEXT, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS experiments (
              id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, kind TEXT,
              baseline_session TEXT, candidate_session TEXT, config_json TEXT,
              result_json TEXT, created_ns INTEGER
            );
            CREATE TABLE IF NOT EXISTS adapter_health (
              session_id TEXT, adapter TEXT, ts_ns INTEGER, available INTEGER,
              detail TEXT, PRIMARY KEY (session_id, adapter)
            );
            """
        )
        self.con.commit()

    def create_session(self, session_id: str, kind: str, engine: str = "",
                       model_id: str = "", config: dict | None = None,
                       capability: dict | None = None, source: str = "live",
                       notes: str = "") -> None:
        with self._lock:
            self.con.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (session_id, kind, engine, model_id, time.time_ns(), None,
                 json.dumps(config or {}), json.dumps(capability or {}), source, notes))
            self.con.commit()

    def end_session(self, session_id: str) -> None:
        with self._lock:
            self.con.execute("UPDATE sessions SET ended_ns=? WHERE session_id=?",
                             (time.time_ns(), session_id))
            self.con.commit()

    def sessions(self) -> list[dict]:
        rows = self.con.execute(
            "SELECT * FROM sessions ORDER BY started_ns DESC").fetchall()
        return [dict(r) for r in rows]

    def session(self, session_id: str) -> Optional[dict]:
        r = self.con.execute("SELECT * FROM sessions WHERE session_id=?",
                             (session_id,)).fetchone()
        return dict(r) if r else None

    def record_health(self, session_id: str, adapter: str, available: bool,
                      detail: str) -> None:
        with self._lock:
            self.con.execute(
                "INSERT OR REPLACE INTO adapter_health VALUES (?,?,?,?)",
                (session_id, adapter, time.time_ns(), 1 if available else 0))
            self.con.commit()

    def add_experiment(self, name: str, kind: str, baseline: str, candidate: str,
                       config: dict, result: dict) -> int:
        with self._lock:
            cur = self.con.execute(
                "INSERT INTO experiments (name,kind,baseline_session,candidate_session,"
                "config_json,result_json,created_ns) VALUES (?,?,?,?,?,?,?)",
                (name, kind, baseline, candidate, json.dumps(config),
                 json.dumps(result), time.time_ns()))
            self.con.commit()
            return int(cur.lastrowid)

    def experiments(self) -> list[dict]:
        rows = self.con.execute("SELECT * FROM experiments ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def set_kind(self, session_id: str, kind: str) -> None:
        """Reclassify a session in place (e.g. live -> profile)."""
        with self._lock:
            self.con.execute("UPDATE sessions SET kind=? WHERE session_id=?",
                             (kind, session_id))
            self.con.commit()

    def superseded_sessions(self, kind: str, keep: str) -> list[str]:
        """Sessions of one kind that a newer session has replaced.

        A restart always opens a fresh live session, so the previous ones hold
        telemetry that nothing can still be appended to. They are dead weight
        in the picker, where a dozen identical "live" rows hide the one session
        that actually has execution events in it.
        """
        rows = self.con.execute(
            "SELECT session_id FROM sessions WHERE kind=? AND session_id<>?",
            (kind, keep)).fetchall()
        return [r["session_id"] for r in rows]

    def prune_sessions(self, keep: int) -> list[str]:
        """Return session_ids beyond the retention window (caller deletes data)."""
        rows = self.con.execute(
            "SELECT session_id FROM sessions ORDER BY started_ns DESC LIMIT -1 OFFSET ?",
            (keep,)).fetchall()
        return [r["session_id"] for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self.con.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            self.con.execute("DELETE FROM adapter_health WHERE session_id=?", (session_id,))
            self.con.commit()

    def close(self) -> None:
        self.con.close()
