"""Session-scoped storage for F3 externalized tool results."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ToolArtifact:
    result_id: str
    session_id: str
    tool_name: str
    content: str
    content_type: str
    sha256: str
    byte_count: int
    token_count: int
    created_at: float


class ArtifactStore(Protocol):
    def put(
        self,
        *,
        session_id: str,
        tool_name: str,
        content: str,
        content_type: str,
        token_count: int,
    ) -> ToolArtifact: ...

    def get(self, session_id: str, result_id: str) -> ToolArtifact | None: ...

    def delete(self, session_id: str, result_id: str) -> None: ...

    def close(self) -> None: ...


def _new_artifact(
    *,
    session_id: str,
    tool_name: str,
    content: str,
    content_type: str,
    token_count: int,
) -> ToolArtifact:
    raw = content.encode("utf-8")
    return ToolArtifact(
        result_id=f"tr_{secrets.token_urlsafe(12)}",
        session_id=session_id,
        tool_name=tool_name,
        content=content,
        content_type=content_type,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        token_count=token_count,
        created_at=time.time(),
    )


class MemoryArtifactStore:
    """Thread-safe volatile backend used by unit tests and dry runs."""

    def __init__(self, *, ttl_seconds: float = 3600.0):
        self.ttl_seconds = float(ttl_seconds)
        self._items: dict[tuple[str, str], ToolArtifact] = {}
        self._lock = threading.RLock()

    def put(
        self,
        *,
        session_id: str,
        tool_name: str,
        content: str,
        content_type: str,
        token_count: int,
    ) -> ToolArtifact:
        artifact = _new_artifact(
            session_id=session_id,
            tool_name=tool_name,
            content=content,
            content_type=content_type,
            token_count=token_count,
        )
        with self._lock:
            self._items[(session_id, artifact.result_id)] = artifact
        return artifact

    def get(self, session_id: str, result_id: str) -> ToolArtifact | None:
        with self._lock:
            artifact = self._items.get((session_id, result_id))
            if artifact is None:
                return None
            if self.ttl_seconds > 0 and time.time() - artifact.created_at > self.ttl_seconds:
                self._items.pop((session_id, result_id), None)
                return None
            return artifact

    def delete(self, session_id: str, result_id: str) -> None:
        with self._lock:
            self._items.pop((session_id, result_id), None)

    def close(self) -> None:
        return None


class SQLiteArtifactStore:
    """SQLite backend with WAL and session-scoped lookups."""

    def __init__(self, path: str, *, ttl_seconds: float = 3600.0):
        if not path:
            raise ValueError("SQLite artifact store requires a path")
        self.path = str(Path(path))
        self.ttl_seconds = float(ttl_seconds)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_artifacts (
                session_id TEXT NOT NULL,
                result_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (session_id, result_id)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_artifacts_created_at "
            "ON tool_artifacts(created_at)"
        )
        self._conn.commit()

    def put(
        self,
        *,
        session_id: str,
        tool_name: str,
        content: str,
        content_type: str,
        token_count: int,
    ) -> ToolArtifact:
        artifact = _new_artifact(
            session_id=session_id,
            tool_name=tool_name,
            content=content,
            content_type=content_type,
            token_count=token_count,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tool_artifacts (
                    session_id, result_id, tool_name, content, content_type,
                    sha256, byte_count, token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.session_id,
                    artifact.result_id,
                    artifact.tool_name,
                    artifact.content,
                    artifact.content_type,
                    artifact.sha256,
                    artifact.byte_count,
                    artifact.token_count,
                    artifact.created_at,
                ),
            )
            self._cleanup_expired_locked()
            self._conn.commit()
        return artifact

    def get(self, session_id: str, result_id: str) -> ToolArtifact | None:
        cutoff = time.time() - self.ttl_seconds if self.ttl_seconds > 0 else None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT result_id, session_id, tool_name, content, content_type,
                       sha256, byte_count, token_count, created_at
                FROM tool_artifacts
                WHERE session_id = ? AND result_id = ?
                """,
                (session_id, result_id),
            ).fetchone()
            if row is None:
                return None
            if cutoff is not None and float(row[8]) < cutoff:
                self._conn.execute(
                    "DELETE FROM tool_artifacts WHERE session_id = ? AND result_id = ?",
                    (session_id, result_id),
                )
                self._conn.commit()
                return None
        return ToolArtifact(*row)

    def delete(self, session_id: str, result_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM tool_artifacts WHERE session_id = ? AND result_id = ?",
                (session_id, result_id),
            )
            self._conn.commit()

    def _cleanup_expired_locked(self) -> None:
        if self.ttl_seconds <= 0:
            return
        self._conn.execute(
            "DELETE FROM tool_artifacts WHERE created_at < ?",
            (time.time() - self.ttl_seconds,),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def build_artifact_store(
    kind: str,
    *,
    path: str = "",
    ttl_seconds: float = 3600.0,
) -> ArtifactStore:
    if kind == "memory":
        return MemoryArtifactStore(ttl_seconds=ttl_seconds)
    if kind == "sqlite":
        return SQLiteArtifactStore(path, ttl_seconds=ttl_seconds)
    raise ValueError("artifact store must be 'memory' or 'sqlite'")
