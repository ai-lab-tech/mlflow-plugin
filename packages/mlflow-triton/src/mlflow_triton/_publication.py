"""Durable operation receipts and cooperative per-Registry serialization."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    MetaData,
    String,
    Table,
    create_engine,
    select,
    text,
)
from sqlalchemy.pool import NullPool

metadata = MetaData()
receipts = Table(
    "triton_serving_publications_v1",
    metadata,
    Column("operation", String(64), primary_key=True),
    Column("payload", JSON, nullable=False),
)


class AmbiguousPublication(RuntimeError):
    """A write may have succeeded; retry reconciliation, never blindly recreate."""


class PublicationStore:
    def __init__(self, url):
        if not url:
            raise ValueError(
                "Configure MLFLOW_TRITON_PUBLICATION_DATABASE_URL (shared PostgreSQL for multiple hosts)"
            )
        self.engine = create_engine(url, poolclass=NullPool)
        if self.engine.dialect.name not in {"sqlite", "postgresql"}:
            raise ValueError("Publication receipts require PostgreSQL or local SQLite")
        if self.engine.dialect.name == "sqlite":
            database = self.engine.url.database
            if not database or database == ":memory:":
                raise ValueError("Publication receipts require a durable file")
            self.lock_root = Path(database).absolute().parent / (
                Path(database).name + ".locks"
            )
            self.lock_root.mkdir(parents=True, exist_ok=True)
        with self.lock("schema-v1"):
            metadata.create_all(self.engine)

    @contextmanager
    def lock(self, scope):
        key = hashlib.sha256(scope.encode()).digest()
        if self.engine.dialect.name == "postgresql":
            number = int.from_bytes(key[:8], "big", signed=True)
            # NullPool closes the session and releases its advisory lock on exit.
            with self.engine.connect() as connection:
                connection.execute(
                    text("SELECT pg_advisory_lock(:key)"), {"key": number}
                )
                connection.commit()
                yield
        else:
            import fcntl

            with (self.lock_root / key.hex()).open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def get(self, operation):
        with self.engine.connect() as connection:
            return connection.execute(
                select(receipts.c.payload).where(receipts.c.operation == operation)
            ).scalar_one_or_none()

    def save(self, operation, payload):
        with self.engine.begin() as connection:
            if self.get(operation) is None:
                connection.execute(
                    receipts.insert().values(operation=operation, payload=payload)
                )
            else:
                connection.execute(
                    receipts.update()
                    .where(receipts.c.operation == operation)
                    .values(payload=payload)
                )
