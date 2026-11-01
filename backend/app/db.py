from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models import Base

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
elif settings.database_url.startswith("postgresql"):
    @event.listens_for(engine, "connect")
    def _register_pgvector(dbapi_connection, _connection_record) -> None:
        from pgvector.psycopg2 import register_vector

        try:
            register_vector(dbapi_connection)
        except Exception:
            # The first local startup may create the extension after opening
            # this connection; init_db registers it again after CREATE EXTENSION.
            dbapi_connection.rollback()


def init_db() -> None:
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            from pgvector.psycopg2 import register_vector

            register_vector(connection.connection.driver_connection)
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
