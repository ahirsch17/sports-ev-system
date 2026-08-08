from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Base


def create_db_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    connect_args: dict = {}
    if settings.database_url.startswith("postgresql"):
        connect_args["connect_timeout"] = 5
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args=connect_args or {},
    )


def init_db(engine: Engine | None = None, settings: Settings | None = None) -> None:
    engine = engine or create_db_engine(settings)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(
    engine: Engine | None = None, settings: Settings | None = None
) -> Generator[Session, None, None]:
    engine = engine or create_db_engine(settings)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
