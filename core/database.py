from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from core.config import DATABASE_URL
from core.models import Base
from core.persistence import schedule_db_push

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(SessionLocal, "after_commit")
def _persist_after_commit(session) -> None:
    schedule_db_push()


def _maybe_migrate() -> None:
    """Add missing columns to existing holdings table without data loss."""
    from sqlalchemy import inspect
    insp = inspect(engine)
    if "holdings" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("holdings")}
    with engine.begin() as conn:
        if "shares" not in cols:
            conn.execute(text("ALTER TABLE holdings ADD COLUMN shares REAL"))
        if "purchase_price" not in cols:
            conn.execute(text("ALTER TABLE holdings ADD COLUMN purchase_price REAL"))

    if "portfolio_allocations" in insp.get_table_names():
        alloc_cols = {c["name"] for c in insp.get_columns("portfolio_allocations")}
        with engine.begin() as conn:
            if "frontier_json" not in alloc_cols:
                conn.execute(text("ALTER TABLE portfolio_allocations ADD COLUMN frontier_json TEXT DEFAULT '[]'"))


def _ensure_default_portfolio() -> None:
    """Guarantee at least one portfolio exists after schema creation."""
    db = SessionLocal()
    try:
        from core.models import PortfolioDB
        if not db.query(PortfolioDB).first():
            db.add(PortfolioDB(name="Default"))
            db.commit()
    finally:
        db.close()


def init_db() -> None:
    _maybe_migrate()
    Base.metadata.create_all(bind=engine)
    _ensure_default_portfolio()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
