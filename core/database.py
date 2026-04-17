from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from core.config import DATABASE_URL
from core.models import Base

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _maybe_migrate() -> None:
    """Migrate a pre-multi-portfolio database to the new schema."""
    with engine.connect() as conn:
        has_portfolios = conn.execute(
            text(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name='portfolios'"
            )
        ).scalar()
        if has_portfolios:
            return  # already on new schema

        has_holdings = conn.execute(
            text(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name='holdings'"
            )
        ).scalar()
        if not has_holdings:
            return  # fresh DB — create_all handles everything

        # Legacy holdings table exists without portfolio support.
        # Create portfolios table and slot existing holdings into "Default".
        conn.execute(
            text(
                "CREATE TABLE portfolios "
                "(id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL, "
                "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        conn.execute(
            text("INSERT INTO portfolios (id, name) VALUES (1, 'Default')")
        )
        cols = [
            r[1]
            for r in conn.execute(text("PRAGMA table_info(holdings)")).fetchall()
        ]
        if "portfolio_id" not in cols:
            conn.execute(
                text("ALTER TABLE holdings ADD COLUMN portfolio_id INTEGER DEFAULT 1")
            )
        conn.commit()


def _ensure_default_portfolio() -> None:
    """Guarantee at least one portfolio exists after schema creation."""
    db = SessionLocal()
    try:
        from core.models import Portfolio
        if not db.query(Portfolio).first():
            db.add(Portfolio(name="Default"))
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
