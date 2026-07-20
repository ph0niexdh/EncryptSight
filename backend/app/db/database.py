import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("encryptsight.db")

DATABASE_URL = os.getenv("DATABASE_URL")
engine = None

if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
    try:
        # Try creating PostgreSQL engine with a short timeout to prevent blocking startup
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 3})
        # Test connection
        with engine.connect() as conn:
            pass
        logger.info("Connected to PostgreSQL database successfully.")
    except Exception as e:
        logger.warning(f"Failed to connect to PostgreSQL database ({DATABASE_URL}): {e}. Falling back to SQLite.")
        engine = None

if engine is None:
    # Local SQLite database path. Placing it in the workspace folder.
    db_path = os.path.join(os.getcwd(), "encryptsight.db")
    DATABASE_URL = f"sqlite:///{db_path}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    logger.info(f"Using SQLite database at {db_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
