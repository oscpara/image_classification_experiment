from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import Float, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import get_database_url
from src.paths import PREDICTIONS_LOG_PATH


DEFAULT_PREDICTIONS_LOG_PATH = PREDICTIONS_LOG_PATH


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class Prediction(Base):
    """Prediction metadata with source-log fields for audit traceability."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "source_log_path",
            "source_line_number",
            name="uq_predictions_source_line",
        ),
    )

    prediction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    uploaded_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    predicted_breed: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    prediction_timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_log_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_raw_json: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[str] = mapped_column(String(64), nullable=False)


REQUIRED_FIELDS = {
    "id",
    "predicted_breed",
    "confidence",
    "model_version",
    "timestamp",
    "image_path",
}


def create_schema(database_url: str) -> None:
    """Create database tables if they do not already exist."""

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)


def _validate_record(record: dict[str, Any], line_number: int) -> None:
    """Fail fast when a JSONL row cannot produce a complete database row."""

    missing_fields = sorted(REQUIRED_FIELDS - record.keys())
    if missing_fields:
        joined_fields = ", ".join(missing_fields)
        raise ValueError(f"Line {line_number} is missing required fields: {joined_fields}")


def _prediction_from_record(
    record: dict[str, Any],
    source_log_path: Path,
    source_line_number: int,
    source_raw_json: str,
    ingested_at: str,
) -> Prediction:
    """Convert one prediction log record into a database model."""

    return Prediction(
        prediction_id=record["id"],
        uploaded_filename=record.get("uploaded_filename"),
        predicted_breed=record["predicted_breed"],
        confidence=float(record["confidence"]),
        model_version=record["model_version"],
        prediction_timestamp=record["timestamp"],
        image_path=record["image_path"],
        source_log_path=str(source_log_path),
        source_line_number=source_line_number,
        source_raw_json=source_raw_json,
        ingested_at=ingested_at,
    )


def _prediction_exists(session, prediction: Prediction) -> bool:
    """Check both stable ids: prediction id and source log position."""

    existing_prediction = session.scalar(
        select(Prediction.prediction_id).where(
            (Prediction.prediction_id == prediction.prediction_id)
            | (
                (Prediction.source_log_path == prediction.source_log_path)
                & (Prediction.source_line_number == prediction.source_line_number)
            )
        )
    )
    return existing_prediction is not None


def _insert_prediction(session, prediction: Prediction) -> bool:
    """Insert a prediction once; return False when it was already loaded."""

    if _prediction_exists(session, prediction):
        return False

    session.add(prediction)
    return True


def ingest_predictions(predictions_log_path: Path, database_url: str) -> int:
    """Read prediction JSONL rows and load new rows into the database."""

    if not predictions_log_path.exists():
        raise FileNotFoundError(f"Predictions log not found: {predictions_log_path}")

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    source_log_path = predictions_log_path.resolve()
    ingested_count = 0

    with session_factory() as session:
        with predictions_log_path.open("r", encoding="utf-8") as predictions_log:
            for line_number, line in enumerate(predictions_log, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line_number} is not valid JSON") from exc

                _validate_record(record, line_number)
                prediction = _prediction_from_record(
                    record=record,
                    source_log_path=source_log_path,
                    source_line_number=line_number,
                    source_raw_json=raw_line,
                    ingested_at=datetime.now(timezone.utc).isoformat(),
                )
                if _insert_prediction(session, prediction):
                    ingested_count += 1

        session.commit()

    return ingested_count


def parse_args() -> argparse.Namespace:
    """Parse command line options for the ingestion job."""

    parser = argparse.ArgumentParser(
        description="Load prediction metadata from predictions.jsonl into PostgreSQL."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_PREDICTIONS_LOG_PATH,
        help="Path to predictions.jsonl.",
    )
    parser.add_argument(
        "--database-url",
        default=get_database_url(),
        help="SQLAlchemy database URL. Defaults to DATABASE_URL, then config.ini.",
    )
    return parser.parse_args()


def _mask_database_url(database_url: str) -> str:
    """Hide credentials when reporting the target database."""

    parts = urlsplit(database_url)
    if not parts.password:
        return database_url

    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    username = parts.username or ""
    netloc = f"{username}:***@{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def main() -> None:
    """Run the ingestion job from the command line."""

    args = parse_args()
    if not args.database_url:
        raise SystemExit("Set DATABASE_URL, set [database].url in config.ini, or pass --database-url.")

    ingested_count = ingest_predictions(args.source, args.database_url)
    print(f"Ingested {ingested_count} new prediction rows into {_mask_database_url(args.database_url)}")
