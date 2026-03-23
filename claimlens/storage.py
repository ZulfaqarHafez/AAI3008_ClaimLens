import json
from datetime import datetime
from typing import Optional

import redis
from sqlalchemy import create_engine, text

from .config import settings


redis_client = None
engine = None


def init_redis():
    global redis_client
    if redis_client is None and settings.REDIS_HOST:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            ssl=settings.REDIS_SSL,
            decode_responses=True,
        )
    return redis_client


def init_postgres():
    global engine
    if engine is None and all([
        settings.DB_HOST,
        settings.DB_NAME,
        settings.DB_USER,
        settings.DB_PASSWORD,
    ]):
        url = (
            f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        engine = create_engine(url, pool_pre_ping=True)
    return engine


def create_tables():
    eng = init_postgres()
    if eng is None:
        return

    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS verification_reports (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                original_text TEXT NOT NULL,
                overall_trust_score FLOAT,
                summary TEXT,
                report_json JSONB NOT NULL
            )
        """))


def save_report(report):
    eng = init_postgres()
    if eng is None:
        return

    payload = json.loads(report.model_dump_json())

    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO verification_reports
                (id, created_at, original_text, overall_trust_score, summary, report_json)
                VALUES (:id, :created_at, :original_text, :overall_trust_score, :summary, CAST(:report_json AS JSONB))
                ON CONFLICT (id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    original_text = EXCLUDED.original_text,
                    overall_trust_score = EXCLUDED.overall_trust_score,
                    summary = EXCLUDED.summary,
                    report_json = EXCLUDED.report_json
            """),
            {
                "id": report.id,
                "created_at": report.created_at,
                "original_text": report.original_text,
                "overall_trust_score": report.overall_trust_score,
                "summary": report.summary,
                "report_json": json.dumps(payload),
            },
        )


def save_job(job_id: str, job_data: dict):
    r = init_redis()
    if r is None:
        return
    r.set(f"job:{job_id}", json.dumps(job_data, default=str), ex=86400)


def get_job(job_id: str) -> Optional[dict]:
    r = init_redis()
    if r is None:
        return None
    raw = r.get(f"job:{job_id}")
    if not raw:
        return None
    return json.loads(raw)


def delete_job(job_id: str):
    r = init_redis()
    if r is None:
        return
    r.delete(f"job:{job_id}")


def check_rate_limit(client_ip: str, limit: int, window: int) -> bool:
    r = init_redis()
    if r is None:
        return True

    key = f"ratelimit:{client_ip}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, window)
    return current <= limit