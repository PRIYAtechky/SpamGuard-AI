"""
SpamGuard AI - SQLite Database Manager
Handles prediction audit history, aggregate statistics, and privacy-preserving sanitized previews.
"""

import os
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

DB_FILENAME = "spam_guard.db"


def get_db_path(custom_path: Optional[str] = None) -> str:
    if custom_path:
        return custom_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DB_FILENAME)


def sanitize_preview(text: str, max_length: int = 140) -> str:
    """
    Sanitize and truncate email text for privacy and storage efficiency.
    Removes HTML tags, linebreaks, and collapse whitespace.
    """
    if not text:
        return ""
    # Strip basic HTML tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Collapse multiple whitespaces and line breaks into a single space
    clean = re.sub(r'\s+', ' ', clean).strip()
    if len(clean) > max_length:
        return clean[:max_length].rstrip() + "..."
    return clean


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = get_db_path(db_path)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes."""
    path = get_db_path(db_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with get_connection(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_preview TEXT NOT NULL,
                prediction TEXT NOT NULL CHECK(prediction IN ('ham', 'spam')),
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_pred 
            ON predictions(prediction)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created 
            ON predictions(created_at DESC)
        """)


def insert_prediction(message: str, prediction: str, confidence: float, db_path: Optional[str] = None) -> int:
    """
    Insert a sanitized prediction record.
    Returns the newly created row ID.
    """
    preview = sanitize_preview(message)
    pred_clean = prediction.lower().strip()
    if pred_clean not in ('ham', 'spam'):
        raise ValueError(f"Invalid prediction label: {prediction}")

    path = get_db_path(db_path)
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (message_preview, prediction, confidence, created_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (preview, pred_clean, float(confidence))
        )
        return cursor.lastrowid


def get_history(
    limit: int = 50,
    offset: int = 0,
    filter_type: Optional[str] = None,
    search_term: Optional[str] = None,
    db_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Retrieve paginated prediction records with optional class filtering and search.
    Returns (items, total_count).
    """
    path = get_db_path(db_path)
    where_clauses = []
    params = []

    if filter_type and filter_type.lower() in ('ham', 'spam'):
        where_clauses.append("prediction = ?")
        params.append(filter_type.lower())

    if search_term and search_term.strip():
        where_clauses.append("message_preview LIKE ?")
        params.append(f"%{search_term.strip()}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with get_connection(path) as conn:
        cursor = conn.cursor()

        # Count total matching
        count_query = f"SELECT COUNT(*) FROM predictions {where_sql}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        # Fetch page
        query = f"""
            SELECT id, message_preview, prediction, confidence, created_at
            FROM predictions
            {where_sql}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()

        items = [
            {
                "id": row["id"],
                "message_preview": row["message_preview"],
                "prediction": row["prediction"],
                "confidence": round(row["confidence"], 4),
                "created_at": row["created_at"]
            }
            for row in rows
        ]

        return items, total_count


def get_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute aggregate statistics from SQLite."""
    path = get_db_path(db_path)
    with get_connection(path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'spam'")
        spam_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'ham'")
        ham_count = cursor.fetchone()[0]

        # Activity by date (last 14 days)
        cursor.execute("""
            SELECT 
                date(created_at) as log_date,
                COUNT(*) as total,
                SUM(CASE WHEN prediction = 'spam' THEN 1 ELSE 0 END) as spam,
                SUM(CASE WHEN prediction = 'ham' THEN 1 ELSE 0 END) as ham
            FROM predictions
            GROUP BY date(created_at)
            ORDER BY log_date ASC
            LIMIT 14
        """)
        activity = [
            {
                "date": row["log_date"],
                "total": row["total"],
                "spam": row["spam"] or 0,
                "ham": row["ham"] or 0
            }
            for row in cursor.fetchall()
        ]

    spam_pct = round((spam_count / total * 100), 2) if total > 0 else 0.0
    ham_pct = round((ham_count / total * 100), 2) if total > 0 else 0.0

    return {
        "total": total,
        "spam": spam_count,
        "ham": ham_count,
        "spam_percentage": spam_pct,
        "ham_percentage": ham_pct,
        "activity_by_date": activity
    }


def clear_history(db_path: Optional[str] = None) -> None:
    """Clear all prediction records and reset autoincrement."""
    path = get_db_path(db_path)
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        try:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        except sqlite3.OperationalError:
            pass
        conn.commit()


# Initialize upon module import
init_db()
