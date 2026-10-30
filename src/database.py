"""
SQLite Database Storage Module for Prediction History and Analytics.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions.db")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create a thread-safe connection with row dictionary mapping."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database tables with indexes."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities_json TEXT NOT NULL,
            explanation TEXT NOT NULL,
            top_features_json TEXT,
            source TEXT DEFAULT 'single',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
    """)

    conn.commit()
    conn.close()


def insert_prediction(
    input_text: str,
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float],
    explanation: str,
    top_features: Optional[List[Dict[str, Any]]] = None,
    source: str = "single",
    db_path: str = DB_PATH
) -> int:
    """Insert a new prediction record into SQLite."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            input_text, predicted_class, confidence, probabilities_json, 
            explanation, top_features_json, source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        input_text,
        predicted_class,
        float(confidence),
        json.dumps(probabilities),
        explanation,
        json.dumps(top_features or []),
        source,
        datetime.utcnow().isoformat()
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_history(
    limit: int = 50,
    offset: int = 0,
    search_query: Optional[str] = None,
    class_filter: Optional[str] = None,
    db_path: str = DB_PATH
) -> Tuple[List[Dict[str, Any]], int]:
    """Retrieve filtered predictions history with pagination."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
    params: List[Any] = []

    if class_filter and class_filter.lower() != "all":
        query += " AND LOWER(predicted_class) = LOWER(?)"
        count_query += " AND LOWER(predicted_class) = LOWER(?)"
        params.append(class_filter.strip())

    if search_query:
        query += " AND (input_text LIKE ? OR explanation LIKE ?)"
        count_query += " AND (input_text LIKE ? OR explanation LIKE ?)"
        wildcard = f"%{search_query.strip()}%"
        params.extend([wildcard, wildcard])

    # Count total matching rows
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]

    # Order by newest first
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    query_params = list(params) + [limit, offset]
    cursor.execute(query, query_params)
    rows = cursor.fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "input_text": r["input_text"],
            "predicted_class": r["predicted_class"],
            "confidence": round(r["confidence"], 4),
            "probabilities": json.loads(r["probabilities_json"]) if r["probabilities_json"] else {},
            "explanation": r["explanation"],
            "top_features": json.loads(r["top_features_json"]) if r["top_features_json"] else [],
            "source": r["source"],
            "created_at": r["created_at"]
        })

    conn.close()
    return results, total_count


def get_statistics(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Calculate aggregated dashboard metrics from stored predictions."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_class = 'Sarcastic'")
    sarcastic = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_class = 'Ironical'")
    ironical = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_class IN ('Non-Sarcastic / Literal', 'Literal')")
    literal = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cursor.fetchone()[0] or 0.0

    conn.close()

    sarcastic_pct = round((sarcastic / total * 100), 1) if total > 0 else 0.0
    ironical_pct = round((ironical / total * 100), 1) if total > 0 else 0.0
    literal_pct = round((literal / total * 100), 1) if total > 0 else 0.0

    return {
        "total_predictions": total,
        "sarcastic_count": sarcastic,
        "ironical_count": ironical,
        "literal_count": literal,
        "sarcastic_pct": sarcastic_pct,
        "ironical_pct": ironical_pct,
        "literal_pct": literal_pct,
        "avg_confidence": round(avg_conf * 100, 1)
    }


def clear_history(db_path: str = DB_PATH) -> int:
    """Clear all records from predictions table."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_rows


def export_all_predictions(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Export all prediction records for CSV download."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, input_text, predicted_class, confidence, source, created_at FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
