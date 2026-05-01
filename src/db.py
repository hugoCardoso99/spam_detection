"""
db.py – PostgreSQL persistence layer for raw messages and labels.

Handles schema creation, message ingestion, and data retrieval.
Expects a running Postgres instance (e.g., via docker-compose).
"""

import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

CONN_KWARGS = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "spam_detection"),
    "user": os.getenv("POSTGRES_USER", "spam_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "spam_password"),
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(128) NOT NULL,
    message TEXT NOT NULL,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_dataset ON messages(dataset_name);
"""


def _connect():
    """Return a new connection to the Postgres database."""
    return psycopg2.connect(**CONN_KWARGS)


def init_db():
    """Create the messages table and index if they don't exist."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        print("Database schema initialised.")
    finally:
        conn.close()


def insert_messages(dataset_name, messages, labels):
    """Bulk-insert raw messages + labels into Postgres.

    Parameters
    ----------
    dataset_name : str
        Namespaces rows per dataset (e.g., 'sms', 'enron').
    messages : iterable of str
        Raw text messages.
    labels : iterable of int
        0=ham, 1=spam.
    """
    if len(messages) != len(labels):
        raise ValueError("messages and labels must have the same length")

    conn = _connect()
    try:
        with conn.cursor() as cur:
            data = [(dataset_name, msg, lbl) for msg, lbl in zip(messages, labels)]
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO messages (dataset_name, message, label) VALUES %s",
                data,
            )
        conn.commit()
        print("Inserted %d rows into 'messages' (dataset=%s)." % (len(data), dataset_name))
    finally:
        conn.close()


def load_dataset(dataset_name):
    """Load all messages + labels for a given dataset.

    Returns
    -------
    list of dict with keys: message, label
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT message, label FROM messages WHERE dataset_name = %s",
                (dataset_name,),
            )
            rows = cur.fetchall()
            return [{"message": msg, "label": lbl} for msg, lbl in rows]
    finally:
        conn.close()


def count_messages(dataset_name):
    """Return the number of rows for a dataset."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM messages WHERE dataset_name = %s",
                (dataset_name,),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()
