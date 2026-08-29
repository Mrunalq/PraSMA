"""
PraSMA SQLite Database Layer
Independent of Streamlit.
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().with_name("prasma_database.db")

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            loan_amount REAL NOT NULL CHECK (loan_amount > 0),
            interest_rate REAL NOT NULL CHECK (interest_rate >= 0),
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            tenure_months INTEGER NOT NULL CHECK (tenure_months > 0),
            emi_due REAL NOT NULL CHECK (emi_due > 0),
            archetype TEXT,
            sub_pattern TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (end_date >= start_date)
        );

        CREATE TABLE IF NOT EXISTS payments (
            payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            month_date TEXT NOT NULL,
            amount_paid REAL NOT NULL CHECK (amount_paid >= 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
            UNIQUE (account_id, month_date)
        );

        CREATE INDEX IF NOT EXISTS idx_payments_account ON payments(account_id);
        CREATE INDEX IF NOT EXISTS idx_payments_month ON payments(month_date);
        """)

def create_account(account_id, loan_amount, interest_rate, start_date,
                   end_date, tenure_months, emi_due,
                   archetype=None, sub_pattern=None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO accounts
            (account_id, loan_amount, interest_rate, start_date, end_date,
             tenure_months, emi_due, archetype, sub_pattern)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (account_id, loan_amount, interest_rate, str(start_date),
              str(end_date), tenure_months, emi_due, archetype, sub_pattern))

def get_account(account_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
        ).fetchone()
        return dict(row) if row else None

def get_all_accounts():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM accounts ORDER BY account_id").fetchall()
        return [dict(row) for row in rows]

def add_payment(account_id, month_date, amount_paid):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO payments (account_id, month_date, amount_paid)
            VALUES (?, ?, ?)
        """, (account_id, str(month_date), amount_paid))

def get_payments(account_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT payment_id, account_id, month_date, amount_paid, created_at
            FROM payments WHERE account_id = ? ORDER BY month_date
        """, (account_id,)).fetchall()
        return [dict(row) for row in rows]

def delete_account(account_id):
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM accounts WHERE account_id = ?", (account_id,))
        return cur.rowcount > 0

def update_account(account_id, **fields):
    allowed = {
        "loan_amount", "interest_rate", "start_date", "end_date",
        "tenure_months", "emi_due", "archetype", "sub_pattern"
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unsupported fields: {sorted(unknown)}")
    if not fields:
        return False

    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = [str(v) if key in {"start_date", "end_date"} else v
              for key, v in fields.items()]
    values.append(account_id)

    with get_connection() as conn:
        cur = conn.execute(
            f"UPDATE accounts SET {assignments} WHERE account_id = ?", values
        )
        return cur.rowcount > 0

def get_database_stats():
    with get_connection() as conn:
        result = {}
        for table in ("accounts", "payments"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            result[table] = {
                "rows": count,
                "columns": len(cols),
                "column_names": [r["name"] for r in cols]
            }
        return result

init_db()
