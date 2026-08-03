"""
proctor_db.py
Database layer for the AI Exam Proctoring System.
SQLite-backed persistence for students, exam sessions, alerts, and trust scores.
No external dependencies beyond the Python standard library.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proctor.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exams (
    exam_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    duration_sec    INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    student_id      TEXT NOT NULL,
    exam_id         TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    trust_score     INTEGER NOT NULL DEFAULT 100,
    verdict         TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (exam_id) REFERENCES exams(exam_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    message         TEXT,
    offset_sec      INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id);
"""

PENALTIES = {
    "no_face": 8,
    "multiple_faces": 12,
    "looking_away": 5,
    "camera_tamper": 15,
    "tab_switch": 10,
}


def _now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------- Students ----------------------------------

def upsert_student(student_id, name, email=None, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO students (student_id, name, email, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(student_id) DO UPDATE SET name=excluded.name, email=excluded.email""",
            (student_id, name, email, _now()),
        )


def get_student(student_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------- Exams ----------------------------------

def upsert_exam(exam_id, title, duration_sec, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO exams (exam_id, title, duration_sec, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(exam_id) DO UPDATE SET title=excluded.title, duration_sec=excluded.duration_sec""",
            (exam_id, title, duration_sec, _now()),
        )


# ---------------------------------- Sessions ----------------------------------

def start_session(session_id, student_id, exam_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO sessions (session_id, student_id, exam_id, started_at, trust_score, status)
               VALUES (?, ?, ?, ?, 100, 'active')""",
            (session_id, student_id, exam_id, _now()),
        )


def end_session(session_id, db_path=DB_PATH):
    score = get_trust_score(session_id, db_path)
    verdict = _verdict_from_score(score)
    with get_conn(db_path) as conn:
        conn.execute(
            """UPDATE sessions SET ended_at = ?, verdict = ?, status = 'completed'
               WHERE session_id = ?""",
            (_now(), verdict, session_id),
        )
    return verdict


def _verdict_from_score(score):
    if score >= 80:
        return "pass"
    if score >= 50:
        return "review"
    return "fail"


def get_session(session_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def list_sessions(student_id=None, status=None, db_path=DB_PATH):
    query = "SELECT * FROM sessions WHERE 1=1"
    params = []
    if student_id:
        query += " AND student_id = ?"
        params.append(student_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY started_at DESC"
    with get_conn(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------- Alerts ----------------------------------

def log_alert(session_id, alert_type, message="", severity=None, offset_sec=0, db_path=DB_PATH):
    """Insert an alert and apply its penalty to the session's trust score."""
    severity = severity or _default_severity(alert_type)
    penalty = PENALTIES.get(alert_type, 5)

    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO alerts (session_id, alert_type, severity, message, offset_sec, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, alert_type, severity, message, offset_sec, _now()),
        )
        conn.execute(
            """UPDATE sessions SET trust_score = MAX(0, trust_score - ?)
               WHERE session_id = ?""",
            (penalty, session_id),
        )

    return get_trust_score(session_id, db_path)


def _default_severity(alert_type):
    return {
        "no_face": "high",
        "multiple_faces": "high",
        "camera_tamper": "high",
        "looking_away": "medium",
        "tab_switch": "medium",
    }.get(alert_type, "low")


def get_trust_score(session_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT trust_score FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["trust_score"] if row else None


def list_alerts(session_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE session_id = ? ORDER BY offset_sec ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def alert_counts_by_severity(session_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT severity, COUNT(*) as count FROM alerts
               WHERE session_id = ? GROUP BY severity""",
            (session_id,),
        ).fetchall()
        return {r["severity"]: r["count"] for r in rows}


# ---------------------------------- Reporting ----------------------------------

def session_report(session_id, db_path=DB_PATH):
    """Full report payload: session + student + exam + alerts + verdict."""
    with get_conn(db_path) as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not session:
            return None
        session = dict(session)

        student = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (session["student_id"],)
        ).fetchone()
        exam = conn.execute(
            "SELECT * FROM exams WHERE exam_id = ?", (session["exam_id"],)
        ).fetchone()
        alerts = conn.execute(
            "SELECT * FROM alerts WHERE session_id = ? ORDER BY offset_sec ASC",
            (session_id,),
        ).fetchall()

    return {
        "session": session,
        "student": dict(student) if student else None,
        "exam": dict(exam) if exam else None,
        "alerts": [dict(a) for a in alerts],
        "verdict": session.get("verdict") or _verdict_from_score(session["trust_score"]),
    }


def export_alerts_csv(session_id, filepath, db_path=DB_PATH):
    import csv
    alerts = list_alerts(session_id, db_path)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["alert_id", "session_id", "alert_type", "severity", "message", "offset_sec", "created_at"]
        )
        writer.writeheader()
        writer.writerows(alerts)
    return filepath


# ---------------------------------- CLI smoke test ----------------------------------

if __name__ == "__main__":
    init_db()

    upsert_student("STU-20481", "Aditi Sharma", "aditi.sharma@example.edu")
    upsert_exam("EXAM-DS-MID", "Data Structures — Midterm", duration_sec=3600)

    sid = "SESSION-" + datetime.now().strftime("%Y%m%d%H%M%S")
    start_session(sid, "STU-20481", "EXAM-DS-MID")

    log_alert(sid, "no_face", "No face detected — camera obstructed", offset_sec=134)
    log_alert(sid, "looking_away", "Gaze deviated from screen", offset_sec=561)
    log_alert(sid, "tab_switch", "Window focus lost", offset_sec=943)

    verdict = end_session(sid)

    report = session_report(sid)
    print(f"Session {sid} — verdict: {verdict}")
    print(f"Trust score: {report['session']['trust_score']}")
    print(f"Alerts logged: {len(report['alerts'])}")
    print(f"Severity breakdown: {alert_counts_by_severity(sid)}")