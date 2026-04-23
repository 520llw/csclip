"""
Enterprise-level SQLite database for BALF Cell Annotation System.
Manages projects, sessions, audit logs, annotation versions, and statistics.
"""

import os
import sqlite3
import json
import threading
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

_DB_PATH = os.path.join(os.path.dirname(__file__), "annotation_system.db")
_local = threading.local()

SCHEMA_VERSION = 1


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


@contextmanager
def get_db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );

        -- Projects: top-level organization
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            config TEXT DEFAULT '{}',
            annotation_guide TEXT DEFAULT '',
            category_schema TEXT DEFAULT '[]'
        );

        -- Dataset-project associations
        CREATE TABLE IF NOT EXISTS project_datasets (
            project_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            added_at TEXT NOT NULL,
            role TEXT DEFAULT 'primary',
            PRIMARY KEY (project_id, group_id),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        -- Annotation versions: track every save
        CREATE TABLE IF NOT EXISTS annotation_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            label_set_id TEXT NOT NULL,
            subset TEXT NOT NULL,
            filename TEXT NOT NULL,
            version_num INTEGER NOT NULL,
            annotations_json TEXT NOT NULL,
            annotation_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            created_by TEXT DEFAULT 'system',
            change_summary TEXT DEFAULT '',
            checksum TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_ann_versions_lookup
            ON annotation_versions(group_id, label_set_id, subset, filename);

        -- Audit log: all user actions
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            user_id TEXT DEFAULT 'default',
            details TEXT DEFAULT '{}',
            group_id TEXT DEFAULT '',
            filename TEXT DEFAULT '',
            ip_address TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
        CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_log(category);

        -- Session tracking
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'default',
            started_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            group_id TEXT DEFAULT '',
            label_set_id TEXT DEFAULT '',
            subset TEXT DEFAULT 'train',
            current_image TEXT DEFAULT '',
            current_image_index INTEGER DEFAULT -1,
            annotations_saved INTEGER DEFAULT 0,
            images_viewed INTEGER DEFAULT 0,
            session_data TEXT DEFAULT '{}'
        );

        -- Daily statistics
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT NOT NULL,
            user_id TEXT DEFAULT 'default',
            group_id TEXT DEFAULT '',
            images_annotated INTEGER DEFAULT 0,
            annotations_created INTEGER DEFAULT 0,
            annotations_modified INTEGER DEFAULT 0,
            annotations_deleted INTEGER DEFAULT 0,
            total_time_seconds INTEGER DEFAULT 0,
            ai_assists_used INTEGER DEFAULT 0,
            PRIMARY KEY (date, user_id, group_id)
        );

        -- Tag system for datasets
        CREATE TABLE IF NOT EXISTS dataset_tags (
            group_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (group_id, tag)
        );

        -- Annotation flags (problem images, review status)
        CREATE TABLE IF NOT EXISTS image_flags (
            group_id TEXT NOT NULL,
            label_set_id TEXT NOT NULL,
            subset TEXT NOT NULL,
            filename TEXT NOT NULL,
            flag_type TEXT NOT NULL,
            flag_value TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            created_by TEXT DEFAULT 'default',
            PRIMARY KEY (group_id, label_set_id, subset, filename, flag_type)
        );

        CREATE INDEX IF NOT EXISTS idx_flags_lookup
            ON image_flags(group_id, label_set_id, subset);

        -- Export history
        CREATE TABLE IF NOT EXISTS export_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            label_set_id TEXT DEFAULT '',
            export_format TEXT NOT NULL,
            export_path TEXT DEFAULT '',
            image_count INTEGER DEFAULT 0,
            annotation_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'completed',
            config TEXT DEFAULT '{}'
        );

        -- User preferences
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preferences TEXT DEFAULT '{}',
            keyboard_shortcuts TEXT DEFAULT '{}',
            ui_state TEXT DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        """
        )

        existing = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )


# ── Audit Logging ──────────────────────────────────────────────


def log_action(
    action: str,
    category: str = "general",
    user_id: str = "default",
    details: dict = None,
    group_id: str = "",
    filename: str = "",
    ip_address: str = "",
):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO audit_log (timestamp, action, category, user_id, details, group_id, filename, ip_address)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                action,
                category,
                user_id,
                json.dumps(details or {}, ensure_ascii=False),
                group_id,
                filename,
                ip_address,
            ),
        )


def get_audit_log(
    limit: int = 100,
    offset: int = 0,
    category: str = None,
    action: str = None,
    start_date: str = None,
    end_date: str = None,
) -> List[Dict]:
    with get_db() as conn:
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if action:
            query += " AND action = ?"
            params.append(action)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ── Annotation Versions ────────────────────────────────────────


def save_annotation_version(
    group_id: str,
    label_set_id: str,
    subset: str,
    filename: str,
    annotations: list,
    user_id: str = "default",
    change_summary: str = "",
):
    ann_json = json.dumps(annotations, ensure_ascii=False)
    checksum = hashlib.md5(ann_json.encode()).hexdigest()

    with get_db() as conn:
        last = conn.execute(
            """SELECT version_num, checksum FROM annotation_versions
               WHERE group_id=? AND label_set_id=? AND subset=? AND filename=?
               ORDER BY version_num DESC LIMIT 1""",
            (group_id, label_set_id, subset, filename),
        ).fetchone()

        if last and last["checksum"] == checksum:
            return None

        version_num = (last["version_num"] + 1) if last else 1

        conn.execute(
            """INSERT INTO annotation_versions
               (group_id, label_set_id, subset, filename, version_num,
                annotations_json, annotation_count, created_at, created_by,
                change_summary, checksum)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                group_id,
                label_set_id,
                subset,
                filename,
                version_num,
                ann_json,
                len(annotations),
                datetime.now().isoformat(),
                user_id,
                change_summary,
                checksum,
            ),
        )

        old_versions = conn.execute(
            """SELECT id FROM annotation_versions
               WHERE group_id=? AND label_set_id=? AND subset=? AND filename=?
               ORDER BY version_num DESC""",
            (group_id, label_set_id, subset, filename),
        ).fetchall()
        if len(old_versions) > 50:
            for old in old_versions[50:]:
                conn.execute(
                    "DELETE FROM annotation_versions WHERE id=?", (old["id"],)
                )

        return version_num


def get_annotation_versions(
    group_id: str,
    label_set_id: str,
    subset: str,
    filename: str,
    limit: int = 20,
) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, version_num, annotation_count, created_at, created_by,
                      change_summary
               FROM annotation_versions
               WHERE group_id=? AND label_set_id=? AND subset=? AND filename=?
               ORDER BY version_num DESC LIMIT ?""",
            (group_id, label_set_id, subset, filename, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_annotation_version_data(version_id: int) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM annotation_versions WHERE id=?", (version_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["annotations"] = json.loads(d["annotations_json"])
            del d["annotations_json"]
            return d
        return None


# ── Session Management ─────────────────────────────────────────


def create_or_update_session(
    session_id: str,
    group_id: str = "",
    label_set_id: str = "",
    subset: str = "train",
    current_image: str = "",
    current_image_index: int = -1,
) -> Dict:
    now = datetime.now().isoformat()
    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE sessions SET last_active=?, group_id=?, label_set_id=?,
                   subset=?, current_image=?, current_image_index=?
                   WHERE id=?""",
                (
                    now,
                    group_id,
                    label_set_id,
                    subset,
                    current_image,
                    current_image_index,
                    session_id,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO sessions (id, started_at, last_active, group_id,
                   label_set_id, subset, current_image, current_image_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    now,
                    now,
                    group_id,
                    label_set_id,
                    subset,
                    current_image,
                    current_image_index,
                ),
            )
        row = conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else {}


def get_session(session_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def increment_session_stats(session_id: str, saved: int = 0, viewed: int = 0):
    with get_db() as conn:
        conn.execute(
            """UPDATE sessions SET
               annotations_saved = annotations_saved + ?,
               images_viewed = images_viewed + ?,
               last_active = ?
               WHERE id = ?""",
            (saved, viewed, datetime.now().isoformat(), session_id),
        )


# ── Daily Statistics ────────────────────────────────────────────


def record_daily_stat(
    group_id: str = "",
    user_id: str = "default",
    images_annotated: int = 0,
    annotations_created: int = 0,
    annotations_modified: int = 0,
    annotations_deleted: int = 0,
    ai_assists: int = 0,
):
    today = datetime.now().strftime("%Y-%m-%d")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO daily_stats
               (date, user_id, group_id, images_annotated, annotations_created,
                annotations_modified, annotations_deleted, ai_assists_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(date, user_id, group_id) DO UPDATE SET
                images_annotated = images_annotated + excluded.images_annotated,
                annotations_created = annotations_created + excluded.annotations_created,
                annotations_modified = annotations_modified + excluded.annotations_modified,
                annotations_deleted = annotations_deleted + excluded.annotations_deleted,
                ai_assists_used = ai_assists_used + excluded.ai_assists_used""",
            (
                today,
                user_id,
                group_id,
                images_annotated,
                annotations_created,
                annotations_modified,
                annotations_deleted,
                ai_assists,
            ),
        )


def get_daily_stats(days: int = 30, group_id: str = None) -> List[Dict]:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_db() as conn:
        query = "SELECT * FROM daily_stats WHERE date >= ?"
        params = [start_date]
        if group_id:
            query += " AND group_id = ?"
            params.append(group_id)
        query += " ORDER BY date DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_stats_summary(days: int = 30, group_id: str = None) -> Dict:
    stats = get_daily_stats(days, group_id)
    total = {
        "images_annotated": sum(s["images_annotated"] for s in stats),
        "annotations_created": sum(s["annotations_created"] for s in stats),
        "annotations_modified": sum(s["annotations_modified"] for s in stats),
        "annotations_deleted": sum(s["annotations_deleted"] for s in stats),
        "ai_assists_used": sum(s["ai_assists_used"] for s in stats),
        "active_days": len(set(s["date"] for s in stats)),
    }
    if total["active_days"] > 0:
        total["avg_daily_images"] = round(
            total["images_annotated"] / total["active_days"], 1
        )
        total["avg_daily_annotations"] = round(
            total["annotations_created"] / total["active_days"], 1
        )
    else:
        total["avg_daily_images"] = 0
        total["avg_daily_annotations"] = 0
    return total


# ── Image Flags ──────────────────────────────────────────────


def set_image_flag(
    group_id: str,
    label_set_id: str,
    subset: str,
    filename: str,
    flag_type: str,
    flag_value: str = "",
    user_id: str = "default",
):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO image_flags (group_id, label_set_id, subset, filename,
               flag_type, flag_value, created_at, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(group_id, label_set_id, subset, filename, flag_type)
               DO UPDATE SET flag_value=excluded.flag_value, created_at=excluded.created_at""",
            (
                group_id,
                label_set_id,
                subset,
                filename,
                flag_type,
                flag_value,
                datetime.now().isoformat(),
                user_id,
            ),
        )


def remove_image_flag(
    group_id: str,
    label_set_id: str,
    subset: str,
    filename: str,
    flag_type: str,
):
    with get_db() as conn:
        conn.execute(
            """DELETE FROM image_flags
               WHERE group_id=? AND label_set_id=? AND subset=? AND filename=? AND flag_type=?""",
            (group_id, label_set_id, subset, filename, flag_type),
        )


def get_image_flags(
    group_id: str, label_set_id: str, subset: str, filename: str = None
) -> List[Dict]:
    with get_db() as conn:
        if filename:
            rows = conn.execute(
                """SELECT * FROM image_flags
                   WHERE group_id=? AND label_set_id=? AND subset=? AND filename=?""",
                (group_id, label_set_id, subset, filename),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM image_flags
                   WHERE group_id=? AND label_set_id=? AND subset=?""",
                (group_id, label_set_id, subset),
            ).fetchall()
        return [dict(r) for r in rows]


# ── User Preferences ──────────────────────────────────────────


def get_user_preferences(user_id: str = "default") -> Dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM user_preferences WHERE user_id=?", (user_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["preferences"] = json.loads(d["preferences"])
            d["keyboard_shortcuts"] = json.loads(d["keyboard_shortcuts"])
            d["ui_state"] = json.loads(d["ui_state"])
            return d
        return {
            "user_id": user_id,
            "preferences": {},
            "keyboard_shortcuts": {},
            "ui_state": {},
        }


def save_user_preferences(
    user_id: str = "default",
    preferences: dict = None,
    keyboard_shortcuts: dict = None,
    ui_state: dict = None,
):
    now = datetime.now().isoformat()
    current = get_user_preferences(user_id)
    if preferences is not None:
        current["preferences"] = preferences
    if keyboard_shortcuts is not None:
        current["keyboard_shortcuts"] = keyboard_shortcuts
    if ui_state is not None:
        current["ui_state"] = ui_state

    with get_db() as conn:
        conn.execute(
            """INSERT INTO user_preferences (user_id, preferences, keyboard_shortcuts, ui_state, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                preferences=excluded.preferences,
                keyboard_shortcuts=excluded.keyboard_shortcuts,
                ui_state=excluded.ui_state,
                updated_at=excluded.updated_at""",
            (
                user_id,
                json.dumps(current["preferences"], ensure_ascii=False),
                json.dumps(current["keyboard_shortcuts"], ensure_ascii=False),
                json.dumps(current["ui_state"], ensure_ascii=False),
                now,
            ),
        )


# ── Export History ──────────────────────────────────────────────


def record_export(
    group_id: str,
    label_set_id: str,
    export_format: str,
    export_path: str = "",
    image_count: int = 0,
    annotation_count: int = 0,
    config: dict = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO export_history
               (group_id, label_set_id, export_format, export_path, image_count,
                annotation_count, created_at, config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                group_id,
                label_set_id,
                export_format,
                export_path,
                image_count,
                annotation_count,
                datetime.now().isoformat(),
                json.dumps(config or {}, ensure_ascii=False),
            ),
        )
        return cursor.lastrowid


def get_export_history(group_id: str = None, limit: int = 50) -> List[Dict]:
    with get_db() as conn:
        if group_id:
            rows = conn.execute(
                """SELECT * FROM export_history WHERE group_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (group_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM export_history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ── Project Management ──────────────────────────────────────────


def create_project(
    name: str, description: str = "", config: dict = None
) -> Dict:
    import uuid as _uuid

    project_id = str(_uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO projects (id, name, description, created_at, updated_at, config)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                name,
                description,
                now,
                now,
                json.dumps(config or {}, ensure_ascii=False),
            ),
        )
    return {
        "id": project_id,
        "name": name,
        "description": description,
        "created_at": now,
        "status": "active",
    }


def list_projects() -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"])
            datasets = conn.execute(
                "SELECT group_id FROM project_datasets WHERE project_id=?", (d["id"],)
            ).fetchall()
            d["dataset_count"] = len(datasets)
            d["dataset_ids"] = [ds["group_id"] for ds in datasets]
            result.append(d)
        return result


def get_project(project_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = json.loads(d["config"])
        datasets = conn.execute(
            "SELECT * FROM project_datasets WHERE project_id=?", (project_id,)
        ).fetchall()
        d["datasets"] = [dict(ds) for ds in datasets]
        return d


def update_project(project_id: str, updates: Dict) -> Optional[Dict]:
    with get_db() as conn:
        fields = []
        values = []
        for key in ["name", "description", "status", "annotation_guide"]:
            if key in updates:
                fields.append(f"{key}=?")
                values.append(updates[key])
        if "config" in updates:
            fields.append("config=?")
            values.append(json.dumps(updates["config"], ensure_ascii=False))
        if not fields:
            return get_project(project_id)
        fields.append("updated_at=?")
        values.append(datetime.now().isoformat())
        values.append(project_id)
        conn.execute(
            f"UPDATE projects SET {', '.join(fields)} WHERE id=?", values
        )
    return get_project(project_id)


def add_dataset_to_project(project_id: str, group_id: str, role: str = "primary"):
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO project_datasets (project_id, group_id, added_at, role)
               VALUES (?, ?, ?, ?)""",
            (project_id, group_id, datetime.now().isoformat(), role),
        )


def remove_dataset_from_project(project_id: str, group_id: str):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM project_datasets WHERE project_id=? AND group_id=?",
            (project_id, group_id),
        )


# ── Dataset Tags ──────────────────────────────────────────────


def add_dataset_tag(group_id: str, tag: str):
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO dataset_tags (group_id, tag, created_at)
               VALUES (?, ?, ?)""",
            (group_id, tag, datetime.now().isoformat()),
        )


def remove_dataset_tag(group_id: str, tag: str):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM dataset_tags WHERE group_id=? AND tag=?", (group_id, tag)
        )


def get_dataset_tags(group_id: str) -> List[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT tag FROM dataset_tags WHERE group_id=? ORDER BY tag",
            (group_id,),
        ).fetchall()
        return [r["tag"] for r in rows]


def get_all_tags() -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT tag, COUNT(*) as count FROM dataset_tags
               GROUP BY tag ORDER BY count DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


# Initialize database on import
init_db()
