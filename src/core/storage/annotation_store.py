from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aflutils.logger import get_logger
from core.models.annotation_models import (
    ClarityScore,
    ExpressionIntensity,
    FacialFeatures,
    FacialMotion,
    Transcription,
)

logger = get_logger(__name__)


AnnotationModel = Union[
    FacialFeatures,
    FacialMotion,
    ClarityScore,
    ExpressionIntensity,
    Transcription,
    Dict[str, Any],
]


class AnnotationStore:
    """SQLite storage for Milestone 2 annotation records and label outputs."""

    def __init__(self, db_path: str = "data/annotations.db"):
        self.db_path = db_path
        self._memory_conn: Optional[sqlite3.Connection] = None
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def close(self) -> None:
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(self.db_path, timeout=30)
                self._memory_conn.row_factory = sqlite3.Row
                self._memory_conn.execute("PRAGMA foreign_keys=ON")
            return self._memory_conn

        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    label_1_status TEXT DEFAULT 'pending',
                    label_2_status TEXT DEFAULT 'pending',
                    label_3_status TEXT DEFAULT 'pending',
                    label_4_status TEXT DEFAULT 'pending',
                    whisper_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS label_1_facial_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    age INTEGER,
                    race TEXT,
                    hair_color TEXT,
                    confidence_json TEXT,
                    raw_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform),
                    FOREIGN KEY (video_id, platform) REFERENCES annotations(video_id, platform)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS label_2_facial_motion (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    description TEXT,
                    key_movements_json TEXT,
                    duration_category TEXT,
                    raw_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform),
                    FOREIGN KEY (video_id, platform) REFERENCES annotations(video_id, platform)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS label_3_clarity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    mean_clarity REAL,
                    median_clarity REAL,
                    std_clarity REAL,
                    min_clarity REAL,
                    max_clarity REAL,
                    face_detected_ratio REAL,
                    per_frame_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform),
                    FOREIGN KEY (video_id, platform) REFERENCES annotations(video_id, platform)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS label_4_expression (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    intensity INTEGER CHECK(intensity >= 1 AND intensity <= 5),
                    rationale TEXT,
                    dominant_expressions_json TEXT,
                    raw_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform),
                    FOREIGN KEY (video_id, platform) REFERENCES annotations(video_id, platform)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    full_text TEXT,
                    language TEXT,
                    segments_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform),
                    FOREIGN KEY (video_id, platform) REFERENCES annotations(video_id, platform)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_status
                ON annotations(status, updated_at)
            """)
        logger.info(f"AnnotationStore initialized at {self.db_path}")

    @staticmethod
    def _dump_json(value: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _load_json(value: Optional[str], default: Any = None) -> Any:
        if value is None:
            return default
        return json.loads(value)

    @staticmethod
    def _model_dump(model: AnnotationModel) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return dict(model)

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return dict(row) if row else None

    def create_annotation(
        self,
        video_id: str,
        platform: str,
        status: str = "pending",
        error_message: Optional[str] = None,
    ) -> None:
        """Create or refresh the parent annotation record."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO annotations (
                    video_id, platform, status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(video_id, platform) DO UPDATE SET
                    status = excluded.status,
                    error_message = excluded.error_message,
                    updated_at = CURRENT_TIMESTAMP
            """, (video_id, platform, status, error_message))

    def get_annotation(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT *
                FROM annotations
                WHERE video_id = ? AND platform = ?
            """, (video_id, platform)).fetchone()
        return self._row_to_dict(row)

    def update_annotation_status(
        self,
        video_id: str,
        platform: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute("""
                UPDATE annotations
                SET status = ?,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = ? AND platform = ?
            """, (status, error_message, video_id, platform))

    def update_label_status(
        self,
        video_id: str,
        platform: str,
        label: str,
        status: str,
    ) -> None:
        columns = {
            "label_1": "label_1_status",
            "label_1_status": "label_1_status",
            "label_2": "label_2_status",
            "label_2_status": "label_2_status",
            "label_3": "label_3_status",
            "label_3_status": "label_3_status",
            "label_4": "label_4_status",
            "label_4_status": "label_4_status",
            "whisper": "whisper_status",
            "whisper_status": "whisper_status",
        }
        if label not in columns:
            raise ValueError(f"Unknown label status column: {label}")
        column = columns[label]

        with self._connect() as conn:
            conn.execute(f"""
                UPDATE annotations
                SET {column} = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = ? AND platform = ?
            """, (status, video_id, platform))

    def save_facial_features(
        self,
        video_id: str,
        platform: str,
        features: Union[FacialFeatures, Dict[str, Any]],
        raw_response: Any = None,
    ) -> None:
        data = self._model_dump(features)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO label_1_facial_features (
                    video_id, platform, age, race, hair_color, confidence_json, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, platform) DO UPDATE SET
                    age = excluded.age,
                    race = excluded.race,
                    hair_color = excluded.hair_color,
                    confidence_json = excluded.confidence_json,
                    raw_response = excluded.raw_response
            """, (
                video_id,
                platform,
                data.get("age"),
                data.get("race"),
                data.get("hair_color"),
                self._dump_json(data.get("confidence")),
                self._dump_json(raw_response),
            ))

    def get_facial_features(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT *
                FROM label_1_facial_features
                WHERE video_id = ? AND platform = ?
            """, (video_id, platform)).fetchone()
        if row is None:
            return None
        return {
            "age": row["age"],
            "race": row["race"],
            "hair_color": row["hair_color"],
            "confidence": self._load_json(row["confidence_json"], {}),
            "raw_response": self._load_json(row["raw_response"]),
        }

    def save_facial_motion(
        self,
        video_id: str,
        platform: str,
        motion: Union[FacialMotion, Dict[str, Any]],
        raw_response: Any = None,
    ) -> None:
        data = self._model_dump(motion)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO label_2_facial_motion (
                    video_id, platform, description, key_movements_json,
                    duration_category, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, platform) DO UPDATE SET
                    description = excluded.description,
                    key_movements_json = excluded.key_movements_json,
                    duration_category = excluded.duration_category,
                    raw_response = excluded.raw_response
            """, (
                video_id,
                platform,
                data.get("description"),
                self._dump_json(data.get("key_movements")),
                data.get("duration_category"),
                self._dump_json(raw_response),
            ))

    def get_facial_motion(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT *
                FROM label_2_facial_motion
                WHERE video_id = ? AND platform = ?
            """, (video_id, platform)).fetchone()
        if row is None:
            return None
        return {
            "description": row["description"],
            "key_movements": self._load_json(row["key_movements_json"], []),
            "duration_category": row["duration_category"],
            "raw_response": self._load_json(row["raw_response"]),
        }

    def save_clarity(
        self,
        video_id: str,
        platform: str,
        clarity: Union[ClarityScore, Dict[str, Any]],
    ) -> None:
        data = self._model_dump(clarity)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO label_3_clarity (
                    video_id, platform, mean_clarity, median_clarity, std_clarity,
                    min_clarity, max_clarity, face_detected_ratio, per_frame_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, platform) DO UPDATE SET
                    mean_clarity = excluded.mean_clarity,
                    median_clarity = excluded.median_clarity,
                    std_clarity = excluded.std_clarity,
                    min_clarity = excluded.min_clarity,
                    max_clarity = excluded.max_clarity,
                    face_detected_ratio = excluded.face_detected_ratio,
                    per_frame_json = excluded.per_frame_json
            """, (
                video_id,
                platform,
                data.get("mean_clarity"),
                data.get("median_clarity"),
                data.get("std_clarity"),
                data.get("min_clarity"),
                data.get("max_clarity"),
                data.get("face_detected_ratio"),
                self._dump_json(data.get("per_frame")),
            ))

    def get_clarity(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT *
                FROM label_3_clarity
                WHERE video_id = ? AND platform = ?
            """, (video_id, platform)).fetchone()
        if row is None:
            return None
        return {
            "mean_clarity": row["mean_clarity"],
            "median_clarity": row["median_clarity"],
            "std_clarity": row["std_clarity"],
            "min_clarity": row["min_clarity"],
            "max_clarity": row["max_clarity"],
            "face_detected_ratio": row["face_detected_ratio"],
            "per_frame": self._load_json(row["per_frame_json"]),
        }

    def save_expression(
        self,
        video_id: str,
        platform: str,
        expression: Union[ExpressionIntensity, Dict[str, Any]],
        raw_response: Any = None,
    ) -> None:
        data = self._model_dump(expression)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO label_4_expression (
                    video_id, platform, intensity, rationale,
                    dominant_expressions_json, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, platform) DO UPDATE SET
                    intensity = excluded.intensity,
                    rationale = excluded.rationale,
                    dominant_expressions_json = excluded.dominant_expressions_json,
                    raw_response = excluded.raw_response
            """, (
                video_id,
                platform,
                data.get("intensity"),
                data.get("rationale"),
                self._dump_json(data.get("dominant_expressions")),
                self._dump_json(raw_response),
            ))

    def get_expression(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT *
                FROM label_4_expression
                WHERE video_id = ? AND platform = ?
            """, (video_id, platform)).fetchone()
        if row is None:
            return None
        return {
            "intensity": row["intensity"],
            "rationale": row["rationale"],
            "dominant_expressions": self._load_json(row["dominant_expressions_json"], []),
            "raw_response": self._load_json(row["raw_response"]),
        }

    def save_transcription(
        self,
        video_id: str,
        platform: str,
        transcription: Union[Transcription, Dict[str, Any]],
    ) -> None:
        data = self._model_dump(transcription)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO transcriptions (
                    video_id, platform, full_text, language, segments_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id, platform) DO UPDATE SET
                    full_text = excluded.full_text,
                    language = excluded.language,
                    segments_json = excluded.segments_json
            """, (
                video_id,
                platform,
                data.get("text", data.get("full_text")),
                data.get("language"),
                self._dump_json(data.get("segments")),
            ))

    def get_transcription(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT *
                FROM transcriptions
                WHERE video_id = ? AND platform = ?
            """, (video_id, platform)).fetchone()
        if row is None:
            return None
        return {
            "text": row["full_text"],
            "language": row["language"],
            "segments": self._load_json(row["segments_json"], []),
        }

    def list_pending_annotations(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_annotations_by_status("pending", limit)

    def list_completed_annotations(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_annotations_by_status("completed", limit)

    def _list_annotations_by_status(self, status: str, limit: int) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT *
                FROM annotations
                WHERE status = ?
                ORDER BY updated_at ASC, created_at ASC, id ASC
                LIMIT ?
            """, (status, limit)).fetchall()
        return [dict(row) for row in rows]

    def get_full_annotation(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    a.*,
                    ff.age AS ff_age,
                    ff.race AS ff_race,
                    ff.hair_color AS ff_hair_color,
                    ff.confidence_json AS ff_confidence_json,
                    ff.raw_response AS ff_raw_response,
                    fm.description AS fm_description,
                    fm.key_movements_json AS fm_key_movements_json,
                    fm.duration_category AS fm_duration_category,
                    fm.raw_response AS fm_raw_response,
                    cl.mean_clarity AS cl_mean_clarity,
                    cl.median_clarity AS cl_median_clarity,
                    cl.std_clarity AS cl_std_clarity,
                    cl.min_clarity AS cl_min_clarity,
                    cl.max_clarity AS cl_max_clarity,
                    cl.face_detected_ratio AS cl_face_detected_ratio,
                    cl.per_frame_json AS cl_per_frame_json,
                    ex.intensity AS ex_intensity,
                    ex.rationale AS ex_rationale,
                    ex.dominant_expressions_json AS ex_dominant_expressions_json,
                    ex.raw_response AS ex_raw_response,
                    tr.full_text AS tr_full_text,
                    tr.language AS tr_language,
                    tr.segments_json AS tr_segments_json
                FROM annotations a
                LEFT JOIN label_1_facial_features ff
                    ON a.video_id = ff.video_id AND a.platform = ff.platform
                LEFT JOIN label_2_facial_motion fm
                    ON a.video_id = fm.video_id AND a.platform = fm.platform
                LEFT JOIN label_3_clarity cl
                    ON a.video_id = cl.video_id AND a.platform = cl.platform
                LEFT JOIN label_4_expression ex
                    ON a.video_id = ex.video_id AND a.platform = ex.platform
                LEFT JOIN transcriptions tr
                    ON a.video_id = tr.video_id AND a.platform = tr.platform
                WHERE a.video_id = ? AND a.platform = ?
            """, (video_id, platform)).fetchone()
        if row is None:
            return None

        data = dict(row)
        annotation_keys = {
            "id",
            "video_id",
            "platform",
            "status",
            "label_1_status",
            "label_2_status",
            "label_3_status",
            "label_4_status",
            "whisper_status",
            "error_message",
            "created_at",
            "updated_at",
        }
        return {
            "annotation": {key: data[key] for key in annotation_keys},
            "facial_features": None if data["ff_age"] is None else {
                "age": data["ff_age"],
                "race": data["ff_race"],
                "hair_color": data["ff_hair_color"],
                "confidence": self._load_json(data["ff_confidence_json"], {}),
                "raw_response": self._load_json(data["ff_raw_response"]),
            },
            "facial_motion": None if data["fm_description"] is None else {
                "description": data["fm_description"],
                "key_movements": self._load_json(data["fm_key_movements_json"], []),
                "duration_category": data["fm_duration_category"],
                "raw_response": self._load_json(data["fm_raw_response"]),
            },
            "clarity": None if data["cl_mean_clarity"] is None else {
                "mean_clarity": data["cl_mean_clarity"],
                "median_clarity": data["cl_median_clarity"],
                "std_clarity": data["cl_std_clarity"],
                "min_clarity": data["cl_min_clarity"],
                "max_clarity": data["cl_max_clarity"],
                "face_detected_ratio": data["cl_face_detected_ratio"],
                "per_frame": self._load_json(data["cl_per_frame_json"]),
            },
            "expression": None if data["ex_intensity"] is None else {
                "intensity": data["ex_intensity"],
                "rationale": data["ex_rationale"],
                "dominant_expressions": self._load_json(data["ex_dominant_expressions_json"], []),
                "raw_response": self._load_json(data["ex_raw_response"]),
            },
            "transcription": None if data["tr_full_text"] is None else {
                "text": data["tr_full_text"],
                "language": data["tr_language"],
                "segments": self._load_json(data["tr_segments_json"], []),
            },
        }
