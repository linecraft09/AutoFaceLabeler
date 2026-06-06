import sqlite3

import pytest


def get_annotation_run(video_store, run_id):
    with video_store._connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM annotation_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None


def test_annotation_runs_table_exists(video_store):
    with video_store._connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'annotation_runs'"
        ).fetchone()

    assert row is not None


def test_create_annotation_run(video_store):
    video_store.create_annotation_run("annotation-run-1")

    row = get_annotation_run(video_store, "annotation-run-1")
    assert row["run_id"] == "annotation-run-1"
    assert row["target_count"] == 200


def test_create_annotation_run_defaults(video_store):
    video_store.create_annotation_run("annotation-run-1")

    row = get_annotation_run(video_store, "annotation-run-1")
    assert row["stage"] == "pending"
    assert row["status"] == "in_progress"
    assert row["completed_at"] is None


def test_update_annotation_run_stage(video_store):
    video_store.create_annotation_run("annotation-run-1")
    video_store.update_annotation_run_stage("annotation-run-1", "label_1", status="in_progress")

    row = get_annotation_run(video_store, "annotation-run-1")
    assert row["stage"] == "label_1"
    assert row["status"] == "in_progress"


def test_complete_annotation_run(video_store):
    video_store.create_annotation_run("annotation-run-1")
    video_store.complete_annotation_run("annotation-run-1")

    row = get_annotation_run(video_store, "annotation-run-1")
    assert row["status"] == "completed"
    assert row["completed_at"] is not None


def test_fail_annotation_run(video_store):
    video_store.create_annotation_run("annotation-run-1")
    video_store.fail_annotation_run("annotation-run-1")

    row = get_annotation_run(video_store, "annotation-run-1")
    assert row["status"] == "failed"


def test_get_in_progress_annotation_run(video_store):
    video_store.create_annotation_run("old-run")
    video_store.create_annotation_run("new-run")
    video_store.update_annotation_run_stage("new-run", "label_2")

    row = video_store.get_in_progress_annotation_run()
    assert row["run_id"] == "new-run"


def test_get_in_progress_none(video_store):
    video_store.create_annotation_run("annotation-run-1")
    video_store.complete_annotation_run("annotation-run-1")

    assert video_store.get_in_progress_annotation_run() is None


def test_annotation_run_unique_id(video_store):
    video_store.create_annotation_run("annotation-run-1")

    with pytest.raises(sqlite3.IntegrityError):
        video_store.create_annotation_run("annotation-run-1")


def test_prune_old_annotation_runs(video_store):
    for index in range(5):
        run_id = f"completed-{index}"
        video_store.create_annotation_run(run_id)
        video_store.complete_annotation_run(run_id)
        with video_store._connect() as conn:
            conn.execute("""
                UPDATE annotation_runs
                SET created_at = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE run_id = ?
            """, (
                f"2026-01-01 00:00:0{index}",
                f"2026-01-01 00:00:0{index}",
                f"2026-01-01 00:00:0{index}",
                run_id,
            ))
    video_store.create_annotation_run("failed-old")
    video_store.fail_annotation_run("failed-old")
    with video_store._connect() as conn:
        conn.execute("""
            UPDATE annotation_runs
            SET created_at = '2025-01-01 00:00:00',
                updated_at = '2025-01-01 00:00:00'
            WHERE run_id = 'failed-old'
        """)
    video_store.create_annotation_run("active-run")

    deleted = video_store.prune_old_annotation_runs(keep_last=2)

    assert deleted == 4
    with video_store._connect() as conn:
        rows = conn.execute("SELECT run_id FROM annotation_runs ORDER BY run_id").fetchall()
    remaining_ids = {row[0] for row in rows}
    assert remaining_ids == {"active-run", "completed-3", "completed-4"}
