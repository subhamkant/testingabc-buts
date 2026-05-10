"""
Smoke tests for CheckpointStore. Run with:
    PYTHONIOENCODING=utf-8 python -m pipeline.test_checkpoint

These are lightweight assertion-based tests, not pytest — keeps the test
deps zero. Returns non-zero exit code on failure.
"""

import json
import os
import shutil
import sys
import tempfile

from pipeline.checkpoint import (
    CheckpointStore,
    default_run_id,
    resolve_run_id,
)


def _tempdir() -> str:
    return tempfile.mkdtemp(prefix="ck_test_")


def test_default_run_id_format():
    rid = default_run_id("mahabharata", "hi")
    assert rid.startswith("mahabharata_hi_"), rid
    assert len(rid.split("_")) == 4  # series_lang_YYYYMMDD_HH
    rid_w = default_run_id("whatif")
    assert rid_w.startswith("whatif_"), rid_w
    assert "_hi_" not in rid_w and "_en_" not in rid_w, "whatif should be language-agnostic"


def test_resolve_run_id_env_override():
    os.environ["PIPELINE_RUN_ID"] = "custom_run_42"
    try:
        assert resolve_run_id("anything") == "custom_run_42"
    finally:
        del os.environ["PIPELINE_RUN_ID"]
    # falls through to default when unset
    rid = resolve_run_id("krishna", "hi")
    assert rid.startswith("krishna_hi_")


def test_has_missing_file():
    root = _tempdir()
    try:
        ck = CheckpointStore("r1", root=root)
        assert ck.has("nope.json") is False
    finally:
        shutil.rmtree(root)


def test_save_and_load_json_roundtrip():
    root = _tempdir()
    try:
        ck = CheckpointStore("r1", root=root)
        payload = {"title": "हिन्दी unicode 🎬", "scenes": [1, 2, 3]}
        ck.save_json("script.json", payload)
        assert ck.has("script.json")
        loaded = ck.load_json("script.json")
        assert loaded == payload
    finally:
        shutil.rmtree(root)


def test_atomic_write_no_partial_file_on_replace():
    """An interrupted save_json must leave NO file (or a complete one) —
    never a half-written file. Simulate by inspecting that the .tmp file
    never coexists with the final file after a successful save."""
    root = _tempdir()
    try:
        ck = CheckpointStore("r1", root=root)
        ck.save_json("script.json", {"a": 1})
        # tmp file should not exist after successful save
        assert not os.path.exists(ck.path("script.json.tmp"))
        # final file should exist and load
        assert ck.load_json("script.json") == {"a": 1}
    finally:
        shutil.rmtree(root)


def test_empty_file_treated_as_missing():
    """Defends against a half-write that left a zero-byte file."""
    root = _tempdir()
    try:
        ck = CheckpointStore("r1", root=root)
        # Manually create an empty file (simulating crash mid-write)
        with open(ck.path("script.json"), "w") as f:
            pass
        assert ck.has("script.json") is False, "empty file must count as missing"
    finally:
        shutil.rmtree(root)


def test_save_file_copies_and_atomic():
    root = _tempdir()
    src = _tempdir()
    try:
        # Source file outside the cache dir
        src_path = os.path.join(src, "scene_00.jpg")
        with open(src_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0fakejpg" * 100)

        ck = CheckpointStore("r1", root=root)
        cached = ck.save_file("visuals/scene_00.jpg", src_path)

        # File now in the cache, .tmp gone, contents match
        assert os.path.exists(cached)
        assert not os.path.exists(cached + ".tmp")
        with open(cached, "rb") as f:
            assert f.read().startswith(b"\xff\xd8\xff\xe0")
        assert ck.has("visuals/scene_00.jpg")
    finally:
        shutil.rmtree(root)
        shutil.rmtree(src)


def test_save_files_bulk():
    root = _tempdir()
    src = _tempdir()
    try:
        srcs = []
        for i in range(3):
            p = os.path.join(src, f"clip_{i:02d}.mp4")
            with open(p, "wb") as f:
                f.write(b"x" * 1000)
            srcs.append(p)

        ck = CheckpointStore("r1", root=root)
        cached = ck.save_files("clips", srcs)

        assert len(cached) == 3
        for p in cached:
            assert os.path.exists(p)
            assert "clips" in p
    finally:
        shutil.rmtree(root)
        shutil.rmtree(src)


def test_mark_done_and_list_entries():
    root = _tempdir()
    try:
        ck = CheckpointStore("r1", root=root)
        ck.save_json("script.json", {"x": 1})
        ck.mark_done("visuals.done")
        ck.save_json("audio_meta.json", {"chars": 1234})

        entries = ck.list_entries()
        assert "script.json" in entries
        assert "visuals.done" in entries
        assert "audio_meta.json" in entries
        # Markers are non-empty (have an ISO timestamp inside)
        assert ck.has("visuals.done")
    finally:
        shutil.rmtree(root)


def test_resume_pattern_skip_when_present():
    """Simulates the canonical 'if has: skip; else: do work' pattern."""
    root = _tempdir()
    try:
        ck1 = CheckpointStore("same_run", root=root)
        ck1.save_json("script.json", {"title": "first run"})

        # New CheckpointStore on the SAME run_id (this is what a retry job
        # does — reopens the same cache directory). It should see the
        # existing checkpoint.
        ck2 = CheckpointStore("same_run", root=root)
        assert ck2.has("script.json")
        assert ck2.load_json("script.json")["title"] == "first run"
    finally:
        shutil.rmtree(root)


def test_different_run_ids_isolated():
    root = _tempdir()
    try:
        ck_a = CheckpointStore("run_a", root=root)
        ck_b = CheckpointStore("run_b", root=root)
        ck_a.save_json("script.json", {"id": "a"})
        ck_b.save_json("script.json", {"id": "b"})

        assert ck_a.load_json("script.json")["id"] == "a"
        assert ck_b.load_json("script.json")["id"] == "b"
    finally:
        shutil.rmtree(root)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  [OK] {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e) or "assertion failed"))
            print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  [ERR ] {t.__name__}: {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
