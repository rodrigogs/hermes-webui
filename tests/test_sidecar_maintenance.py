"""Rollback and inspection. A storage format without a way back is a trap."""
import json
import shutil
from collections import OrderedDict

import pytest

import api.models as M
from api.sidecar_maintenance import fsck_sessions, unchunk_session


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_MSGS", 10)
    monkeypatch.setattr(M, "_SIDECAR_TAIL_KEEP", 4)
    return sdir


def _msgs(n):
    return [{"role": "user", "ts": float(i), "content": f"m{i}"} for i in range(n)]


def _segmented(session_store, sid, n=22):
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(n))
    s.save()
    return session_store / f"{sid}.json"


def test_unchunk_folds_everything_back_into_one_file(session_store):
    p = _segmented(session_store, "m1", n=22)
    out = unchunk_session("m1")
    assert out["unchunked"] is True and out["messages"] == 22
    doc = json.loads(p.read_bytes())
    assert not doc.get("message_chunks")
    assert [m["content"] for m in doc["messages"]] == [f"m{i}" for i in range(22)]
    assert not (session_store / "m1.msgs").exists()
    assert [m["content"] for m in M.Session.load("m1").messages] == [f"m{i}" for i in range(22)]


def test_unchunk_is_idempotent(session_store):
    _segmented(session_store, "m2", n=22)
    unchunk_session("m2")
    out = unchunk_session("m2")
    assert out["unchunked"] is False and out["messages"] == 22


def test_unchunk_refuses_when_a_chunk_is_missing(session_store):
    p = _segmented(session_store, "m3", n=22)
    next(iter((session_store / "m3.msgs").glob("*.json"))).unlink()
    out = unchunk_session("m3")
    assert out["unchunked"] is False and "error" in out
    assert json.loads(p.read_bytes()).get("message_chunks"), "the manifest must survive a refusal"
    assert (session_store / "m3.msgs").exists(), "nothing removed on a refusal"


def test_fsck_reports_a_healthy_segmented_session(session_store):
    _segmented(session_store, "m4", n=22)
    rep = fsck_sessions(session_store)
    row = next(r for r in rep["sessions"] if r["session_id"] == "m4")
    assert row["chunks"] == 1 and row["total_messages"] == 22 and row["errors"] == []
    assert rep["orphans"] == []


def test_fsck_lists_orphan_chunks_without_touching_them(session_store):
    _segmented(session_store, "m5", n=22)
    orphan = session_store / "m5.msgs" / "009999.json"
    orphan.write_text('{"session_id": "m5", "seq": 9999, "first_idx": 0, "count": 0, "messages": []}',
                      encoding="utf-8")
    rep = fsck_sessions(session_store)
    assert any(o["file"].endswith("009999.json") for o in rep["orphans"])
    assert orphan.exists(), "fsck must never delete"


def test_fsck_reports_a_corrupted_chunk(session_store):
    _segmented(session_store, "m6", n=22)
    f = next(iter((session_store / "m6.msgs").glob("*.json")))
    f.write_bytes(f.read_bytes().replace(b"m0", b"XX"))
    rep = fsck_sessions(session_store)
    row = next(r for r in rep["sessions"] if r["session_id"] == "m6")
    assert any("sha256" in e for e in row["errors"])


def test_unchunk_leaves_a_bak_only_chunk_alone_after_a_reseal(session_store):
    """Differing-manifest case: a shrink big enough to eat into the sealed
    prefix forces a re-seal, so the live head ends up naming a NEW chunk
    while the `.bak` this same save just wrote still names the OLD one.
    unchunk_session must not delete a chunk its own `.bak` still needs, even
    though the live manifest no longer names it.
    """
    p = _segmented(session_store, "m7", n=22)
    s = M.Session.load("m7")
    assert len(s.messages) == 22
    s.messages = _msgs(6)  # shrinks below the sealed 18 -> forces a re-seal
    s.save()
    bak = session_store / "m7.json.bak"
    assert bak.exists()
    bak_manifest = json.loads(bak.read_bytes())["message_chunks"]
    bak_files = {e["file"] for e in bak_manifest}
    assert bak_files, "the .bak must claim at least one sealed chunk for this test to mean anything"

    out = unchunk_session("m7")
    assert out["unchunked"] is True and out["messages"] == 6

    chunk_dir = session_store / "m7.msgs"
    for fname in bak_files:
        assert (chunk_dir / fname).exists(), f"{fname} is needed by m7.json.bak"

    # Restoring the .bak must still recover the full pre-shrink history.
    shutil.copyfile(bak, p)
    restored = M.Session.load("m7")
    assert not restored._chunk_read_incomplete, restored._chunk_read_incomplete
    assert len(restored.messages) == 22


def test_unchunk_leaves_a_chunk_alone_when_the_bak_shares_it(session_store):
    """Shared-chunk case: a tail-only shrink does NOT force a re-seal (the
    sealed prefix is untouched), so the live head and its `.bak` can name
    the exact SAME sealed chunk file. unchunk_session must not delete a
    chunk its own `.bak` still needs, even though the LIVE manifest
    (correctly, for the live head's own purposes) names it too.
    """
    p = _segmented(session_store, "m8", n=30)
    s = M.Session.load("m8")
    assert len(s.messages) == 30
    s.messages = s.messages[:-2]  # tail-only shrink: 30 -> 28, no re-seal
    s.save()
    bak = session_store / "m8.json.bak"
    assert bak.exists()
    bak_manifest = json.loads(bak.read_bytes())["message_chunks"]
    bak_files = {e["file"] for e in bak_manifest}
    live_manifest = json.loads(p.read_bytes())["message_chunks"]
    live_files = {e["file"] for e in live_manifest}
    assert bak_files == live_files, "this test only means something when live and .bak share a chunk"

    out = unchunk_session("m8")
    assert out["unchunked"] is True and out["messages"] == 28

    chunk_dir = session_store / "m8.msgs"
    for fname in bak_files:
        assert (chunk_dir / fname).exists(), f"{fname} is needed by m8.json.bak"

    # Restoring the .bak must still recover the full pre-shrink history.
    shutil.copyfile(bak, p)
    restored = M.Session.load("m8")
    assert not restored._chunk_read_incomplete, restored._chunk_read_incomplete
    assert len(restored.messages) == 30
