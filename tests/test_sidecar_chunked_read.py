"""Reading a segmented sidecar: concatenation, order, and integrity.

Every fixture here is written by hand rather than by save(). At this point in
the plan nothing produces this format, and even later that separation is worth
keeping: a reader test that depends on the writer cannot tell a reader bug from
a writer bug.
"""
import hashlib
import json
from collections import OrderedDict

import pytest

import api.models as M


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    return sdir


def _msgs(start, n):
    return [{"role": "user", "ts": float(i), "content": f"m{i}"} for i in range(start, start + n)]


def _write_chunk(session_store, sid, seq, msgs, first_idx):
    d = session_store / f"{sid}.msgs"
    d.mkdir(parents=True, exist_ok=True)
    body = {"session_id": sid, "seq": seq, "first_idx": first_idx,
            "count": len(msgs), "messages": msgs}
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    (d / f"{seq:06d}.json").write_bytes(raw)
    return {"seq": seq, "file": f"{seq:06d}.json", "count": len(msgs),
            "first_idx": first_idx, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_head(session_store, sid, manifest, tail, **extra):
    doc = {"session_id": sid, "title": "T", "workspace": "", "model": "glm",
           "created_at": 1.0, "updated_at": 2.0,
           "message_count": M._sealed_total(manifest) + len(tail),
           "anchor_scene_index": {},
           "message_chunks": manifest, "messages": tail, "tool_calls": []}
    doc.update(extra)
    p = session_store / f"{sid}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def test_unsegmented_file_reads_exactly_as_today(session_store):
    """A head with no manifest is today's file. This is the no-migration
    guarantee: the reader must not require anything new to exist."""
    doc = {"session_id": "u1", "title": "T", "created_at": 1.0, "updated_at": 2.0,
           "messages": _msgs(0, 3)}
    p = session_store / "u1.json"
    p.write_text(json.dumps(doc), encoding="utf-8")

    out = M._read_sidecar_document(p, "u1")

    assert [m["content"] for m in out["messages"]] == ["m0", "m1", "m2"]
    assert "chunk_errors" not in out


def test_chunks_concatenate_before_the_tail_in_manifest_order(session_store):
    c1 = _write_chunk(session_store, "s1", 1, _msgs(0, 3), 0)
    c2 = _write_chunk(session_store, "s1", 2, _msgs(3, 2), 3)
    p = _write_head(session_store, "s1", [c1, c2], _msgs(5, 2))

    out = M._read_sidecar_document(p, "s1")

    assert [m["content"] for m in out["messages"]] == ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]
    assert "chunk_errors" not in out


def test_manifest_order_wins_over_filename_order(session_store):
    """The manifest is the authority. A directory listing is not: an orphan or a
    re-sealed chunk can make filename order disagree with history."""
    c2 = _write_chunk(session_store, "s2", 2, _msgs(0, 2), 0)
    c1 = _write_chunk(session_store, "s2", 1, _msgs(2, 2), 2)
    p = _write_head(session_store, "s2", [c2, c1], _msgs(4, 1))

    out = M._read_sidecar_document(p, "s2")

    assert [m["content"] for m in out["messages"]] == ["m0", "m1", "m2", "m3", "m4"]


def test_orphan_chunk_not_in_the_manifest_is_ignored(session_store):
    c1 = _write_chunk(session_store, "s3", 1, _msgs(0, 2), 0)
    _write_chunk(session_store, "s3", 2, _msgs(100, 5), 2)  # orphan: crash before the head
    p = _write_head(session_store, "s3", [c1], _msgs(2, 1))

    out = M._read_sidecar_document(p, "s3")

    assert [m["content"] for m in out["messages"]] == ["m0", "m1", "m2"]
    assert "chunk_errors" not in out, "an orphan is expected after a crash, not an error"


def test_missing_chunk_file_reports_and_keeps_loading(session_store):
    c1 = _write_chunk(session_store, "s4", 1, _msgs(0, 2), 0)
    c2 = _write_chunk(session_store, "s4", 2, _msgs(2, 2), 2)
    p = _write_head(session_store, "s4", [c1, c2], _msgs(4, 1))
    (session_store / "s4.msgs" / "000002.json").unlink()

    out = M._read_sidecar_document(p, "s4")

    assert out is not None, "the session must still open"
    assert [m["content"] for m in out["messages"]] == ["m0", "m1", "m4"]
    assert any("000002.json" in e for e in out["chunk_errors"])
    assert out["message_count"] == 5, "the claimed total stays, so the gap is visible"


def test_corrupted_chunk_fails_its_checksum_and_is_reported(session_store):
    c1 = _write_chunk(session_store, "s5", 1, _msgs(0, 2), 0)
    p = _write_head(session_store, "s5", [c1], _msgs(2, 1))
    f = session_store / "s5.msgs" / "000001.json"
    f.write_bytes(f.read_bytes().replace(b"m0", b"XX"))

    out = M._read_sidecar_document(p, "s5")

    assert [m["content"] for m in out["messages"]] == ["m2"]
    assert any("sha256" in e and "000001.json" in e for e in out["chunk_errors"])


def test_chunk_whose_count_disagrees_with_its_array_is_reported(session_store):
    c1 = _write_chunk(session_store, "s6", 1, _msgs(0, 2), 0)
    c1["count"] = 3  # manifest lies; the chunk body holds 2
    p = _write_head(session_store, "s6", [c1], _msgs(2, 1))

    out = M._read_sidecar_document(p, "s6")

    assert any("count" in e for e in out["chunk_errors"])


def test_torn_head_returns_none(session_store):
    p = session_store / "s7.json"
    p.write_text('{"session_id": "s7", "messages": [', encoding="utf-8")
    assert M._read_sidecar_document(p, "s7") is None


def test_session_load_reads_a_segmented_session(session_store):
    """The integration point: Session.load must return the full history."""
    c1 = _write_chunk(session_store, "s8", 1, _msgs(0, 4), 0)
    _write_head(session_store, "s8", [c1], _msgs(4, 2))

    s = M.Session.load("s8")

    assert s is not None
    assert [m["content"] for m in s.messages] == ["m0", "m1", "m2", "m3", "m4", "m5"]
