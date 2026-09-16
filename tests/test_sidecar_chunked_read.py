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


def test_manifest_entry_with_path_traversal_filename_is_refused(session_store):
    """A hand-edited or corrupted head can name a `file` outside the chunk
    dir. It must be refused before it is ever opened, not silently followed.

    The refusal now happens one layer earlier: `_normalised_manifest` requires
    a name its own writer could have produced, so the entry never reaches the
    read loop and every OTHER consumer of the manifest inherits the same
    rejection. The reported message therefore names the position in the
    manifest rather than the filename; the guard in the read loop stays as
    defence in depth for a manifest that did not come through here.
    """
    c1 = _write_chunk(session_store, "s10", 1, _msgs(0, 2), 0)
    # A decoy outside the chunk dir: if the guard were missing, this is what a
    # "../" escape would read instead of refusing the entry.
    (session_store / "escape.json").write_text(
        json.dumps({"messages": _msgs(900, 9)}), encoding="utf-8")
    c1["file"] = "../escape.json"
    p = _write_head(session_store, "s10", [c1], _msgs(2, 1))

    out = M._read_sidecar_document(p, "s10")

    assert [m["content"] for m in out["messages"]] == ["m2"], "decoy never read; rest of session still loads"
    assert any("malformed" in e for e in out["chunk_errors"]), out["chunk_errors"]


def test_manifest_truncation_from_broken_continuity_is_reported(session_store):
    """entry 3's first_idx does not chain: _normalised_manifest silently drops
    it (and everything after) from the returned prefix. The reader must not
    let that truncation look like a complete read."""
    c1 = _write_chunk(session_store, "s11", 1, _msgs(0, 2), 0)
    c2 = _write_chunk(session_store, "s11", 2, _msgs(2, 2), 2)
    c3 = _write_chunk(session_store, "s11", 3, _msgs(4, 2), 4)
    c3["first_idx"] = 999  # continuity break: normalised_manifest stops before this entry
    p = _write_head(session_store, "s11", [c1, c2, c3], _msgs(6, 1))

    out = M._read_sidecar_document(p, "s11")

    assert [m["content"] for m in out["messages"]] == ["m0", "m1", "m2", "m3", "m6"]
    assert any("truncat" in e.lower() for e in out["chunk_errors"])
    assert out["message_count"] == M._sealed_total([c1, c2, c3]) + 1, "the original claim stays visible"


def test_a_memoryerror_parsing_a_chunk_is_not_reported_as_corruption(session_store):
    """A transient allocation failure must propagate, not become a gap.

    The chunk parse is the one place where "could not read it" and "it is
    corrupt" are indistinguishable to the caller, and the difference decides
    whether the session opens as its tail and the NEXT ordinary save persists
    that truncation. This host runs with no swap and has been OOM-killed six
    times in a week, so MemoryError is a real event.

    Fails if the chunk-body guard in `_read_sidecar_document` is widened back
    to `except Exception`: the MemoryError is then swallowed, the call returns
    a document whose `messages` holds only the tail and whose `chunk_errors`
    accuses the chunk of not being valid JSON.
    """
    c1 = _write_chunk(session_store, "s12", 1, _msgs(0, 2), 0)
    p = _write_head(session_store, "s12", [c1], _msgs(2, 1))
    # Match the chunk's exact bytes, not a substring: the HEAD's own manifest
    # carries "seq"/"first_idx" too, so a key-based match would also fire on
    # the head parse and prove nothing about the chunk path.
    chunk_raw = (session_store / "s12.msgs" / "000001.json").read_bytes()
    real_loads = json.loads

    def fake_loads(s, *a, **k):
        if s == chunk_raw:
            raise MemoryError("simulated allocation failure")
        return real_loads(s, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M.json, "loads", fake_loads)
        with pytest.raises(MemoryError):
            M._read_sidecar_document(p, "s12")


def test_session_load_reads_a_segmented_session(session_store):
    """The integration point: Session.load must return the full history."""
    c1 = _write_chunk(session_store, "s8", 1, _msgs(0, 4), 0)
    _write_head(session_store, "s8", [c1], _msgs(4, 2))

    s = M.Session.load("s8")

    assert s is not None
    assert [m["content"] for m in s.messages] == ["m0", "m1", "m2", "m3", "m4", "m5"]
