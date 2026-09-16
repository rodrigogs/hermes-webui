"""A gapped load must be VISIBLE, not just logged (spec §3).

`_read_sidecar_document` has always named a missing or unreadable chunk in
`doc['chunk_errors']`, but `cls(**data)` dropped that key into `**kwargs` --
`Session.__init__` has no generic setattr -- and `_chunk_read_incomplete` was
written by `load()` and read nowhere in production. A session that came up as
its TAIL therefore looked healthy in the UI and in `compact()`, which is the
one thing spec §3 says it must not do: the marker is persisted into the head on
the next save so the gap is visible in metadata.

The other half of the requirement is that the marker is DERIVED from the read
that just happened, never carried forward: the head is not an authority on
whether ITS OWN chunks are readable now. A head that persisted `chunk_errors`
and whose chunks are back must load clean and drop the key on its next save,
or the marker is permanent and stops meaning anything.
"""
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
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_MSGS", 10)
    monkeypatch.setattr(M, "_SIDECAR_TAIL_KEEP", 4)
    return sdir


def _msgs(start, n):
    return [{"role": "user", "timestamp": 1757000000.0 + i, "content": f"m{i}"}
            for i in range(start, start + n)]


def _two_chunk_session(session_store, sid):
    """A session sealed into TWO chunks, so removing the second still leaves a
    tail long enough to re-seal -- i.e. the head written after the gapped load
    still carries `message_chunks`, which is what the placement assertion in
    the first test needs."""
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 22))
    s.save()
    s.messages = s.messages + _msgs(22, 11)
    s.save()
    doc = json.loads((session_store / f"{sid}.json").read_bytes())
    assert [e["file"] for e in doc["message_chunks"]] == ["000001.json", "000002.json"]
    assert doc["message_count"] == 33
    return session_store / f"{sid}.json"


def test_a_gapped_load_is_visible_on_the_object_in_compact_and_in_the_head(session_store):
    """The three places the gap was invisible, in one pass.

    Fails on any of: `Session.__init__` no longer storing `chunk_errors` (the
    key falls back into `**kwargs` and `s.chunk_errors` raises
    AttributeError); `compact()` no longer emitting it (the UI and the sidebar
    row go back to showing a truncated session as healthy); `save()` emitting
    it after `messages` instead of next to `message_chunks` (it leaves the
    cheap metadata prefix, so no prefix reader can see it).
    """
    head = _two_chunk_session(session_store, "e1")
    (session_store / "e1.msgs" / "000002.json").unlink()

    s = M.Session.load("e1")

    assert len(s.messages) == 22, "chunk 1 plus the tail; chunk 2's 11 are the hole"
    assert s.chunk_errors and any("000002.json" in e for e in s.chunk_errors), s.chunk_errors
    assert s.compact()["chunk_errors"] == s.chunk_errors

    s.save()

    raw = head.read_text(encoding="utf-8")
    assert '"chunk_errors"' in raw, "the marker must be persisted into the head"
    off = {k: raw.index(f'"{k}"') for k in ("message_chunks", "chunk_errors", "messages")}
    assert off["message_chunks"] < off["chunk_errors"] < off["messages"], off


def test_a_restored_chunk_clears_the_marker_the_head_still_carries(session_store):
    """The marker is recomputed by the read, never inherited from the head.

    Fails if `_read_sidecar_document` stops discarding the head's persisted
    `chunk_errors` before it looks at the chunks: the stale list is then
    carried onto the object by `cls(**data)`, re-persisted by every later
    save, and -- via `load()`'s `data.get('chunk_errors')` branch -- keeps
    denying the fast-path identity to a session whose chunks are all readable.
    """
    head = _two_chunk_session(session_store, "e2")
    # The FIRST chunk, so the re-seal the gapped save performs takes a seq past
    # both existing names (_next_chunk_seq scans the directory) and restoring
    # the stashed bytes below cannot land on top of it.
    chunk = session_store / "e2.msgs" / "000001.json"
    stashed = chunk.read_bytes()
    chunk.unlink()

    gapped = M.Session.load("e2")
    assert gapped.chunk_errors
    gapped.save()
    assert '"chunk_errors"' in head.read_text(encoding="utf-8")

    chunk.write_bytes(stashed)  # the transient failure is over
    live = json.loads(head.read_bytes())["message_chunks"]
    assert all((session_store / "e2.msgs" / e["file"]).exists() for e in live), \
        "every chunk this head names must be readable, or the load is right to complain"

    healed = M.Session.load("e2")

    assert healed.chunk_errors == [], "a clean read must not inherit the head's marker"
    assert healed._chunk_read_incomplete == []
    assert "chunk_errors" not in healed.compact()

    healed.save()

    assert '"chunk_errors"' not in head.read_text(encoding="utf-8"), \
        "the next save must drop a marker that no longer describes the disk"


def test_a_healthy_session_never_carries_the_key(session_store):
    """An empty list must not appear in every head in the store.

    Fails if `save()` or `compact()` emits `chunk_errors` unconditionally: the
    key then lands in every head's cheap metadata prefix (the budgeted region
    #5854 exists to keep small) and in every sidebar row, where a marker that
    is always present tells the operator nothing.
    """
    head = _two_chunk_session(session_store, "e3")

    assert '"chunk_errors"' not in head.read_text(encoding="utf-8")

    s = M.Session.load("e3")

    assert s.chunk_errors == []
    assert "chunk_errors" not in s.compact()

    s.messages = s.messages + _msgs(99, 1)
    s.save()

    assert '"chunk_errors"' not in head.read_text(encoding="utf-8")
    assert len(M.Session.load("e3").messages) == 34
