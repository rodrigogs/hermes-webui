"""C2d: a head that names a chunk not on disk is never published (spec 2026-09-17 §3.5);
a gapped load whose head moved under it re-reads (§3.8)."""
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
    monkeypatch.setattr(M, "_SIDECAR_TAIL_KEEP", 5)
    return sdir


def _msg(i, role="user", content=None):
    return {"role": role, "timestamp": float(i), "content": content if content is not None else f"{role} message {i}"}


def _segmented(session_store, sid, n=30):
    """A session sealed once: with TAIL_MAX=10/KEEP=5, n=30 seals 25 into one chunk, 5 stay in the tail."""
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent), model="glm",
                  messages=[_msg(i) for i in range(n)])
    s.save(touch_updated_at=False, skip_index=True)
    return s


def _chunks(session_store, sid):
    d = session_store / f"{sid}.msgs"
    return sorted(p.name for p in d.glob("*.json")) if d.exists() else []


def _manifest(session_store, sid):
    return M._normalised_manifest(json.loads((session_store / f"{sid}.json").read_bytes()).get("message_chunks"))


def test_a_chunk_that_vanishes_before_publish_is_resealed_not_named(session_store, monkeypatch):
    """Fails if save() publishes the payload it built at the top without re-checking the files.
    The chunk is deleted INSIDE _safe_replace's caller window: between the tmp write and the rename."""
    s = _segmented(session_store, "pw", n=30)
    victim = session_store / "pw.msgs" / _chunks(session_store, "pw")[0]
    real_missing = M._missing_manifest_files
    state = {"deleted": False}

    def delete_then_check(sid, manifest):
        # emulate a deleter racing us: the chunk disappears AFTER the top-of-save
        # _reusable_manifest_prefix check and right as the publish sweep runs
        if not state["deleted"]:
            victim.unlink()
            state["deleted"] = True
        return real_missing(sid, manifest)

    monkeypatch.setattr(M, "_missing_manifest_files", delete_then_check)
    s.messages = s.messages + [_msg(30)]
    s.save(touch_updated_at=False, skip_index=True)
    head = json.loads((session_store / "pw.json").read_bytes())
    named = [e["file"] for e in head.get("message_chunks", [])]
    assert all((session_store / "pw.msgs" / f).exists() for f in named), f"published head names a missing file: {named}"
    assert victim.name not in named
    M.SESSIONS.clear()
    loaded = M.Session.load("pw")
    assert len(loaded.messages) == 31 and not loaded.chunk_errors


def test_three_failures_raise_and_leave_the_previous_head(session_store, monkeypatch):
    """Fails if save() gives up silently or publishes anyway."""
    s = _segmented(session_store, "p3", n=30)
    before = (session_store / "p3.json").read_bytes()
    monkeypatch.setattr(M, "_missing_manifest_files", lambda sid, manifest: ["000001.json"] if manifest else [])
    s.messages = s.messages + [_msg(30)]
    with pytest.raises(RuntimeError):
        s.save(touch_updated_at=False, skip_index=True)
    assert (session_store / "p3.json").read_bytes() == before
    assert not list(session_store.glob("p3.tmp*"))


def test_load_rereads_when_chunk_errors_and_the_head_changed(session_store, monkeypatch):
    """Fails if load() believes chunk_errors without noticing the head moved under it."""
    s = _segmented(session_store, "lr", n=30)
    real_read = M._read_sidecar_document
    calls = {"n": 0}

    def first_read_gapped(path, sid=None):
        calls["n"] += 1
        doc = real_read(path, sid)
        if calls["n"] == 1:
            doc["chunk_errors"] = ["000001.json: unreadable (transient)"]
            doc["messages"] = doc["messages"][-5:]
            # the head "changes" under the read
            (session_store / "lr.json").write_bytes((session_store / "lr.json").read_bytes() + b"\n")
        return doc

    monkeypatch.setattr(M, "_read_sidecar_document", first_read_gapped)
    M.SESSIONS.clear()
    loaded = M.Session.load("lr")
    assert calls["n"] == 2, "one re-read"
    assert len(loaded.messages) == 30 and not loaded.chunk_errors


def test_load_does_not_retry_when_the_head_is_stable(session_store, monkeypatch):
    """Fails if load() retries on every chunk_errors regardless of identity (would loop on real corruption)."""
    _segmented(session_store, "ls", n=30)
    (session_store / "ls.msgs" / "000001.json").unlink()
    real_read = M._read_sidecar_document
    calls = {"n": 0}

    def counting(path, sid=None):
        calls["n"] += 1
        return real_read(path, sid)

    monkeypatch.setattr(M, "_read_sidecar_document", counting)
    M.SESSIONS.clear()
    loaded = M.Session.load("ls")
    assert calls["n"] == 1 and loaded.chunk_errors
