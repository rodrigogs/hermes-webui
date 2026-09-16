"""Deleting a session must not leave its chunks behind.

Four call sites unlink session files today (routes.py 16106, 16262, 22627,
27364). Each one that forgets the directory leaks the bulk of the session's
bytes -- the head is now the small part.
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


def _segmented(session_store, sid, n=22):
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent), model="glm",
                  messages=[{"role": "user", "ts": float(i), "content": f"m{i}"} for i in range(n)])
    s.save()
    s.messages = s.messages[:8]
    s.save()  # forces a re-seal, so orphan chunks exist too
    return session_store / f"{sid}.json"


def test_remove_session_files_removes_head_bak_and_chunk_dir(session_store):
    p = _segmented(session_store, "d1")
    p.with_suffix(".json.bak").write_text("{}", encoding="utf-8")
    assert (session_store / "d1.msgs").is_dir()

    M._remove_session_files("d1")

    assert not p.exists()
    assert not p.with_suffix(".json.bak").exists()
    assert not (session_store / "d1.msgs").exists(), "orphans included"


def test_remove_session_files_is_idempotent_and_quiet(session_store):
    M._remove_session_files("never-existed")
    M._remove_session_files("never-existed")


def test_remove_session_files_refuses_an_unsafe_sid(session_store):
    other = session_store / "keep.json"
    other.write_text("{}", encoding="utf-8")
    M._remove_session_files("../keep")
    assert other.exists(), "an unsafe sid must not delete anything"


def test_remove_session_files_leaves_other_sessions_alone(session_store):
    _segmented(session_store, "d2")
    _segmented(session_store, "d3")
    M._remove_session_files("d2")
    assert not (session_store / "d2.msgs").exists()
    assert (session_store / "d3.msgs").is_dir()
    assert (session_store / "d3.json").exists()
