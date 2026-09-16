"""Deleting a session must not leave its chunks behind.

Four call sites unlink session files today (routes.py 16106, 16262, 22627,
27364). Each one that forgets the directory leaks the bulk of the session's
bytes -- the head is now the small part.

`_handle_sessions_cleanup`'s orphan/empty-session sweep is a fifth, unlisted
site with the same shape: it fully removes a session (zero messages, title
"Untitled"), but historically unlinked only the head -- leaking a `.bak` and
(with segmenting) the chunk dir, the largest of the three.
"""
import io
import json
import threading
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


# ── _handle_sessions_cleanup's orphan/empty sweep (fifth site) ─────────────


@pytest.fixture
def cleanup_env(session_store, monkeypatch, tmp_path):
    """Point `api.routes._handle_sessions_cleanup` at the same SESSION_DIR
    `session_store` already set up on `api.models`.

    `from api.models import SESSION_DIR` (etc.) in routes.py binds a separate
    name in routes.py's own module namespace, so patching `api.models` alone
    (what `session_store` does) leaves `api.routes` still pointed at the real
    on-disk default -- mirrors the established pattern in
    tests/test_issue5331_index_only_ghost_cleanup.py's `mock_env` fixture.
    """
    import api.routes as routes

    monkeypatch.setattr(routes, "SESSION_DIR", session_store)
    monkeypatch.setattr(routes, "SESSION_INDEX_FILE", session_store / "_index.json")
    monkeypatch.setattr(routes, "SESSIONS", {})
    monkeypatch.setattr(routes, "LOCK", threading.Lock())
    # _handle_sessions_cleanup also tries to drop the session's attachment
    # dir now; keep that inside the sandbox instead of touching the real
    # HERMES state dir.
    monkeypatch.setenv("HERMES_WEBUI_ATTACHMENT_DIR", str(tmp_path / "attachments"))
    return routes


def _fake_handler():
    """Minimal handler mock -- `j()` writes JSON to `handler.wfile`."""
    handler = type("FakeHandler", (), {})()
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: None
    handler.send_header = lambda key, value: None
    handler.end_headers = lambda: None
    return handler


def _fake_handler_result(handler):
    return json.loads(handler.wfile.getvalue())


def test_sessions_cleanup_sweep_removes_bak_and_chunk_dir_for_orphan(cleanup_env, session_store, tmp_path):
    routes = cleanup_env

    # An orphan: was segmented, then cleared to zero messages and retitled
    # "Untitled" -- exactly what the phase-1 sweep targets for removal.
    p = _segmented(session_store, "orphan1")
    s = M.Session.load("orphan1")
    s.messages = []
    s.title = "Untitled"
    s.save()
    p.with_suffix(".json.bak").write_text("{}", encoding="utf-8")
    assert (session_store / "orphan1.msgs").is_dir()

    # Give it an attachment dir too, so the check on whether that peer gets
    # dropped alongside head/.bak/chunks is actually exercised.
    from api.upload import _session_attachment_dir
    attach_dir = _session_attachment_dir("orphan1")
    attach_dir.mkdir(parents=True, exist_ok=True)
    (attach_dir / "note.txt").write_text("x", encoding="utf-8")

    # A healthy sibling with real messages -- must survive the sweep.
    _segmented(session_store, "healthy1")
    assert (session_store / "healthy1.msgs").is_dir()

    handler = _fake_handler()
    routes._handle_sessions_cleanup(handler, {})
    result = _fake_handler_result(handler)

    assert result["ok"] is True
    assert result["cleaned"] == 1, "cleaned counts sessions, not files"
    assert not (session_store / "orphan1.json").exists()
    assert not (session_store / "orphan1.json.bak").exists()
    assert not (session_store / "orphan1.msgs").exists(), "chunk dir leaked"
    assert not attach_dir.exists(), "attachment dir leaked"
    assert (session_store / "healthy1.json").exists()
    assert (session_store / "healthy1.msgs").is_dir(), "healthy sibling touched"
