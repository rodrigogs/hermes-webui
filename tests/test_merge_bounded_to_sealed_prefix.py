# tests/test_merge_bounded_to_sealed_prefix.py
"""C1: the per-turn merge filters must not touch the sealed prefix.

Production 2026-09-17: session 5b7a1c15217f re-sealed twice in one minute at
the end of a streaming turn because these filters dropped a compaction marker
below the sealed boundary, shifting every later index. Each test names the
production change that would make it fail.
"""
import json
import logging
from collections import OrderedDict
from pathlib import Path

import pytest

import api.models as M
from api.streaming import _merge_display_messages_after_agent_result, _sealed_prefix_len_for


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


def _marker(i):
    # is_context_compression_marker: role not tool, text starts with "[context compaction"
    return {"role": "system", "timestamp": float(i), "content": "[CONTEXT COMPACTION] older turns were summarised"}


def _history(n, marker_at=None):
    out = [_msg(i, "user" if i % 2 == 0 else "assistant") for i in range(n)]
    if marker_at is not None:
        out[marker_at] = _marker(marker_at)
    return out


def _turn(history, i):
    """A normal append-only agent result: context == history (minus markers), plus user+assistant."""
    ctx = [m for m in history if not m["content"].startswith("[CONTEXT COMPACTION")]
    new = [_msg(i, "user", "question"), _msg(i + 1, "assistant", "answer")]
    return ctx, ctx + new, "question"


def test_default_is_byte_for_byte_todays_behaviour():
    """Fails if the default sealed_prefix_len stops being 0 or the prefix split changes output."""
    hist = _history(300, marker_at=40)
    ctx, result, text = _turn(hist, 300)
    out = _merge_display_messages_after_agent_result(list(hist), ctx, result, text)
    assert not any(m["content"].startswith("[CONTEXT COMPACTION") for m in out), "today drops the marker"
    assert out[-2:] == result[-2:]


def test_marker_below_the_boundary_survives_and_the_delta_is_identical():
    """Fails if the filters run over the whole array again (the marker at 40 vanishes)
    or if bounding changes the appended delta."""
    hist = _history(3000, marker_at=40)
    ctx, result, text = _turn(hist, 3000)
    bounded = _merge_display_messages_after_agent_result(list(hist), ctx, result, text, sealed_prefix_len=2500)
    clean = _history(3000)
    ctx_c, result_c, _ = _turn(clean, 3000)
    unbounded_clean = _merge_display_messages_after_agent_result(list(clean), ctx_c, result_c, text)
    assert bounded[:2500] == hist[:2500], "the sealed prefix is untouched, marker included"
    assert bounded[40]["content"].startswith("[CONTEXT COMPACTION")
    assert bounded[2500:] == unbounded_clean[2500:], "the suffix and the delta are what today produces"


def test_partial_dedupe_is_bounded_too():
    """Fails if the _partial dedupe scans below the boundary: the duplicate at 10 would be dropped."""
    hist = _history(100)
    dup = {"role": "assistant", "timestamp": 10.0, "content": "half an answer", "_partial": True}
    hist[10] = dict(dup)
    hist[95] = dict(dup, timestamp=95.0)
    ctx, result, text = _turn(hist, 100)
    out = _merge_display_messages_after_agent_result(list(hist), ctx, result, text, sealed_prefix_len=50)
    assert out[10] == hist[10], "below the boundary: kept"
    assert sum(1 for m in out[50:] if m.get("_partial")) <= 1, "above the boundary: deduped as today"


def test_backfill_below_the_boundary_is_logged_not_forbidden(caplog):
    """Fails if a below-boundary backfill stops logging, or if it drops any prefix row."""
    hist = _history(200)
    hidden = _msg(150, "assistant", "a turn that was only in context")
    ctx = list(hist[:150]) + [hidden] + list(hist[150:])
    result = ctx + [_msg(200, "user", "q"), _msg(201, "assistant", "a")]
    with caplog.at_level(logging.INFO, logger="api.streaming"):
        out = _merge_display_messages_after_agent_result(list(hist), ctx, result, "q", sealed_prefix_len=180)
    assert hidden in out, "the backfill still restores the hidden turn"
    assert all(m in out for m in hist[:180]), "no prefix row is lost"
    assert any("below the sealed boundary" in r.getMessage() for r in caplog.records)


def test_sealed_prefix_len_for_reads_the_manifest(session_store):
    """Fails if the helper stops deriving from _message_chunks (returns 0 for a segmented session)."""
    s = _segmented(session_store, "p1", n=30)
    assert _sealed_prefix_len_for(s) == 25
    assert _sealed_prefix_len_for(M.Session(session_id="p2", title="T", workspace=".", model="glm")) == 0


def test_end_to_end_a_marker_drop_no_longer_reseals(session_store, monkeypatch):
    """THE production case. Fails if C1 is reverted: with the whole-array filter the marker at 3
    disappears, the sealed prefix diverges, and save() writes a new chunk."""
    msgs = [_msg(i) for i in range(30)]
    msgs[3] = _marker(3)                    # sealed WITH the marker in place. (Editing an interior message
    s = M.Session(session_id="e2e", title="T", workspace=str(session_store.parent), model="glm",
                  messages=msgs)            # AFTER sealing would not persist: only boundary keys are checked.)
    s.save(touch_updated_at=False, skip_index=True)
    M.SESSIONS.clear()
    s = M.Session.load("e2e")
    before = _chunks(session_store, "e2e")
    sealed = _sealed_prefix_len_for(s)
    assert sealed == 25 and s.messages[3]["content"].startswith("[CONTEXT COMPACTION")
    ctx = [m for m in s.messages if not m["content"].startswith("[CONTEXT COMPACTION")]
    result = ctx + [_msg(30, "user", "q"), _msg(31, "assistant", "a")]
    s.messages = _merge_display_messages_after_agent_result(list(s.messages), ctx, result, "q",
                                                            sealed_prefix_len=sealed)
    seals = []
    real = M._seal_chunk
    monkeypatch.setattr(M, "_seal_chunk", lambda sid, seq, msgs, first_idx: seals.append(seq) or real(sid, seq, msgs, first_idx))
    s.save(touch_updated_at=False, skip_index=True)
    assert seals == [], f"a bounded merge must not cause a re-seal; sealed {seals}"
    assert _chunks(session_store, "e2e") == before
    # and the unbounded call DOES (proves the test has teeth)
    M.SESSIONS.clear(); s2 = M.Session.load("e2e")
    s2.messages = _merge_display_messages_after_agent_result(list(s2.messages), ctx, result + [_msg(32, "user", "q2"), _msg(33, "assistant", "a2")], "q2")
    seals.clear(); s2.save(touch_updated_at=False, skip_index=True)
    assert seals, "with sealed_prefix_len=0 the marker is dropped and save() re-seals"
