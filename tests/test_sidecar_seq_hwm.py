"""C2c: a chunk number, once used, is never handed out again (spec 2026-09-17 §3.6).

All three refuters of the design found the same hole by different routes: a
deleter frees a number, _next_chunk_seq (disk max + 1) reissues it, and a stale
manifest -- in a cached object, a .bak, an operator's copy -- names a file that
exists with different messages.
"""
import json
from collections import OrderedDict
from pathlib import Path

import pytest

import api.models as M
from api import sidecar_maintenance as SM


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


def test_next_seq_honours_the_high_water_mark(session_store):
    """Fails if _next_chunk_seq ignores .seq_hwm (returns disk max + 1 = 2 here)."""
    _segmented(session_store, "h1", n=30)
    assert _chunks(session_store, "h1") == ["000001.json"]
    assert M._write_seq_hwm("h1", 7) is True
    assert M._read_seq_hwm("h1") == 7
    assert M._next_chunk_seq("h1") == 8


def test_hwm_never_lowers(session_store):
    """Fails if _write_seq_hwm overwrites with a smaller value."""
    _segmented(session_store, "h2", n=30)
    M._write_seq_hwm("h2", 9)
    M._write_seq_hwm("h2", 3)
    assert M._read_seq_hwm("h2") == 9


def test_corrupt_or_absent_hwm_reads_as_zero(session_store):
    """Fails if a bad .seq_hwm blocks sealing instead of degrading to today's behaviour."""
    _segmented(session_store, "h3", n=30)
    assert M._read_seq_hwm("h3") == 0
    (session_store / "h3.msgs" / ".seq_hwm").write_text("not a number", encoding="utf-8")
    assert M._read_seq_hwm("h3") == 0
    assert M._next_chunk_seq("h3") == 2


def test_a_freed_number_is_not_reissued_after_a_deleter_raised_the_mark(session_store):
    """THE scenario. Fails if the next seal after a deletion reuses the deleted number."""
    s = _segmented(session_store, "h4", n=30)
    # a deleter (gc/unchunk, Tasks 6-7) writes the mark first, then unlinks
    M._write_seq_hwm("h4", 1)
    (session_store / "h4.msgs" / "000001.json").unlink()
    s.messages = s.messages + [_msg(30 + i) for i in range(12)]   # forces a re-seal (chunk 1 is gone) + seal
    s.save(touch_updated_at=False, skip_index=True)
    names = _chunks(session_store, "h4")
    assert names and "000001.json" not in names, f"000001 must never come back: {names}"
    assert min(int(n[:6]) for n in names) >= 2


def test_a_chunk_vanished_before_save_never_has_its_number_reissued(session_store):
    """The reviewer's probe. Fails if `_sealed_layout` stops raising the mark for
    the manifest entries it drops: the released number is reissued to genuinely
    different bytes, and a stale cached Session then publishes a manifest naming
    a file whose sha256 no longer matches (55 of 61 messages lost, no .bak --
    the array grew, so the #1558 shrink guard did not fire -- and
    `recover_session` only stats existence, so it would not refuse either)."""
    s = _segmented(session_store, "vz", n=30)
    assert _chunks(session_store, "vz") == ["000001.json"]
    sha_before = _manifest(session_store, "vz")[0]["sha256"]

    # A SECOND live object for the same sid: a cached Session in a running webui.
    M.SESSIONS.clear()
    stale = M.Session.load("vz")
    assert [e["file"] for e in stale._message_chunks] == ["000001.json"]

    # An operator `rm`: the file goes, nothing raises the mark.
    (session_store / "vz.msgs" / "000001.json").unlink()

    # The first object saves: the TOP-of-save path discovers the gap and re-seals.
    s.messages = s.messages + [_msg(30)]
    s.save(touch_updated_at=False, skip_index=True)
    assert M._read_seq_hwm("vz") >= 1, "the discovery point must record the released number"
    man = _manifest(session_store, "vz")
    assert man and man[0]["file"] > "000001.json", f"000001 must never be reissued: {man!r:.120}"
    assert man[0]["sha256"] != sha_before, "the re-seal covers more messages, so different bytes"

    # The stale object saves an ORDINARY tail-only grow: it republishes its own
    # 000001.json entry verbatim if (and only if) that name came back.
    stale.messages = stale.messages + [_msg(31)]
    stale.save(touch_updated_at=False, skip_index=True)
    M.SESSIONS.clear()
    loaded = M.Session.load("vz")
    assert loaded.chunk_errors == [], f"the stale republish poisoned the head: {loaded.chunk_errors}"
    assert len(loaded.messages) == len(stale.messages)


def test_fsck_reports_the_mark_under_other_files_never_orphans(session_store):
    """Fails if fsck lists .seq_hwm as an orphan or crashes on it."""
    _segmented(session_store, "h5", n=30)
    M._write_seq_hwm("h5", 1)
    rep = SM.fsck_sessions(session_store)
    assert rep["orphans"] == []
    assert rep["other_files"] == [{"session_id": "h5", "count": 1}]
