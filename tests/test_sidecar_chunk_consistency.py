"""When memory no longer matches the sealed chunks, re-seal rather than lie.

Every test here is a real code path that rewrites history in place, not a
hypothetical: the #2592 collapse-on-load, /api/session/clear, an intentional
shrink, and a truncation.
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
    return [{"role": "user", "ts": float(i), "content": f"m{i}"} for i in range(start, start + n)]


def _segmented(session_store, sid, n=22):
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, n))
    s.save()
    assert json.loads((session_store / f"{sid}.json").read_bytes())["message_chunks"]
    return M.Session.load(sid)


def test_load_records_the_manifest_so_the_next_save_can_append(session_store):
    s = _segmented(session_store, "k1")
    assert M._sealed_total(s._message_chunks) > 0
    s.messages = s.messages + _msgs(100, 1)
    s.save()
    doc = json.loads((session_store / "k1.json").read_bytes())
    assert doc["message_count"] == 23
    assert [m["content"] for m in M.Session.load("k1").messages][-1] == "m100"


def test_truncating_into_sealed_territory_reseals_from_memory(session_store):
    s = _segmented(session_store, "k2", n=30)
    sealed_before = M._sealed_total(s._message_chunks)
    assert sealed_before > 8
    s.messages = s.messages[:8]  # shorter than the sealed prefix
    s.save()
    doc = json.loads((session_store / "k2.json").read_bytes())
    assert M._sealed_total(doc.get("message_chunks")) + len(doc["messages"]) == 8
    assert [m["content"] for m in M.Session.load("k2").messages] == [f"m{i}" for i in range(8)]


def test_rewriting_a_sealed_message_reseals_from_memory(session_store):
    s = _segmented(session_store, "k3", n=30)
    s.messages[0] = {"role": "user", "ts": 0.0, "content": "REPLACED-and-longer"}
    s.save()
    assert M.Session.load("k3").messages[0]["content"] == "REPLACED-and-longer"
    assert len(M.Session.load("k3").messages) == 30


def test_reseal_orphans_the_old_chunks_and_never_deletes_them(session_store):
    s = _segmented(session_store, "k4", n=30)
    old_files = sorted(p.name for p in (session_store / "k4.msgs").glob("*.json"))
    s.messages = s.messages[:8]
    s.save()
    now = sorted(p.name for p in (session_store / "k4.msgs").glob("*.json"))
    assert all(f in now for f in old_files), "a re-seal must not delete chunk files"
    doc = json.loads((session_store / "k4.json").read_bytes())
    named = {e["file"] for e in doc.get("message_chunks") or []}
    assert not (named & set(old_files)), "the new manifest must name only new files"


def test_clearing_all_messages_leaves_a_consistent_head(session_store):
    s = _segmented(session_store, "k5", n=30)
    s.messages = []
    s.save()
    doc = json.loads((session_store / "k5.json").read_bytes())
    assert doc["message_count"] == 0
    assert not doc.get("message_chunks")
    assert M.Session.load("k5").messages == []


def test_manifest_matches_memory_detects_each_divergence():
    msgs = _msgs(0, 6)
    manifest = [{"seq": 1, "file": "000001.json", "count": 4, "first_idx": 0, "sha256": "x",
                 "first_key": list(M._structural_key(msgs[0])),
                 "last_key": list(M._structural_key(msgs[3]))}]
    assert M._manifest_matches_memory(manifest, msgs) is True
    assert M._manifest_matches_memory(manifest, msgs[:2]) is False, "too short"
    edited = list(msgs)
    edited[0] = {"role": "user", "ts": 0.0, "content": "much longer content"}
    assert M._manifest_matches_memory(manifest, edited) is False, "first key changed"
    edited2 = list(msgs)
    edited2[3] = {"role": "assistant", "ts": 3.0, "content": "m3"}
    assert M._manifest_matches_memory(manifest, edited2) is False, "last key changed"
    assert M._manifest_matches_memory([], msgs) is True, "no manifest, nothing to contradict"


def test_collapse_on_load_reseals_and_still_backs_up_the_head(session_store):
    """#2592 + #1558 together on a segmented session: load() collapses adjacent
    partials and immediately saves the shorter transcript, which is a shrink, so
    the head must still be backed up AND the manifest must be re-sealed."""
    dup = {"role": "assistant", "content": "", "_partial": True, "timestamp": 123,
           "reasoning": "same reasoning",
           "_partial_tool_calls": [{"name": "execute_code", "args": {"code": "x"},
                                    "done": True, "is_error": True, "duration": 1.0}]}
    s = M.Session(session_id="k6", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 18) + [dup, dict(dup), dict(dup)])
    s.save()
    assert json.loads((session_store / "k6.json").read_bytes())["message_chunks"]

    loaded = M.Session.load("k6")

    assert sum(1 for m in loaded.messages if m.get("_partial")) == 1
    assert (session_store / "k6.json.bak").exists(), "a shrinking save must back the head up"
    again = M.Session.load("k6")
    assert len(again.messages) == len(loaded.messages)
