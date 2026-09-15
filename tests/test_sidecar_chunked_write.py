"""Sealing on save: threshold, ordering, and what a crash can leave behind."""
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
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(M, "_SIDECAR_TAIL_KEEP", 4)
    return sdir


def _msgs(start, n, fill=""):
    return [{"role": "user", "ts": float(i), "content": f"m{i}{fill}"} for i in range(start, start + n)]


def _sess(session_store, sid, msgs):
    return M.Session(session_id=sid, title="T", workspace=str(session_store.parent),
                     model="glm", messages=list(msgs))


def test_below_the_threshold_nothing_is_sealed(session_store):
    s = _sess(session_store, "w1", _msgs(0, 9))
    s.save()
    doc = json.loads((session_store / "w1.json").read_bytes())
    assert "message_chunks" not in doc or doc["message_chunks"] == []
    assert len(doc["messages"]) == 9
    assert not (session_store / "w1.msgs").exists(), "no directory until it is needed"


def test_crossing_the_message_threshold_seals_all_but_tail_keep(session_store):
    s = _sess(session_store, "w2", _msgs(0, 11))
    s.save()
    doc = json.loads((session_store / "w2.json").read_bytes())
    assert [e["count"] for e in doc["message_chunks"]] == [7], "11 - TAIL_KEEP(4) = 7 sealed"
    assert [m["content"] for m in doc["messages"]] == ["m7", "m8", "m9", "m10"]
    assert doc["message_count"] == 11, "message_count stays the TOTAL"
    body = json.loads((session_store / "w2.msgs" / "000001.json").read_bytes())
    assert body["session_id"] == "w2" and body["seq"] == 1 and body["first_idx"] == 0
    assert [m["content"] for m in body["messages"]] == [f"m{i}" for i in range(7)]


def test_crossing_the_byte_threshold_seals_even_with_few_messages(session_store, monkeypatch):
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_BYTES", 4096)
    s = _sess(session_store, "w3", _msgs(0, 6, fill="X" * 2000))
    s.save()
    doc = json.loads((session_store / "w3.json").read_bytes())
    assert doc["message_chunks"], "a few very large messages must seal too"
    assert len(doc["messages"]) == 4


def test_round_trip_after_sealing_is_lossless(session_store):
    s = _sess(session_store, "w4", _msgs(0, 25))
    s.save()
    loaded = M.Session.load("w4")
    assert [m["content"] for m in loaded.messages] == [f"m{i}" for i in range(25)]


def test_successive_saves_seal_successive_chunks_and_never_rewrite_one(session_store):
    s = _sess(session_store, "w5", _msgs(0, 11))
    s.save()
    first = (session_store / "w5.msgs" / "000001.json").read_bytes()
    s.messages = _msgs(0, 22)
    s.save()
    doc = json.loads((session_store / "w5.json").read_bytes())
    assert [e["seq"] for e in doc["message_chunks"]] == [1, 2]
    assert (session_store / "w5.msgs" / "000001.json").read_bytes() == first, \
        "a sealed chunk is immutable"
    assert [m["content"] for m in M.Session.load("w5").messages] == [f"m{i}" for i in range(22)]


def test_the_head_is_written_after_the_chunk(session_store, monkeypatch):
    """Crash safety, stated as an ordering assertion: at the moment the chunk
    lands the head must not yet reference it. The reverse order would leave a
    manifest naming a file that does not exist -- unrecoverable rather than
    merely untidy."""
    order = []
    real_seal = M._seal_chunk
    real_replace = M._safe_replace

    def seal(sid, seq, msgs, first_idx):
        out = real_seal(sid, seq, msgs, first_idx)
        order.append("chunk")
        return out

    def replace(src, dst):
        if str(dst).endswith("w6.json"):
            order.append("head")
        return real_replace(src, dst)

    monkeypatch.setattr(M, "_seal_chunk", seal)
    monkeypatch.setattr(M, "_safe_replace", replace)
    _sess(session_store, "w6", _msgs(0, 11)).save()
    assert order == ["chunk", "head"]


def test_a_failed_seal_leaves_the_previous_head_intact(session_store, monkeypatch):
    s = _sess(session_store, "w7", _msgs(0, 5))
    s.save()
    before = (session_store / "w7.json").read_bytes()
    monkeypatch.setattr(M, "_seal_chunk", lambda *a, **k: None)
    s.messages = _msgs(0, 20)
    s.save()
    doc = json.loads((session_store / "w7.json").read_bytes())
    assert len(doc["messages"]) == 20, "sealing failed, so everything stays in the head"
    assert not doc.get("message_chunks"), "no manifest entry for a chunk that did not land"
    assert [m["content"] for m in M.Session.load("w7").messages] == [f"m{i}" for i in range(20)]


def test_sealing_disabled_writes_everything_into_the_head(session_store, monkeypatch):
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_MSGS", 0)
    s = _sess(session_store, "w8", _msgs(0, 50))
    s.save()
    doc = json.loads((session_store / "w8.json").read_bytes())
    assert len(doc["messages"]) == 50
    assert not (session_store / "w8.msgs").exists()


def test_a_segmented_session_still_saves_with_sealing_disabled(session_store, monkeypatch):
    s = _sess(session_store, "w9", _msgs(0, 11))
    s.save()
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_MSGS", 0)
    loaded = M.Session.load("w9")
    loaded.messages = loaded.messages + _msgs(11, 2)
    loaded.save()
    assert [m["content"] for m in M.Session.load("w9").messages] == [f"m{i}" for i in range(13)]


def test_manifest_sits_inside_the_cheap_metadata_prefix(session_store):
    s = _sess(session_store, "w10", _msgs(0, 25))
    s.save()
    raw = (session_store / "w10.json").read_text(encoding="utf-8")
    assert raw.find('"message_chunks"') < raw.find('"messages"')
    prefix = M._read_metadata_json_prefix(session_store / "w10.json")
    assert prefix is not None
    assert json.loads(prefix)["message_count"] == 25


# ── #1558 shrink-backup guard must count SEALED + tail, not just the tail ───

def test_shrink_through_a_fresh_object_after_segmenting_backs_up_the_pre_shrink_total(session_store):
    """Regression: a fresh Session object (no remembered on-disk identity)
    takes save()'s slow fallback path, which used to read only the head's
    `messages` (the tail) to learn the existing count. Against a segmented
    head that undercounts the total, so a genuine shrink read as a grow and
    the #1558 guard silently stopped backing anything up."""
    s = _sess(session_store, "w11", _msgs(0, 30))
    s.save()
    doc_before = json.loads((session_store / "w11.json").read_bytes())
    assert M._sealed_total(doc_before["message_chunks"]) == 26
    assert len(doc_before["messages"]) == 4
    # A fresh object has no _disk_identity_seen, forcing the slow fallback.
    # 4 messages stays at the tail-keep boundary (not > _SIDECAR_TAIL_KEEP), so
    # this save does not itself reseal -- isolating the count fix from
    # unrelated chunk-overwrite behaviour.
    fresh = _sess(session_store, "w11", _msgs(0, 4))  # genuine shrink: 4 < 30
    fresh.save()
    bak_path = session_store / "w11.json.bak"
    assert bak_path.exists(), "a shrink behind a segmented head must still be backed up"
    bak_doc = json.loads(bak_path.read_bytes())
    assert M._sealed_total(bak_doc.get("message_chunks")) + len(bak_doc["messages"]) == 30, \
        "the .bak must describe the pre-shrink TOTAL (sealed + tail), not just the tail"
    live_doc = json.loads((session_store / "w11.json").read_bytes())
    assert len(live_doc["messages"]) == 4


def test_grow_through_a_fresh_object_after_segmenting_produces_no_backup(session_store):
    """The other direction: a grow behind a segmented head must stay
    backup-free, or every save on a segmented session would start writing
    backups."""
    s = _sess(session_store, "w12", _msgs(0, 30))
    s.save()
    fresh = _sess(session_store, "w12", _msgs(0, 31))  # grow: 31 > 30
    fresh.save()
    assert not (session_store / "w12.json.bak").exists(), \
        "a grow behind a segmented head must not produce a backup"


def test_unsegmented_shrink_through_a_fresh_object_is_unaffected(session_store):
    """Guard against over-fixing: an unsegmented head's fallback count must
    stay exactly `len(messages)`, unaffected by the sealed-total addition."""
    s = _sess(session_store, "w13", _msgs(0, 5))  # below every threshold; never segments
    s.save()
    doc = json.loads((session_store / "w13.json").read_bytes())
    assert "message_chunks" not in doc or not doc["message_chunks"]
    fresh = _sess(session_store, "w13", _msgs(0, 2))  # shrink: 2 < 5
    fresh.save()
    bak_path = session_store / "w13.json.bak"
    assert bak_path.exists()
    bak_doc = json.loads(bak_path.read_bytes())
    assert len(bak_doc["messages"]) == 5, "unsegmented fallback must still count messages directly"
