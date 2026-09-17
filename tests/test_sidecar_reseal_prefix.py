"""C2a/C2b: a re-seal keeps every chunk that still matches and reseals from the first that does not."""
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


def _five_chunks(session_store, sid):
    """Seal 5 times by growing past the threshold 5 times: 5 chunks of 10 + a tail of 5."""
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent), model="glm", messages=[_msg(i) for i in range(15)])
    s.save(touch_updated_at=False, skip_index=True)
    for k in range(4):
        s.messages = s.messages + [_msg(15 + k * 10 + j) for j in range(10)]
        s.save(touch_updated_at=False, skip_index=True)
    man = _manifest(session_store, sid)
    assert len(man) == 5, f"fixture must have 5 chunks, has {len(man)}: {[e['count'] for e in man]}"
    return s


def test_v2_key_sees_a_reasoning_edit_and_v1_does_not():
    """Fails if _structural_key_v2 stops hashing the field attach_display_reasoning writes ('reasoning')."""
    a = _msg(1, "assistant", "same text")
    b = dict(a, reasoning="a thought that was attached later")
    assert M._structural_key(a) == M._structural_key(b)
    assert M._structural_key_v2(a) != M._structural_key_v2(b)
    assert M._structural_key_v2(a)[:3] == M._structural_key(a)


def test_seal_chunk_writes_v2_keys(session_store):
    """Fails if _seal_chunk goes back to v1 keys (no key_v, 3-tuples)."""
    _segmented(session_store, "k2", n=30)
    e = _manifest(session_store, "k2")[0]
    assert e["key_v"] == 2 and len(e["first_key"]) == 4 and len(e["last_key"]) == 4


def test_entry_key_version_is_honoured():
    """Fails if comparison ignores key_v: a v1 entry must be checked with the v1 key, a v2 with v2."""
    msgs = [_msg(i, "assistant") for i in range(3)]
    v1 = {"seq": 1, "file": "000001.json", "count": 3, "first_idx": 0, "sha256": "x",
          "first_key": list(M._structural_key(msgs[0])), "last_key": list(M._structural_key(msgs[2]))}
    edited = [dict(m) for m in msgs]; edited[2]["reasoning"] = "new"
    assert M._matching_manifest_prefix([v1], edited) == [v1], "v1 entry: a reasoning edit is invisible, must still match"
    v2 = dict(v1, key_v=2, first_key=list(M._structural_key_v2(msgs[0])), last_key=list(M._structural_key_v2(msgs[2])))
    assert M._matching_manifest_prefix([v2], msgs) == [v2]
    assert M._matching_manifest_prefix([v2], edited) == [], "v2 entry: the same edit diverges"
    assert M._matching_manifest_prefix([dict(v2, key_v=7)], msgs) == [], "unknown version: unverifiable, ends the prefix"


def test_matching_prefix_keeps_chunks_before_the_divergence(session_store):
    """Fails if _matching_manifest_prefix returns all-or-nothing: a shrink into chunk 3 must keep 1-2."""
    s = _five_chunks(session_store, "mp")
    man = _manifest(session_store, "mp")
    cut = man[2]["first_idx"] + 3                       # inside chunk 3
    shorter = s.messages[:cut] + s.messages[cut + 2:]   # remove two messages inside chunk 3
    keep = M._matching_manifest_prefix(man, shorter)
    assert [e["file"] for e in keep] == [man[0]["file"], man[1]["file"]]
    assert M._manifest_matches_memory(man, s.messages) is True
    assert M._manifest_matches_memory(man, shorter) is False


def test_reusable_prefix_stops_at_a_missing_file(session_store):
    """Fails if _reusable_manifest_prefix stops stat-ing chunk files."""
    s = _five_chunks(session_store, "rp")
    man = _manifest(session_store, "rp")
    (session_store / "rp.msgs" / man[1]["file"]).unlink()
    keep = M._reusable_manifest_prefix("rp", man, s.messages)
    assert [e["file"] for e in keep] == [man[0]["file"]]


def test_manifest_matches_memory_contract_unchanged():
    """Fails if the wrapper changes the bool contract (empty manifest True; count<=0 False; short array False)."""
    assert M._manifest_matches_memory([], [_msg(0)]) is True
    bad = {"seq": 1, "file": "000001.json", "count": 0, "first_idx": 0, "sha256": "x", "first_key": [1], "last_key": [1]}
    assert M._manifest_matches_memory([bad], [_msg(0)]) is False
    ok = {"seq": 1, "file": "000001.json", "count": 2, "first_idx": 0, "sha256": "x",
          "first_key": list(M._structural_key(_msg(0))), "last_key": list(M._structural_key(_msg(1)))}
    assert M._manifest_matches_memory([ok], [_msg(0)]) is False
    assert M._manifest_matches_memory([ok], [_msg(0), _msg(1)]) is True
