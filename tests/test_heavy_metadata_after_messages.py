"""The unbounded metadata blobs serialize AFTER `messages`, so the cheap prefix
stays small no matter how large they grow.

#5854 moved `anchor_activity_scenes` behind `messages` for exactly this reason
and left six other unbounded fields in front of it. One of them --
`compression_anchor_summary`, 73,192 bytes on a production session -- pushed
the `messages` key past the 64 KB prefix budget and turned every sidebar poll
into a full 112 MB parse (the #4633 recurrence). The 1 MiB backstop added since
buys headroom; it does not remove the mechanism, because these fields grow for
the life of a session. This does: nothing unbounded is written before
`messages` any more.

Moved (all six grow without bound, and all six are read only off fully-loaded
sessions):
    compression_anchor_summary, compression_anchor_details,
    context_engine_state, compression_recovery, gateway_routing_history,
    composer_draft
Deliberately NOT moved: `share_token`, `process_wakeup_pause` -- small scalars
that some prefix reader may legitimately want.

A metadata-only stub therefore never carries the six; it never did reliably
(any of them could already overflow the prefix), and every reader was audited on
this tree: `compact()` emits them for whatever object it is given; the UI reads
them only off `S.session`, which is only ever assigned from a full load (ui.js
never requests a metadata-only /api/session); the pin-quota helper reads other
fields off `compact()`; the eviction check consults `composer_draft` only after
`_loaded_metadata_only` has already returned; and the routes that read them
obtain their session with `get_session(sid)` -- a full load.

Files already on disk in the old layout keep loading (a full load reads the
whole file; the metadata path falls back) and heal into the new layout on their
first save.
"""
import json
from collections import OrderedDict

import pytest

import api.models as M

MOVED = ("compression_anchor_summary", "compression_anchor_details", "context_engine_state",
         "compression_recovery", "gateway_routing_history", "composer_draft")


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    return sdir


def _populated(session_store, sid, *, summary_bytes=4000, n_msgs=3):
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent), model="glm",
                  messages=[{"role": "user", "content": f"m{i}"} for i in range(n_msgs)])
    s.compression_anchor_summary = "S" * summary_bytes
    s.compression_anchor_details = {"engine": "x", "notes": "N" * 500}
    s.context_engine_state = {"window": [1, 2, 3], "blob": "C" * 500}
    s.compression_recovery = {"action": "restore", "reason": "R" * 200}
    s.gateway_routing_history = [{"hop": i, "model": f"m{i}"} for i in range(20)]
    s.composer_draft = {"text": "D" * 300, "files": []}
    s.share_token = "tok-123"
    s.anchor_activity_scenes = {"scene0": {"version": 1, "updated_at": 1.0, "scene": {}}}
    s.save()
    return s


def _key_offsets(raw):
    return {k: raw.find(f'"{k}"') for k in
            MOVED + ("session_id", "message_count", "anchor_scene_index", "messages",
                     "tool_calls", "anchor_activity_scenes", "share_token", "process_wakeup_pause")}


# ── the layout ──────────────────────────────────────────────────────────────

def test_unbounded_fields_serialize_after_messages(session_store):
    s = _populated(session_store, "l1")
    off = _key_offsets(s.path.read_text(encoding="utf-8"))
    assert all(off[k] > 0 for k in MOVED), "every moved field must still be persisted"
    for k in MOVED:
        assert off[k] > off["messages"], f"{k} at {off[k]} must serialize AFTER messages at {off['messages']}"


def test_prefix_fields_still_precede_messages(session_store):
    """What load_metadata_only needs stays in front: identity, the count, the
    scene fingerprint -- and the small scalars that were NOT moved."""
    s = _populated(session_store, "l2")
    off = _key_offsets(s.path.read_text(encoding="utf-8"))
    for k in ("session_id", "message_count", "anchor_scene_index", "share_token", "process_wakeup_pause"):
        assert -1 < off[k] < off["messages"], f"{k} must stay in the metadata prefix"
    assert off["message_count"] < off["anchor_scene_index"], "#5854 order: count before fingerprint"


def test_moved_set_is_exactly_the_six_unbounded_fields():
    """Pin the set. Adding a small scalar here quietly removes it from every
    metadata-only stub; removing an unbounded one reopens #4633."""
    assert set(M._HEAVY_METADATA_TAIL_FIELDS) == set(MOVED)


# ── the point: the prefix cannot be overflowed by these any more ────────────

def test_prefix_stays_small_with_a_summary_far_beyond_the_backstop(session_store):
    """Two MiB of summary -- past the 1 MiB backstop that would otherwise be
    the last line of defence -- and the cheap read still succeeds in a few KB."""
    s = _populated(session_store, "l3", summary_bytes=2 * 1024 * 1024, n_msgs=5)
    prefix = M._read_metadata_json_prefix(s.path)
    assert prefix is not None, "the cheap prefix must not fall back to a full parse"
    assert len(prefix.encode("utf-8")) < 8192, f"prefix is {len(prefix):,} bytes; the blob leaked into it"
    parsed = json.loads(prefix)
    assert parsed["message_count"] == 5
    assert "compression_anchor_summary" not in parsed


def test_metadata_only_stub_carries_defaults_for_the_moved_fields(session_store):
    """The contract for stubs, stated: the six are absent (their constructor
    defaults), never a partial or stale value."""
    _populated(session_store, "l4")
    stub = M.Session.load_metadata_only("l4")
    assert stub is not None and stub._loaded_metadata_only is True
    assert stub.compression_anchor_summary is None
    assert stub.compression_anchor_details == {}
    assert stub.context_engine_state == {}
    assert stub.compression_recovery == {}
    assert stub.gateway_routing_history == []
    assert stub.composer_draft == {}
    assert stub.share_token == "tok-123", "the small scalar that stayed must still reach the stub"


# ── nothing lost ────────────────────────────────────────────────────────────

def test_full_load_round_trips_all_six_fields(session_store):
    s = _populated(session_store, "l5", summary_bytes=90000)
    full = M.Session.load("l5")
    for k in MOVED:
        assert getattr(full, k) == getattr(s, k), f"{k} did not round-trip"
    assert len(full.messages) == 3


# ── files already on disk in the old layout ─────────────────────────────────

def _rewrite_in_old_layout(path):
    """Emulate a sidecar written before this change: the six unbounded fields
    right after `updated_at`, i.e. BEFORE message_count and messages."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    tail = ("messages", "tool_calls", "anchor_activity_scenes")
    out = {}
    for k, v in doc.items():
        if k in MOVED or k in tail:
            continue
        out[k] = v
        if k == "updated_at":
            for h in MOVED:
                if h in doc:
                    out[h] = doc[h]
    for k in tail:
        if k in doc:
            out[k] = doc[k]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def test_old_layout_file_still_loads_and_serves_metadata(session_store):
    s = _populated(session_store, "l6", summary_bytes=90000, n_msgs=4)
    _rewrite_in_old_layout(s.path)
    raw = s.path.read_text(encoding="utf-8")
    assert raw.find('"compression_anchor_summary"') < raw.find('"messages"'), "fixture must be old-layout"

    full = M.Session.load("l6")
    assert full.compression_anchor_summary == "S" * 90000 and len(full.messages) == 4

    stub = M.Session.load_metadata_only("l6")
    assert stub is not None
    assert (stub._metadata_message_count or len(stub.messages)) == 4


def test_old_layout_file_heals_into_the_new_layout_on_its_first_save(session_store):
    s = _populated(session_store, "l7", summary_bytes=90000)
    _rewrite_in_old_layout(s.path)

    M.Session.load("l7").save(touch_updated_at=False, skip_index=True)

    off = _key_offsets(s.path.read_text(encoding="utf-8"))
    for k in MOVED:
        assert off[k] > off["messages"], f"{k} still precedes messages after a save"
