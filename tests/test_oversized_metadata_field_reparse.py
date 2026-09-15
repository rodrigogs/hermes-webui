"""The #4633 churn returns through a field #5854 did not move -- for sidecars
already on disk in the layout written before the heavy-field reorder.

#5854 stopped `anchor_activity_scenes` from overflowing the 64 KB metadata
prefix, and cached the authoritative facts of a LEGACY sidecar so an unchanged
one is full-parsed at most once. Both halves were gated on the file being legacy
(`'anchor_scene_index' not in data`). A MODERN sidecar that overflowed the
prefix for any OTHER reason therefore got neither protection: the cheap read
failed, `load_metadata_only()` full-parsed, and nothing was cached -- so it
re-parsed on every poll. That is exactly the #4633 allocation churn the earlier
fix removed.

Measured in production on 2026-09-14 (Mac docker stack, 5.77 GiB colima VM):

  /data/hermes/webui/sessions/fa3bca34a0c6.json   111,857,574 bytes, 135,634 msgs
    "compression_anchor_summary"  offset      1,007   (73,192 bytes long)
    "message_count"               offset     75,446
    "anchor_scene_index"          offset     75,473   <- file IS modern
    "messages"                    offset     76,603   <- past the 65,536 budget

Three layers now stand between that layout and a re-parse, and this file tests
the two that protect files ALREADY ON DISK in the old layout:
  * a 1 MiB prefix backstop (`_METADATA_PREFIX_MAX_BYTES`) serves them cheaply,
  * a file whose blob exceeds even that heals into the new layout on its first
    save, because save() now writes the unbounded fields AFTER `messages`
    (tests/test_heavy_metadata_after_messages.py covers the new layout itself).

The fixture writes the OLD layout by hand on purpose: `save()` no longer
produces it, and these tests are about what is on disk today.
"""
import json
from collections import OrderedDict

import pytest

import api.models as M

_HEAVY = ("compression_anchor_summary", "compression_anchor_details", "context_engine_state",
          "compression_recovery", "gateway_routing_history", "composer_draft")


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    sdir.mkdir(parents=True, exist_ok=True)
    return sdir


def _rewrite_in_old_layout(path):
    """Move the unbounded fields to right after `updated_at` -- BEFORE
    message_count, anchor_scene_index and messages -- as save() wrote them
    before the reorder. Production's summary sat at offset ~1,007."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    tail = ("messages", "tool_calls", "anchor_activity_scenes")
    out = {}
    for k, v in doc.items():
        if k in _HEAVY or k in tail:
            continue
        out[k] = v
        if k == "updated_at":
            for h in _HEAVY:
                if h in doc:
                    out[h] = doc[h]
    for k in tail:
        if k in doc:
            out[k] = doc[k]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_old_layout_oversized(session_store, sid, n_msgs=5, summary_bytes=80000):
    """A MODERN sidecar (carries anchor_scene_index) in the OLD layout, whose
    metadata prefix overflows because of one oversized field -- production's
    shape on 2026-09-14. Scenes are tiny: the overflow must come from a field
    #5854 never moved, not from the scene bodies it did."""
    s = M.Session(session_id=sid, title="Oversized", workspace=str(session_store.parent),
                  model="glm", messages=[{"role": "user", "content": f"m{i}"} for i in range(n_msgs)])
    s.anchor_activity_scenes = {"scene0": {"version": 1, "updated_at": 1000.0,
                                          "scene": {"activity_rows": []}}}
    s.compression_anchor_summary = "Z" * summary_bytes
    s.save()
    _rewrite_in_old_layout(s.path)
    return s


def test_fixture_is_modern_and_overflows_the_old_budget(session_store):
    """Guard the premise: a modern file (anchor_scene_index present, scenes after
    messages) whose "messages" key sits beyond the old 64 KB budget. If this
    ever fails, the tests below stop covering what they claim to."""
    s = _make_old_layout_oversized(session_store, "mod1")
    raw = s.path.read_text(encoding="utf-8")
    assert '"anchor_scene_index"' in raw, "fixture must be MODERN, not legacy"
    ci, xi, mi, si = (raw.find(f'"{k}"') for k in
                      ("compression_anchor_summary", "anchor_scene_index", "messages", "anchor_activity_scenes"))
    assert -1 < ci < xi < mi < si, "old layout: summary < scene_index < messages < scenes"
    assert mi > 65536, f'"messages" at {mi} must exceed the old 64 KB budget to reproduce #4633'


def test_backstop_serves_an_old_layout_file_with_an_oversized_field(session_store):
    """Layer one. The 1 MiB backstop reads past a 80 KB blob and the file is
    served cheaply without any rewrite."""
    s = _make_old_layout_oversized(session_store, "mod2", n_msgs=7)
    prefix = M._read_metadata_json_prefix(s.path)
    assert prefix is not None, "cheap prefix must succeed, not fall back to a full parse"
    parsed = json.loads(prefix)
    assert {"session_id", "title", "created_at", "updated_at"}.issubset(parsed.keys())
    assert parsed["message_count"] == 7
    assert "messages" not in parsed


def test_old_layout_beyond_the_backstop_reparses_until_its_first_save(session_store, monkeypatch):
    """Layer two. A blob past even the 1 MiB backstop defeats the cheap read, and
    a failed cheap read yields no metadata to build a stub from -- so such a
    file DOES re-parse per read. That is the residual, stated plainly. It lasts
    exactly until the file's first save, which rewrites it in the new layout;
    from then on the metadata path never touches the array again."""
    M._LEGACY_SIDECAR_FACTS.clear()
    s = _make_old_layout_oversized(session_store, "mod3", n_msgs=5,
                                   summary_bytes=M._METADATA_PREFIX_MAX_BYTES + 200_000)
    assert M._read_metadata_json_prefix(s.path) is None, "fixture must defeat the backstop"

    calls = {"n": 0}
    real_load = M.Session.load.__func__

    def _counting_load(cls, sid, *a, **k):
        if sid == "mod3":
            calls["n"] += 1
        return real_load(cls, sid, *a, **k)

    monkeypatch.setattr(M.Session, "load", classmethod(_counting_load))

    for _ in range(3):
        stub = M.Session.load_metadata_only("mod3")
        assert stub is not None and (stub._metadata_message_count or len(stub.messages)) == 5
    assert calls["n"] >= 1, "before its first save an over-backstop old-layout file still re-parses"

    # The heal: one ordinary save rewrites the file with the blob after messages.
    M.Session.load("mod3").save(touch_updated_at=False, skip_index=True)
    calls["n"] = 0
    for _ in range(3):
        stub = M.Session.load_metadata_only("mod3")
        assert stub is not None and (stub._metadata_message_count or len(stub.messages)) == 5
    assert calls["n"] == 0, f"after its first save the file was still full-loaded {calls['n']}x"
