"""The cost guard: a save must not scale with history.

This is the only test that fails if someone later re-introduces a full rewrite
-- for instance by moving the sealing behind a flag that defaults off, or by
reading the whole array to build the payload. It asserts BYTES WRITTEN rather
than wall time, so it is not flaky on a loaded machine.
"""
import json
import pathlib
from collections import OrderedDict

import pytest

import api.models as M


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    return sdir


def _spy_bytes_written(monkeypatch, session_dir):
    """Count bytes handed to open(...).write for anything under session_dir."""
    written = {"n": 0}
    real_open = open

    class _F:
        def __init__(self, f):
            self._f = f

        def write(self, data):
            written["n"] += len(data)
            return self._f.write(data)

        def __getattr__(self, name):
            return getattr(self._f, name)

        def __enter__(self):
            self._f.__enter__()
            return self

        def __exit__(self, *a):
            return self._f.__exit__(*a)

    def fake_open(file, mode="r", *a, **k):
        f = real_open(file, mode, *a, **k)
        # Filter on the SESSION DIRECTORY, not on a ".json" suffix. Every write
        # in save() lands on a tmp path first -- path.with_suffix('.tmp.<pid>.<tid>')
        # turns "cost.json" into "cost.tmp.123.456" and "000001.json" into
        # "000001.tmp.123.456". A ".json" filter therefore matches NOTHING and the
        # assertions below pass vacuously, which is worse than no test at all.
        if "w" in mode and str(file).startswith(str(session_dir)):
            return _F(f)
        return f

    monkeypatch.setattr("builtins.open", fake_open)
    return written


def test_save_cost_does_not_scale_with_history(session_store, monkeypatch):
    msgs = [{"role": "user", "timestamp": float(i), "content": f"message number {i} " + "x" * 200}
            for i in range(12000)]
    s = M.Session(session_id="cost", title="T", workspace=str(session_store.parent),
                  model="glm", messages=list(msgs))
    s.save()  # seals

    written = _spy_bytes_written(monkeypatch, session_store)
    s.messages = s.messages + [{"role": "user", "timestamp": 99999.0, "content": "one more"}]
    s.save()

    # What a FULL rewrite of this history costs, in the serialisation save()
    # actually writes (indent=2, ensure_ascii=False -- compact json.dumps is
    # ~6.6% smaller per message and would skew every number below).
    total_size = len(json.dumps(msgs, ensure_ascii=False, indent=2).encode("utf-8"))
    # The cap must sit strictly BETWEEN what a bounded save writes (the tail --
    # _SIDECAR_TAIL_KEEP messages plus one appended, plus metadata) and what a
    # full rewrite writes (all 12,000), with margin on both sides -- not a
    # fixed number chosen because it "looked reasonable" above both. A fixed
    # 4 MiB cap against this fixture's ~3.14 MiB full-rewrite size left a
    # regression to a full rewrite UNDER the cap, so the test could not fail
    # on the exact thing it exists to catch. Deriving the cap from the
    # fixture's own average message size and from `_SIDECAR_TAIL_KEEP` means
    # it tracks both if either changes, instead of drifting out of meaning.
    # Measured on this fixture: a sealed grow-save writes ~156 KB; an
    # unsealed (full-rewrite) grow-save writes ~3,630 KB. This formula lands
    # at ~750 KB -- about 4.8x above the sealed write and 4.7x below the full
    # rewrite.
    avg_msg_bytes = total_size / len(msgs)
    cap = avg_msg_bytes * (M._SIDECAR_TAIL_KEEP + 1) * 5 + 32 * 1024
    # The cap is linear in _SIDECAR_TAIL_KEEP. Grow that constant ~5x and the
    # cap climbs past this fixture's full rewrite, and the assertion below can
    # no longer fail -- the silent disarm this test was rewritten to remove,
    # via a different variable. So the test checks its own teeth first: if the
    # cap stops separating the two regimes, fail HERE, loudly, instead of
    # passing on a full rewrite. Grow the fixture when this trips.
    assert cap * 2 < total_size, (
        f"cap {cap:,.0f} is not well below this fixture's {total_size:,}-byte full "
        f"rewrite; the test can no longer tell a sealed save from a full rewrite"
    )
    assert written["n"] < cap, (
        f"a post-seal save wrote {written['n']:,} bytes (cap {cap:,.0f}) for a "
        f"{total_size:,}-byte history; sealing is not bounding the cost"
    )
    assert len(M.Session.load("cost").messages) == 12001, "and it is still lossless"


def test_repeated_saves_stay_bounded(session_store, monkeypatch):
    """The same regression, sustained: 20 saves in a row must stay bounded.

    The cap is DERIVED, for the reason spelled out in the test above. The fixed
    40 MiB it used to be sat 8.6% under what 20 full rewrites of this fixture
    cost (45.6 MB), so a PARTIAL regression -- anything that recovered even a
    tenth of the sealing benefit -- fit under it and this test passed while the
    cost it exists to bound had come back.
    """
    saves = 20
    msgs = [{"role": "user", "timestamp": float(i), "content": "y" * 300} for i in range(6000)]
    s = M.Session(session_id="cost2", title="T", workspace=str(session_store.parent), model="glm",
                  messages=list(msgs))
    s.save()
    written = _spy_bytes_written(monkeypatch, session_store)
    for i in range(saves):
        s.messages = s.messages + [{"role": "user", "timestamp": 10000.0 + i, "content": "z"}]
        s.save()

    # What ONE full rewrite of this history costs, in the serialisation save()
    # actually writes (indent=2, ensure_ascii=False).
    full_rewrite = len(json.dumps(msgs, ensure_ascii=False, indent=2).encode("utf-8"))
    avg_msg_bytes = full_rewrite / len(msgs)
    # A bounded save writes the tail: _SIDECAR_TAIL_KEEP messages, plus however
    # many this loop has appended since the last seal (at most `saves`). Times
    # the number of saves, times a small factor, plus per-save metadata slack.
    # Tracks _SIDECAR_TAIL_KEEP and the fixture's own message size instead of
    # being a round number chosen above one measurement.
    # Measured on this fixture: the 20 sealed saves write 3,890,743 bytes; 20
    # full rewrites cost 44,617,840. This formula lands at 10,322,559 -- 2.7x
    # above the sealed regime and 4.3x below the unsealed one, where the old
    # fixed 41,943,040 sat 6% under the unsealed cost.
    per_save = avg_msg_bytes * (M._SIDECAR_TAIL_KEEP + saves)
    cap = per_save * saves * 2.5 + saves * 32 * 1024
    # Teeth first, as in the test above: if the cap ever stops sitting well
    # below what the unsealed regime costs, fail HERE rather than pass on a
    # full rewrite per save. Grow the fixture when this trips.
    assert cap * 2 < saves * full_rewrite, (
        f"cap {cap:,.0f} is not well below the {saves * full_rewrite:,} bytes {saves} full "
        f"rewrites of this fixture cost; the test can no longer tell the two regimes apart"
    )
    assert written["n"] < cap, (
        f"{saves} saves wrote {written['n']:,} bytes (cap {cap:,.0f}) for a "
        f"{full_rewrite:,}-byte history; that is a full rewrite per save"
    )
    assert len(M.Session.load("cost2").messages) == 6020


def test_a_segmented_save_is_read_free(session_store, monkeypatch):
    """The other half of the cost, and the half the byte spy cannot see.

    The spy above counts bytes WRITTEN, so a regression that re-READ the whole
    sidecar on every save -- the 20,377 ms per save this plan removed -- would
    sail straight past it. Nothing else in the suite covers this either: the
    read-freeness tests in test_save_count_without_full_parse.py use sessions of
    five or six messages, which never segment at the default threshold.

    Blind spot, named rather than closed: this spy patches exactly
    `pathlib.Path.read_bytes` and `pathlib.Path.read_text` -- today's only two
    read call-sites inside save() (traced in api/models.py's grow-save path).
    A future regression that reads via `open(self.path).read()`, `os.read()`,
    or `json.load(fp)` would bypass both patches and this assertion would pass
    vacuously. Deliberately not widened to intercept `builtins.open`: this
    same test also WRITES files (the seal + head + index), so a open()-level
    read spy would have to thread write traffic through unfiltered and risks
    counting or breaking its own writes -- a false positive, not a
    correctness gain. If save() ever grows a new read call-site, this test
    needs a matching new patch, not a broader one.
    """
    reads = {"n": 0}
    real = pathlib.Path.read_bytes
    real_text = pathlib.Path.read_text

    def counting_bytes(self, *a, **k):
        if str(self).startswith(str(session_store)) and self.suffix == ".json":
            reads["n"] += 1
        return real(self, *a, **k)

    def counting_text(self, *a, **k):
        if str(self).startswith(str(session_store)) and self.suffix == ".json":
            reads["n"] += 1
        return real_text(self, *a, **k)

    s = M.Session(session_id="rf", title="T", workspace=str(session_store.parent), model="glm",
                  messages=[{"role": "user", "timestamp": float(i), "content": "q" * 200}
                            for i in range(9000)])
    s.save()
    assert s._message_chunks, "fixture must segment, or this test proves nothing"

    monkeypatch.setattr(type(pathlib.Path()), "read_bytes", counting_bytes)
    monkeypatch.setattr(type(pathlib.Path()), "read_text", counting_text)
    s.messages = s.messages + [{"role": "user", "timestamp": 99999.0, "content": "one more"}]
    s.save()

    assert reads["n"] == 0, (
        f"a segmented grow-save read {reads['n']} session file(s); the stat-identity "
        "fast path is meant to make it read-free"
    )
