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

    total_size = len(json.dumps(msgs))
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
    # at ~702 KB -- about 4.5x above the sealed write and 5x below the full
    # rewrite.
    avg_msg_bytes = total_size / len(msgs)
    cap = avg_msg_bytes * (M._SIDECAR_TAIL_KEEP + 1) * 5 + 32 * 1024
    assert written["n"] < cap, (
        f"a post-seal save wrote {written['n']:,} bytes (cap {cap:,.0f}) for a "
        f"{total_size:,}-byte history; sealing is not bounding the cost"
    )
    assert len(M.Session.load("cost").messages) == 12001, "and it is still lossless"


def test_repeated_saves_stay_bounded(session_store, monkeypatch):
    s = M.Session(session_id="cost2", title="T", workspace=str(session_store.parent), model="glm",
                  messages=[{"role": "user", "timestamp": float(i), "content": "y" * 300} for i in range(6000)])
    s.save()
    written = _spy_bytes_written(monkeypatch, session_store)
    for i in range(20):
        s.messages = s.messages + [{"role": "user", "timestamp": 10000.0 + i, "content": "z"}]
        s.save()
    assert written["n"] < 40 * 1024 * 1024, (
        f"20 saves wrote {written['n']:,} bytes; that is a full rewrite per save"
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
