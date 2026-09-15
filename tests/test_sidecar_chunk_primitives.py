"""Primitives for segmented sidecars: paths, manifest shape, structural keys.

These are pure functions on purpose. The write and read paths (Tasks 2-3) are
where the risk lives; keeping the naming, validation and continuity rules
separately testable means a failure there is never ambiguous about which layer
produced it.
"""
import hashlib
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


def test_chunk_dir_name_cannot_collide_with_a_session_id(session_store):
    d = M._session_chunk_dir("abc123")
    assert d == session_store / "abc123.msgs"
    # A dot is not in _SAFE_SID_CHARS, so no session id can ever produce this name.
    assert not all(c in M._SAFE_SID_CHARS for c in d.name)


def test_chunk_dir_refuses_an_unsafe_sid(session_store):
    for bad in ("../escape", "a/b", "", None, "dot.dot"):
        assert M._session_chunk_dir(bad) is None, f"{bad!r} must not yield a path"


def test_chunk_path_is_zero_padded_and_ordered_lexicographically(session_store):
    assert M._chunk_path("s1", 1) == session_store / "s1.msgs" / "000001.json"
    assert M._chunk_path("s1", 42) == session_store / "s1.msgs" / "000042.json"
    names = [M._chunk_path("s1", n).name for n in (2, 10, 1)]
    assert sorted(names) == [M._chunk_path("s1", n).name for n in (1, 2, 10)], \
        "lexicographic order must equal numeric order, or glob() ordering breaks"


def test_sha256_hex_matches_hashlib():
    assert M._sha256_hex(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_structural_key_uses_role_timestamp_and_content_length():
    # `timestamp` is the spelling every message producer in this repo writes, so
    # it is the one that must be read; `ts` is only a fallback for fixtures.
    assert M._structural_key(
        {"role": "assistant", "timestamp": 1234.5, "content": "abcde"}
    ) == ("assistant", 1234.5, 5)
    assert M._structural_key({"role": "assistant", "ts": 1234.5, "content": "abcde"}) == (
        "assistant", 1234.5, 5)
    assert M._structural_key(
        {"role": "user", "timestamp": 9.0, "ts": 1.0, "content": "ab"}
    ) == ("user", 9.0, 2), "timestamp wins when a message somehow carries both"
    # Tolerates absent fields rather than raising: a malformed message must not
    # crash a save, it must only fail to match.
    assert M._structural_key({}) == (None, None, 0)
    assert M._structural_key("not a dict") == (None, None, 0)


def test_sealed_total_sums_counts_and_tolerates_garbage():
    assert M._sealed_total([{"count": 3}, {"count": 4}]) == 7
    assert M._sealed_total([]) == 0
    assert M._sealed_total(None) == 0
    assert M._sealed_total("nonsense") == 0


def test_normalised_manifest_drops_malformed_entries():
    good = {"seq": 1, "file": "000001.json", "count": 2, "first_idx": 0, "sha256": "ab"}
    raw = [good, {"seq": 2}, "junk", None]
    assert M._normalised_manifest(raw) == [good]


def test_normalised_manifest_drops_entries_that_break_continuity():
    """first_idx must equal the running total. An entry that does not is either
    a bug or a partially-rewritten manifest; either way the safe reading is
    'the sealed prefix ends here', not 'skip a hole and keep going'."""
    a = {"seq": 1, "file": "000001.json", "count": 2, "first_idx": 0, "sha256": "x"}
    b = {"seq": 2, "file": "000002.json", "count": 2, "first_idx": 2, "sha256": "y"}
    bad = {"seq": 3, "file": "000003.json", "count": 2, "first_idx": 99, "sha256": "z"}
    c = {"seq": 4, "file": "000004.json", "count": 1, "first_idx": 4, "sha256": "w"}
    assert M._normalised_manifest([a, b, bad, c]) == [a, b]


def test_tail_thresholds_have_the_documented_defaults():
    assert M._SIDECAR_TAIL_MAX_MSGS == 2000
    assert M._SIDECAR_TAIL_MAX_BYTES == 2 * 1024 * 1024
    assert M._SIDECAR_TAIL_KEEP == 500
