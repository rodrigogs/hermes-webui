"""Rollback and inspection. A storage format without a way back is a trap."""
import json
import pathlib
import shutil
from collections import OrderedDict

import pytest

import api.models as M
from api.sidecar_maintenance import fsck_sessions, unchunk_all, unchunk_session


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_MSGS", 10)
    monkeypatch.setattr(M, "_SIDECAR_TAIL_KEEP", 4)
    return sdir


def _msgs(n):
    return [{"role": "user", "ts": float(i), "content": f"m{i}"} for i in range(n)]


def _segmented(session_store, sid, n=22):
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(n))
    s.save()
    return session_store / f"{sid}.json"


def _two_chunk(session_store, sid):
    """Two sealed chunks, so a test can remove one and still leave one to read."""
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(22))
    s.save()
    s.messages = _msgs(33)  # same prefix, so the sealed chunk is appended to, not re-sealed
    s.save()
    doc = json.loads((session_store / f"{sid}.json").read_bytes())
    assert [e["file"] for e in doc["message_chunks"]] == ["000001.json", "000002.json"]
    return session_store / f"{sid}.json"


def test_unchunk_folds_everything_back_into_one_file(session_store):
    p = _segmented(session_store, "m1", n=22)
    out = unchunk_session("m1")
    assert out["unchunked"] is True and out["messages"] == 22
    doc = json.loads(p.read_bytes())
    assert not doc.get("message_chunks")
    assert [m["content"] for m in doc["messages"]] == [f"m{i}" for i in range(22)]
    assert not (session_store / "m1.msgs").exists()
    assert [m["content"] for m in M.Session.load("m1").messages] == [f"m{i}" for i in range(22)]


def test_unchunk_is_idempotent(session_store):
    _segmented(session_store, "m2", n=22)
    unchunk_session("m2")
    out = unchunk_session("m2")
    assert out["unchunked"] is False and out["messages"] == 22


def test_unchunk_refuses_when_a_chunk_is_missing(session_store):
    p = _segmented(session_store, "m3", n=22)
    next(iter((session_store / "m3.msgs").glob("*.json"))).unlink()
    out = unchunk_session("m3")
    assert out["unchunked"] is False and "error" in out
    assert json.loads(p.read_bytes()).get("message_chunks"), "the manifest must survive a refusal"
    assert (session_store / "m3.msgs").exists(), "nothing removed on a refusal"


def test_fsck_reports_a_healthy_segmented_session(session_store):
    _segmented(session_store, "m4", n=22)
    rep = fsck_sessions(session_store)
    row = next(r for r in rep["sessions"] if r["session_id"] == "m4")
    assert row["chunks"] == 1 and row["total_messages"] == 22 and row["errors"] == []
    assert rep["orphans"] == []


def test_fsck_lists_orphan_chunks_without_touching_them(session_store):
    _segmented(session_store, "m5", n=22)
    orphan = session_store / "m5.msgs" / "009999.json"
    orphan.write_text('{"session_id": "m5", "seq": 9999, "first_idx": 0, "count": 0, "messages": []}',
                      encoding="utf-8")
    rep = fsck_sessions(session_store)
    assert any(o["file"].endswith("009999.json") for o in rep["orphans"])
    assert orphan.exists(), "fsck must never delete"


def test_fsck_reports_a_corrupted_chunk(session_store):
    """Only under `verify=True`, and the default must not pretend otherwise.

    Hashing every chunk of every session is the expensive half of fsck, so it
    is opt-in; the default walk stats chunk files and can therefore see a
    MISSING one but not an ALTERED one. Both halves are asserted here because
    a default that silently reported nothing would look identical to a clean
    store.

    Fails if the `verify=True` branch stops hashing (the mismatch goes
    unreported), or if hashing creeps back into the default walk (the first
    assertion trips, and with it the memory bound this flag exists for).
    """
    _segmented(session_store, "m6", n=22)
    f = next(iter((session_store / "m6.msgs").glob("*.json")))
    f.write_bytes(f.read_bytes().replace(b"m0", b"XX"))

    default_row = next(r for r in fsck_sessions(session_store)["sessions"] if r["session_id"] == "m6")
    assert default_row["errors"] == [], "the default walk stats chunks; it must not hash them"

    row = next(r for r in fsck_sessions(session_store, verify=True)["sessions"] if r["session_id"] == "m6")
    assert any("sha256" in e and "000001.json" in e for e in row["errors"]), row["errors"]
    assert f.exists(), "fsck must never delete"


def test_fsck_default_reads_no_chunk_bytes_and_still_finds_a_missing_one(session_store, monkeypatch):
    """The default walk must be bounded: heads and os.stat, nothing else.

    `_read_sidecar_document` per head read, sha256-ed and json.loads-ed every
    chunk of every session -- on the deployed box, whose archived head alone is
    203,876,949 bytes, that is the allocation that got the webui OOM-killed
    twice. Proven by making any read of a chunk file fail with the error that
    actually happens there (MemoryError, on a 5.9 GiB VM with no swap):
    reporting the missing chunk must not depend on opening the others.

    MemoryError rather than OSError deliberately: the reassembling reader
    CATCHES OSError per chunk and would sail through this test while doing
    exactly the thing it forbids. Fails if any chunk read returns to the
    default walk -- the exception escapes fsck and this test errors out.
    """
    _two_chunk(session_store, "m9")
    (session_store / "m9.msgs" / "000002.json").unlink()
    real_read_bytes = pathlib.Path.read_bytes

    def no_chunk_reads(self, *a, **k):
        if ".msgs" in str(self):
            raise MemoryError(f"refusing to read chunk bytes: {self}")
        return real_read_bytes(self, *a, **k)

    monkeypatch.setattr(type(pathlib.Path()), "read_bytes", no_chunk_reads)

    rep = fsck_sessions(session_store)

    row = next(r for r in rep["sessions"] if r["session_id"] == "m9")
    assert row["chunks"] == 2
    assert any("000002.json" in e and "missing" in e for e in row["errors"]), row["errors"]
    assert not any("000001.json" in e for e in row["errors"]), "the present chunk is fine"


def test_fsck_calls_an_archived_sessions_chunks_archived_not_orphaned(session_store):
    """The operator archives by renaming the HEAD, and one such file is on the
    deployed box right now (`fa3bca34a0c6.json.archived`, 203,876,949 bytes).
    Its chunks hold that session's entire history, so calling them orphans
    invites exactly one action -- deletion -- and loses it.

    Fails if the `.msgs` walk goes back to deciding "orphan" from the
    `*.json` glob alone: with no live head to reference them, every chunk of
    an archived session lands in `orphans`.
    """
    _segmented(session_store, "ma", n=22)
    (session_store / "ma.json").rename(session_store / "ma.json.archived")

    rep = fsck_sessions(session_store)

    assert rep["orphans"] == [], "an archived session's history is not orphaned"
    row = next(r for r in rep["archived_chunk_dirs"] if r["session_id"] == "ma")
    assert row["files"] == 1 and row["bytes"] > 0
    assert not any(r["session_id"] == "ma" for r in rep["headless_chunk_dirs"])
    assert (session_store / "ma.msgs" / "000001.json").exists(), "fsck must never delete"


def test_fsck_reports_a_headless_chunk_dir_under_its_own_key(session_store):
    """No head and no sibling at all: this really is residue, but it is
    reported as a DIRECTORY (one row, from os.stat) rather than as N orphan
    files, so the operator sees one decision instead of thousands of lines.

    Fails if headless directories are folded back into `orphans`, which is the
    key the report shares with re-seal residue under a live head -- two very
    different situations that must not arrive looking the same.
    """
    _segmented(session_store, "mb", n=22)
    (session_store / "mb.json").unlink()
    (session_store / "mb.json.bak").unlink(missing_ok=True)

    rep = fsck_sessions(session_store)

    assert rep["orphans"] == []
    row = next(r for r in rep["headless_chunk_dirs"] if r["session_id"] == "mb")
    assert row["files"] == 1 and row["bytes"] > 0
    assert not any(r["session_id"] == "mb" for r in rep["archived_chunk_dirs"])


def test_unchunk_leaves_a_bak_only_chunk_alone_after_a_reseal(session_store):
    """Differing-manifest case: a shrink big enough to eat into the sealed
    prefix forces a re-seal, so the live head ends up naming a NEW chunk
    while the `.bak` this same save just wrote still names the OLD one.
    unchunk_session must not delete a chunk its own `.bak` still needs, even
    though the live manifest no longer names it.
    """
    p = _segmented(session_store, "m7", n=22)
    s = M.Session.load("m7")
    assert len(s.messages) == 22
    s.messages = _msgs(6)  # shrinks below the sealed 18 -> forces a re-seal
    s.save()
    bak = session_store / "m7.json.bak"
    assert bak.exists()
    bak_manifest = json.loads(bak.read_bytes())["message_chunks"]
    bak_files = {e["file"] for e in bak_manifest}
    assert bak_files, "the .bak must claim at least one sealed chunk for this test to mean anything"

    out = unchunk_session("m7")
    assert out["unchunked"] is True and out["messages"] == 6

    chunk_dir = session_store / "m7.msgs"
    for fname in bak_files:
        assert (chunk_dir / fname).exists(), f"{fname} is needed by m7.json.bak"

    # Restoring the .bak must still recover the full pre-shrink history.
    shutil.copyfile(bak, p)
    restored = M.Session.load("m7")
    assert not restored._chunk_read_incomplete, restored._chunk_read_incomplete
    assert len(restored.messages) == 22


def test_unchunk_leaves_a_chunk_alone_when_the_bak_shares_it(session_store):
    """Shared-chunk case: a tail-only shrink does NOT force a re-seal (the
    sealed prefix is untouched), so the live head and its `.bak` can name
    the exact SAME sealed chunk file. unchunk_session must not delete a
    chunk its own `.bak` still needs, even though the LIVE manifest
    (correctly, for the live head's own purposes) names it too.
    """
    p = _segmented(session_store, "m8", n=30)
    s = M.Session.load("m8")
    assert len(s.messages) == 30
    s.messages = s.messages[:-2]  # tail-only shrink: 30 -> 28, no re-seal
    s.save()
    bak = session_store / "m8.json.bak"
    assert bak.exists()
    bak_manifest = json.loads(bak.read_bytes())["message_chunks"]
    bak_files = {e["file"] for e in bak_manifest}
    live_manifest = json.loads(p.read_bytes())["message_chunks"]
    live_files = {e["file"] for e in live_manifest}
    assert bak_files == live_files, "this test only means something when live and .bak share a chunk"

    out = unchunk_session("m8")
    assert out["unchunked"] is True and out["messages"] == 28

    chunk_dir = session_store / "m8.msgs"
    for fname in bak_files:
        assert (chunk_dir / fname).exists(), f"{fname} is needed by m8.json.bak"

    # Restoring the .bak must still recover the full pre-shrink history.
    shutil.copyfile(bak, p)
    restored = M.Session.load("m8")
    assert not restored._chunk_read_incomplete, restored._chunk_read_incomplete
    assert len(restored.messages) == 30


def test_unchunk_all_folds_every_segmented_session_and_skips_an_archived_one(session_store):
    """The rollback path for the whole store, one session at a time.

    Fails if the work list comes from anywhere but the `.msgs` directories --
    a `*.json` walk cannot see an archived session at all, and parsing every
    head to find the work costs exactly what the rollback is trying to avoid
    -- or if a session whose head is missing gets folded anyway: there is
    nothing to fold into, and an archived session must be neither resurrected
    nor stripped of the chunks that hold its history.
    """
    for sid in ("n1", "n2", "n3"):
        _segmented(session_store, sid, n=22)
    _segmented(session_store, "n4", n=22)
    (session_store / "n4.json").rename(session_store / "n4.json.archived")
    archived_chunks = sorted(p.name for p in (session_store / "n4.msgs").glob("*.json"))
    assert archived_chunks, "the archived session must still have chunks, or this proves nothing"

    out = unchunk_all(session_store)

    assert sorted(out["unchunked"]) == ["n1", "n2", "n3"]
    for sid in ("n1", "n2", "n3"):
        doc = json.loads((session_store / f"{sid}.json").read_bytes())
        assert not doc.get("message_chunks")
        assert [m["content"] for m in doc["messages"]] == [f"m{i}" for i in range(22)]
        assert not (session_store / f"{sid}.msgs").exists()
    assert [r["session_id"] for r in out["skipped"]] == ["n4"]
    assert "head" in out["skipped"][0]["reason"], out["skipped"]
    assert sorted(p.name for p in (session_store / "n4.msgs").glob("*.json")) == archived_chunks


def test_unchunk_all_keeps_going_after_a_session_it_cannot_fold(session_store):
    """One unfoldable session must not strand every session after it.

    The torn head sorts FIRST on purpose: fails if a per-session failure
    propagates out of the loop or returns early, because then n6 is still
    segmented and the store is half rolled back with no report saying so.
    """
    _segmented(session_store, "n5", n=22)
    _segmented(session_store, "n6", n=22)
    (session_store / "n5.json").write_text('{"session_id": "n5", "messages": [', encoding="utf-8")

    out = unchunk_all(session_store)

    assert out["unchunked"] == ["n6"]
    assert [r["session_id"] for r in out["skipped"]] == ["n5"]
    assert not json.loads((session_store / "n6.json").read_bytes()).get("message_chunks")
    assert (session_store / "n5.msgs").exists(), "a refusal must leave the chunks alone"
