"""Compression rotation must not save one session's chunks under another's id.

`_preserve_pre_compression_snapshot` archives the pre-compression history under
the OLD session id by retargeting the LIVE session object -- `s.session_id =
old_sid` -- and saving it. Two things about a segmented session broke there:

1. the count it compares against the file was `len(existing['messages'])`, which
   on a segmented head is only the TAIL, so `len(s.messages) > existing_msgs`
   fired for every segmented session;
2. the retargeted object still carried the OTHER session's `_message_chunks`, so
   the head written under `old_sid` named chunk files that only exist under
   `new_sid.msgs` -- measured as an `old_sid` head claiming 30 messages that
   loaded 4, with `chunk_errors: FileNotFoundError`. Lineage history became
   unreadable.

The archived snapshot is the only persistent copy of the uncompressed
conversation (#2223), so a snapshot that cannot be read back is data loss.
"""
import json
from collections import OrderedDict

import pytest

import api.models as M
import api.streaming as streaming


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSION_INDEX_FILE", sdir / "_index.json")
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_MSGS", 10)
    monkeypatch.setattr(M, "_SIDECAR_TAIL_KEEP", 4)
    monkeypatch.setattr(streaming, "SESSION_DIR", sdir)
    return sdir


def _msgs(start, n):
    return [{"role": "user", "timestamp": float(i), "content": f"m{i}"}
            for i in range(start, start + n)]


def _assert_reads_back(sid, total):
    """The head under `sid` must reassemble `total` messages with no chunk errors."""
    loaded = M.Session.load(sid)
    assert loaded is not None, f"{sid} did not load at all"
    assert not loaded._chunk_read_incomplete, \
        f"{sid} has unreadable chunks: {loaded._chunk_read_incomplete}"
    assert len(loaded.messages) == total, \
        f"{sid} loaded {len(loaded.messages)} of {total} messages"
    return loaded


def test_retargeted_snapshot_reads_back_all_of_its_messages(session_store):
    """The measured failure: 30 claimed, 4 loaded, FileNotFoundError.

    A live session segmented under `newsid`, retargeted to `oldsid` and saved.
    The `oldsid` head must not name chunks that live under `newsid.msgs`.
    """
    (session_store / "oldsid.json").write_text(
        json.dumps({"session_id": "oldsid", "messages": _msgs(0, 3)}), encoding="utf-8")

    s = M.Session(session_id="newsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 30))
    s.save()
    assert M._sealed_total(json.loads((session_store / "newsid.json").read_bytes())
                           ["message_chunks"]) == 26, "fixture must be segmented"

    streaming._preserve_pre_compression_snapshot(s, "oldsid")

    head = json.loads((session_store / "oldsid.json").read_bytes())
    assert head["pre_compression_snapshot"] is True
    assert head["message_count"] == 30
    for entry in head.get("message_chunks") or []:
        chunk = session_store / "oldsid.msgs" / entry["file"]
        assert chunk.exists(), \
            f"oldsid head names {entry['file']}, which does not exist under oldsid.msgs"
    _assert_reads_back("oldsid", 30)
    # And nothing was copied or moved out of the other session's chunk dir.
    assert (session_store / "newsid.msgs").is_dir()
    _assert_reads_back("newsid", 30)


def test_snapshot_count_compares_totals_not_the_head_tail(session_store):
    """A disk snapshot that is already complete must not be rewritten from memory.

    With the count taken from the tail, a segmented `oldsid.json` holding 30
    messages read as 4, so `len(s.messages) > existing_msgs` fired even when
    memory held FEWER messages than disk -- and the branch that fires overwrites
    the archived transcript from memory.
    """
    disk = M.Session(session_id="oldsid", title="Archived",
                     workspace=str(session_store.parent), model="glm",
                     messages=_msgs(0, 30))
    disk.save()
    assert len(json.loads((session_store / "oldsid.json").read_bytes())["messages"]) == 4

    # The live object holds only a compressed remnant: fewer messages than disk.
    s = M.Session(session_id="newsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 6))

    streaming._preserve_pre_compression_snapshot(s, "oldsid")

    head = json.loads((session_store / "oldsid.json").read_bytes())
    assert head["pre_compression_snapshot"] is True
    assert head["message_count"] == 30, \
        "a 6-message memory snapshot overwrote a 30-message archived transcript"
    _assert_reads_back("oldsid", 30)


def test_snapshot_count_honours_the_heads_own_claim(session_store):
    """When the manifest cannot be counted, the head's `message_count` still wins.

    A malformed manifest entry is skipped by `_sealed_total`, so the sum
    under-reports what the file claims. Deciding to overwrite the archive from
    memory on the strength of a number that could not be computed is the exact
    failure shape this plan is about, so the head's own claim is taken as the
    floor -- the same stance, and the same shape, as the #1558 guard in
    `Session.save`.

    The head still degrades to what could be reassembled: a save from an
    incomplete read is fail-open by design (Task 4), and the `.bak` it leaves is
    what keeps the full transcript reachable. That `.bak` is also the witness
    that disk was correctly treated as the larger side here.
    """
    disk = M.Session(session_id="oldsid", title="Archived",
                     workspace=str(session_store.parent), model="glm",
                     messages=_msgs(0, 30))
    disk.save()
    head_path = session_store / "oldsid.json"
    head = json.loads(head_path.read_bytes())
    # Break the manifest so _sealed_total cannot count the sealed 26.
    head["message_chunks"] = [{**head["message_chunks"][0], "count": "twenty-six"}]
    head_path.write_text(json.dumps(head), encoding="utf-8")

    s = M.Session(session_id="newsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 20))

    streaming._preserve_pre_compression_snapshot(s, "oldsid")

    after = json.loads(head_path.read_bytes())
    assert after["message_count"] != 20 and len(after["messages"]) != 20, \
        "the head claimed 30; a 20-message memory snapshot must not replace it"
    bak = json.loads((session_store / "oldsid.json.bak").read_bytes())
    assert bak["message_count"] == 30, "the .bak must still hold the claimed total"


def test_retarget_leaves_the_continuation_object_able_to_save_coherently(session_store):
    """After the helper restores `new_sid`, the object must not still hold the
    OLD session's manifest -- the next continuation save would write a `new_sid`
    head naming chunk files under `old_sid.msgs`."""
    (session_store / "oldsid.json").write_text(
        json.dumps({"session_id": "oldsid", "messages": _msgs(0, 3)}), encoding="utf-8")
    s = M.Session(session_id="newsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 30))

    streaming._preserve_pre_compression_snapshot(s, "oldsid")
    assert s.session_id == "newsid"

    s.messages = _msgs(0, 30) + _msgs(30, 2)
    s.save()

    head = json.loads((session_store / "newsid.json").read_bytes())
    for entry in head.get("message_chunks") or []:
        assert (session_store / "newsid.msgs" / entry["file"]).exists(), \
            f"newsid head names {entry['file']}, missing under newsid.msgs"
    _assert_reads_back("newsid", 32)
    _assert_reads_back("oldsid", 30)


def test_archiving_heals_a_head_that_names_a_missing_chunk(session_store):
    """Adopting the target's manifest must not perpetuate a dangling reference.

    Adopting `old_sid`'s own manifest keeps the archive save append-only, which is
    why it is preferred over clearing. But if that head ALREADY names a chunk that
    is gone, adopting carries the hole forward, while clearing would have healed
    the archive outright -- and at this point memory is provably a superset,
    because the branch only fires when `len(s.messages) > existing_msgs`.

    Measured before the fix, archiving 34 in-memory messages over an `oldsid`
    head whose only chunk had been deleted:

        after archive : message_count 34, chunks ['000001.json']
        loads         : 8   chunk_errors ['000001.json: unreadable (FileNotFoundError)']

    26 messages that were in memory at archive time became permanently
    unreachable from the archive -- the very file that exists to be the last copy.
    """
    old = M.Session(session_id="oldsid", title="A", workspace=str(session_store.parent),
                    model="glm", messages=_msgs(0, 30))
    old.save()
    (session_store / "oldsid.msgs" / "000001.json").unlink()
    assert json.loads((session_store / "oldsid.json").read_bytes())["message_chunks"], \
        "the head must still NAME the chunk it can no longer find"

    # Memory is the superset the branch requires: 34 > the head's claimed 30.
    s = M.Session(session_id="newsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 34))

    streaming._preserve_pre_compression_snapshot(s, "oldsid")

    head = json.loads((session_store / "oldsid.json").read_bytes())
    assert head["message_count"] == 34
    for entry in head.get("message_chunks") or []:
        assert (session_store / "oldsid.msgs" / entry["file"]).exists(), \
            f"the archive still names a missing chunk: {entry['file']}"
    _assert_reads_back("oldsid", 34)


def test_a_present_manifest_is_still_adopted_rather_than_re_sealed(session_store):
    """The guard against over-correcting: a healthy head keeps the cheap path.

    If every named chunk is present the manifest is adopted, so the archive save
    APPENDS. Re-sealing instead would be correct but would pay the full-history
    write price this design exists to avoid, once per compression.
    """
    old = M.Session(session_id="oldsid", title="A", workspace=str(session_store.parent),
                    model="glm", messages=_msgs(0, 30))
    old.save()
    sealed_before = json.loads((session_store / "oldsid.json").read_bytes())["message_chunks"]
    assert [e["file"] for e in sealed_before] == ["000001.json"]
    bytes_before = (session_store / "oldsid.msgs" / "000001.json").read_bytes()

    s = M.Session(session_id="newsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 40))

    streaming._preserve_pre_compression_snapshot(s, "oldsid")

    after = json.loads((session_store / "oldsid.json").read_bytes())["message_chunks"]
    assert after[0] == sealed_before[0], "the existing sealed chunk was re-sealed, not kept"
    assert [e["file"] for e in after] == ["000001.json", "000002.json"], \
        "the archive save must APPEND, not restart the numbering"
    assert (session_store / "oldsid.msgs" / "000001.json").read_bytes() == bytes_before, \
        "an adopted chunk is write-once; its bytes must not be rewritten"
    _assert_reads_back("oldsid", 40)


def test_retarget_refuses_a_manifest_naming_a_traversal_path(session_store):
    """A head is read off disk, so `file` can be anything. Refuse, do not stat it."""
    s = M.Session(session_id="oldsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 30))
    s.save()
    good = json.loads((session_store / "oldsid.json").read_bytes())["message_chunks"]

    streaming._retarget_session_sidecar(
        s, "oldsid", manifest=[{**good[0], "file": "../../etc/passwd"}])

    assert s._message_chunks == [], "a traversal filename must never be adopted"


def test_retarget_helper_drops_every_attribute_that_describes_the_old_file(session_store):
    """`_message_chunks`, `_disk_identity_seen` and `_disk_msg_count` all describe
    ONE file. A retarget must leave none of them behind."""
    s = M.Session(session_id="oldsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 30))
    s.save()
    assert s._message_chunks and s._disk_identity_seen is not None and s._disk_msg_count == 30

    streaming._retarget_session_sidecar(s, "newsid")

    assert s.session_id == "newsid"
    assert s._message_chunks == []
    assert s._disk_identity_seen is None
    assert s._disk_msg_count is None
    # A manifest read out of the TARGET's own head is adopted, not discarded --
    # that is what keeps the following save append-only instead of re-sealing.
    head = json.loads((session_store / "oldsid.json").read_bytes())
    streaming._retarget_session_sidecar(s, "oldsid", manifest=head["message_chunks"])
    assert s._message_chunks == head["message_chunks"]
    # ...but only if it is well-formed. A malformed one is not authority.
    streaming._retarget_session_sidecar(s, "newsid", manifest=[{"count": "nope"}])
    assert s._message_chunks == []


def test_rotation_retarget_survives_a_missing_old_head(session_store):
    """The rotation site needs the retarget in its own right.

    When `old_sid.json` does not exist, `_preserve_pre_compression_snapshot`
    returns immediately without touching the object, so the assignment at the
    rotation site is the only thing standing between the continuation save and a
    `new_sid` head naming chunks under `old_sid.msgs`.
    """
    s = M.Session(session_id="oldsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 30))
    s.save()
    (session_store / "oldsid.json").unlink()  # nothing for the helper to preserve

    # Exactly what the compression-rotation path does, in order.
    streaming._retarget_session_sidecar(s, "newsid")
    streaming._preserve_pre_compression_snapshot(s, "oldsid")
    s.save()

    head = json.loads((session_store / "newsid.json").read_bytes())
    for entry in head.get("message_chunks") or []:
        assert (session_store / "newsid.msgs" / entry["file"]).exists(), \
            f"newsid head names {entry['file']}, missing under newsid.msgs"
    _assert_reads_back("newsid", 30)


def test_rotation_helper_does_not_move_or_delete_chunk_files(session_store):
    """Chunks are write-once. A retarget re-seals; it never relocates bytes."""
    s = M.Session(session_id="newsid", title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, 30))
    s.save()
    before = {p.name: p.read_bytes() for p in (session_store / "newsid.msgs").iterdir()}
    (session_store / "oldsid.json").write_text(
        json.dumps({"session_id": "oldsid", "messages": _msgs(0, 3)}), encoding="utf-8")

    streaming._preserve_pre_compression_snapshot(s, "oldsid")

    after = {p.name: p.read_bytes() for p in (session_store / "newsid.msgs").iterdir()}
    assert after == before, "the other session's sealed chunks were touched"
