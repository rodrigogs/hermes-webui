"""Recovery sees TOTALS on a segmented head, and keeps its -1 contract.

_msg_count drives inspect_session_recovery_status, which decides whether a .bak
wins over the live file. If it counted only the tail, every segmented session
would look like it had lost its history and recovery would restore a .bak over
a perfectly good file.
"""
import json
from collections import OrderedDict

import pytest

import api.models as M
import api.session_recovery as R


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSION_INDEX_FILE", sdir / "_index.json")
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    monkeypatch.setattr(M, "_SIDECAR_TAIL_MAX_MSGS", 10)
    monkeypatch.setattr(M, "_SIDECAR_TAIL_KEEP", 4)
    return sdir


def _msgs(start, n):
    return [{"role": "user", "timestamp": float(i), "content": f"m{i}"}
            for i in range(start, start + n)]


def _segmented(session_store, sid, n=22):
    s = M.Session(session_id=sid, title="T", workspace=str(session_store.parent),
                  model="glm", messages=_msgs(0, n))
    s.save()
    p = session_store / f"{sid}.json"
    assert json.loads(p.read_bytes())["message_chunks"], "fixture must be segmented"
    return p


def test_msg_count_returns_the_total_for_a_segmented_head(session_store):
    p = _segmented(session_store, "r1", n=22)
    head = json.loads(p.read_bytes())
    assert len(head["messages"]) == 4, "the head holds only the tail"
    assert R._msg_count(p) == 22, "_msg_count must report the TOTAL"


def test_msg_count_is_unchanged_for_an_unsegmented_head(session_store):
    doc = {"session_id": "r2", "messages": _msgs(0, 7)}
    p = session_store / "r2.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert R._msg_count(p) == 7


def test_msg_count_keeps_the_minus_one_contract(session_store):
    p = _segmented(session_store, "r3", n=22)
    torn = session_store / "torn.json"
    torn.write_bytes(p.read_bytes()[:-30])
    assert R._msg_count(torn) == -1

    index = session_store / "_index.json"
    index.write_text(json.dumps([{"session_id": "x"}]), encoding="utf-8")
    assert R._msg_count(index) == -1

    assert R._msg_count(session_store / "nope.json") == -1


def test_msg_count_ignores_a_malformed_manifest_rather_than_crashing(session_store):
    """A hand-edited or corrupt manifest must degrade to the tail, not raise.

    Erring low is the safe side for a manifest that cannot be counted: it keeps a
    .bak eligible to win, and the .bak comparison is the only thing _msg_count
    feeds. The exact-equality assertions matter as much as the no-crash: an
    earlier version of this test asserted only `>= tail`, which left the
    DIRECTION unpinned and let the over-count below through review.
    """
    for bad in (None, "nope", 5, [], [{"count": "seven"}], [{"count": -3}], ["x"],
                [{"count": 4}, "x"]):
        doc = {"session_id": "rb", "messages": _msgs(0, 3), "message_chunks": bad}
        p = session_store / "rb.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        assert R._msg_count(p) == 3, f"manifest {bad!r} must count as tail-only"


def _break_continuity_after_first_entry(path):
    """Append a manifest entry whose `first_idx` does not chain, and lie big."""
    head = json.loads(path.read_bytes())
    head["message_chunks"] = head["message_chunks"] + [
        {"seq": 2, "file": "000002.json", "count": 900, "first_idx": 999,
         "sha256": "0" * 64, "first_key": None, "last_key": None}]
    path.write_text(json.dumps(head), encoding="utf-8")
    return head


def test_msg_count_counts_only_the_manifest_prefix_load_can_reach(session_store):
    """The count must match what `Session.load` can actually reassemble.

    `_normalised_manifest` truncates a manifest at a continuity break, and
    `_read_sidecar_document` reads only that surviving prefix. Summing the RAW
    manifest counts past the break -- past entries whose messages no reader will
    ever return -- and a file that over-states itself SUPPRESSES ITS OWN
    RECOVERY. Measured before this fix, on a 30-message file carrying one bogus
    `count: 900` entry at a broken `first_idx`:

        _msg_count            : 930
        Session.load reaches  : 30

    Erring high here is the exact opposite of the err-low stance the rest of this
    function takes, and it is the dangerous direction: the number only ever gets
    compared against a .bak.
    """
    p = _segmented(session_store, "r9", n=30)
    raw = _break_continuity_after_first_entry(p)["message_chunks"]

    assert M._sealed_total(raw) == 926, "the raw manifest claims the bogus 900"
    assert M._sealed_total(M._normalised_manifest(raw)) == 26, "only 26 chain"
    assert R._msg_count(p) == 30, "prefix (26) + tail (4), not the raw 930"
    assert len(M.Session.load("r9").messages) == 30, "and that is what load reaches"


def test_a_manifest_break_at_position_zero_counts_the_tail_alone(session_store):
    """A break in the FIRST entry leaves no reachable sealed prefix at all."""
    p = _segmented(session_store, "r10", n=30)
    head = json.loads(p.read_bytes())
    head["message_chunks"] = [{**head["message_chunks"][0], "first_idx": 7}]
    p.write_text(json.dumps(head), encoding="utf-8")

    assert M._normalised_manifest(head["message_chunks"]) == []
    assert R._msg_count(p) == 4, "nothing chains, so only the tail is reachable"
    assert len(M.Session.load("r10").messages) == 4


def test_an_over_stated_manifest_does_not_suppress_a_legitimate_restore(session_store):
    """The consequence, end to end: the damaged file must not beat its own .bak.

    Pre-fix this returned `recommend: no_action` with live_messages 930 against a
    35-message backup -- the damaged file won, and the sweep left it in place.
    """
    p = _segmented(session_store, "r11", n=30)
    _break_continuity_after_first_entry(p)
    bak = {"session_id": "r11", "messages": _msgs(0, 35), "message_count": 35}
    p.with_suffix(".json.bak").write_text(json.dumps(bak), encoding="utf-8")

    status = R.inspect_session_recovery_status(p)

    assert status["live_messages"] == 30, "the reachable total, not the claimed 930"
    assert status["bak_messages"] == 35
    assert status["recommend"] == "restore"

    report = R.recover_all_sessions_on_startup(session_store)

    assert report["restored"] == 1, f"the good .bak must win: {report['details']}"
    assert len(json.loads(p.read_bytes())["messages"]) == 35


def test_msg_count_does_not_read_chunk_files(session_store, monkeypatch):
    """The counts are in the manifest, which is in the head. Reading chunks here
    would put the cost this whole design removes back into every boot."""
    p = _segmented(session_store, "r4", n=22)
    opened = []
    real = type(p).read_bytes

    def spy(self, *a, **k):
        opened.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(type(p), "read_bytes", spy)
    assert R._msg_count(p) == 22
    assert not any(".msgs" in o for o in opened), f"read a chunk: {opened}"


def test_recovery_status_compares_totals_not_tails(session_store):
    p = _segmented(session_store, "r5", n=22)
    # A .bak with FEWER total messages must not win.
    bak_doc = json.loads(p.read_bytes())
    bak_doc["message_chunks"] = []
    bak_doc["messages"] = _msgs(0, 5)
    bak_doc["message_count"] = 5
    p.with_suffix(".json.bak").write_text(json.dumps(bak_doc), encoding="utf-8")

    status = R.inspect_session_recovery_status(p)

    assert status["live_messages"] == 22 and status["bak_messages"] == 5
    assert status["recommend"] != "restore", "a smaller .bak must not overwrite 22 messages"


def test_startup_sweep_does_not_restore_a_smaller_bak_over_a_segmented_session(session_store):
    """The INVERSION, not merely the undercount.

    A healthy segmented live file (26 sealed + a 4-message tail = 30) next to a
    stale, unsegmented 10-message .bak. Counting the head's `messages` alone
    reads 4 < 10, so the boot-time recovery sweep restores the SMALLER .bak over
    the healthy session: 30 messages replaced by 10, automatically, with no user
    action. That is the whole reason this count must be a total.
    """
    p = _segmented(session_store, "r6", n=30)
    head = json.loads(p.read_bytes())
    assert M._sealed_total(head["message_chunks"]) == 26
    assert len(head["messages"]) == 4

    stale_bak = {"session_id": "r6", "messages": _msgs(0, 10), "message_count": 10}
    p.with_suffix(".json.bak").write_text(json.dumps(stale_bak), encoding="utf-8")

    report = R.recover_all_sessions_on_startup(session_store)

    assert report["restored"] == 0, f"boot restored a smaller .bak: {report['details']}"
    # The live file is untouched and still reassembles all 30 messages.
    assert json.loads(p.read_bytes())["message_chunks"] == head["message_chunks"]
    reloaded = M.Session.load("r6")
    assert len(reloaded.messages) == 30
    assert not reloaded._chunk_read_incomplete


def test_startup_sweep_still_restores_a_genuinely_bigger_bak(session_store):
    """The -1/restore machinery must keep working for the case it exists for."""
    p = _segmented(session_store, "r7", n=22)
    bak = {"session_id": "r7", "messages": _msgs(0, 40), "message_count": 40}
    p.with_suffix(".json.bak").write_text(json.dumps(bak), encoding="utf-8")

    report = R.recover_all_sessions_on_startup(session_store)

    assert report["restored"] == 1
    assert len(json.loads(p.read_bytes())["messages"]) == 40


def test_startup_sweep_restores_over_a_torn_segmented_live_file(session_store):
    """A torn live file must still read as -1 so its .bak can win.

    This is the contract the full parse in _msg_count exists for, and counting
    the manifest must not weaken it: the manifest lives in the head, and a head
    that will not parse yields no manifest either.
    """
    p = _segmented(session_store, "r8", n=22)
    bak = {"session_id": "r8", "messages": _msgs(0, 6), "message_count": 6}
    p.with_suffix(".json.bak").write_text(json.dumps(bak), encoding="utf-8")
    p.write_bytes(p.read_bytes()[:-30])
    assert R._msg_count(p) == -1

    report = R.recover_all_sessions_on_startup(session_store)

    assert report["restored"] == 1
    assert len(json.loads(p.read_bytes())["messages"]) == 6
