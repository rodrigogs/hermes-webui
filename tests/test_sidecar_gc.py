"""C3a: reclaiming orphaned chunks, operator-invoked, webui stopped (spec 2026-09-17 §4.1)."""
import json
import os
import time
from collections import OrderedDict

import pytest

import api.models as M
from api import sidecar_maintenance as SM

import json
from collections import OrderedDict
from pathlib import Path

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


def _age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


def _orphaned(session_store, sid):
    """The production shape: live head names only the highest seq; two lower files are orphans."""
    s = _segmented(session_store, sid, n=30)
    d = session_store / f"{sid}.msgs"
    live = _chunks(session_store, sid)[0]
    for name in ("000002.json", "000003.json"):
        (d / name).write_bytes(b'{"session_id":"%s","seq":2,"first_idx":0,"count":1,"messages":[]}' % sid.encode())
    for p in list(d.glob("*.json")) + [session_store / f"{sid}.json"]:
        _age(p, 3600)
    return s, live


def test_dry_run_reports_and_touches_nothing(session_store):
    """Fails if the default (apply=False) unlinks anything."""
    _, live = _orphaned(session_store, "g1")
    rep = SM.gc_sessions(session_store)
    row = rep["sessions"][0]
    assert sorted(row["candidates"]) == ["000002.json", "000003.json"] and row["reclaimed"] == []
    assert rep["would_reclaim"] == 2 and rep["reclaimed"] == 0
    assert sorted(_chunks(session_store, "g1")) == sorted([live, "000002.json", "000003.json"])


def test_apply_requires_the_webui_stopped_assertion(session_store):
    """Fails if apply=True runs without webui_stopped=True."""
    _orphaned(session_store, "g2")
    with pytest.raises(ValueError):
        SM.gc_sessions(session_store, apply=True)


def test_apply_reclaims_the_orphans_writes_the_mark_first_and_keeps_the_live_chunk(session_store, monkeypatch):
    """Fails if gc unlinks before raising .seq_hwm, or removes a chunk the live head names."""
    _, live = _orphaned(session_store, "g3")
    order = []
    real_write, real_unlink = M._write_seq_hwm, os.unlink
    monkeypatch.setattr(M, "_write_seq_hwm", lambda sid, v: order.append(("hwm", v)) or real_write(sid, v))
    monkeypatch.setattr(SM.Path, "unlink", lambda self, *a, **k: order.append(("unlink", self.name)) or real_unlink(self))
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    assert order[0] == ("hwm", 3), f"the mark is raised to the highest seq BEFORE the first unlink: {order}"
    assert rep["reclaimed"] == 2 and _chunks(session_store, "g3") == [live]
    assert M._read_seq_hwm("g3") == 3 and M._next_chunk_seq("g3") == 4


def test_grace_skips_young_chunks_and_busy_heads(session_store):
    """Fails if min_age_s is not applied to BOTH the chunk and the head mtime."""
    s, live = _orphaned(session_store, "g4")
    _age(session_store / "g4.msgs" / "000002.json", 10)          # young orphan
    rep = SM.gc_sessions(session_store, min_age_s=900)
    assert rep["sessions"][0]["candidates"] == ["000003.json"]
    _age(session_store / "g4.json", 10)                          # busy head
    rep = SM.gc_sessions(session_store, min_age_s=900)
    assert rep["sessions"] == [] and "g4" in rep["skipped_busy"]


def test_a_bak_named_chunk_is_never_a_candidate(session_store):
    """Fails if gc consults only the live head (I4)."""
    _, live = _orphaned(session_store, "g5")
    bak = json.loads((session_store / "g5.json").read_bytes())
    bak["message_chunks"] = [dict(bak["message_chunks"][0], file="000002.json")]
    (session_store / "g5.json.bak").write_text(json.dumps(bak), encoding="utf-8")
    rep = SM.gc_sessions(session_store)
    assert rep["sessions"][0]["candidates"] == ["000003.json"]


def test_archived_headless_and_unmanifested_dirs_are_skipped(session_store):
    """Fails if gc treats an archived, headless or unmanifested directory's files as candidates."""
    _orphaned(session_store, "g6"); (session_store / "g6.json").rename(session_store / "g6.json.archived")
    _orphaned(session_store, "g7"); (session_store / "g7.json").unlink()
    _orphaned(session_store, "g8")
    head = json.loads((session_store / "g8.json").read_bytes()); head.pop("message_chunks"); head["messages"] = [_msg(i) for i in range(30)]
    (session_store / "g8.json").write_text(json.dumps(head), encoding="utf-8"); _age(session_store / "g8.json", 3600)
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    assert rep["reclaimed"] == 0
    assert [r["session_id"] for r in rep["archived_chunk_dirs"]] == ["g6"]
    assert [r["session_id"] for r in rep["headless_chunk_dirs"]] == ["g7"]
    assert [r["session_id"] for r in rep["unmanifested_chunk_dirs"]] == ["g8"]
    assert len(_chunks(session_store, "g6")) == 3 and len(_chunks(session_store, "g7")) == 3 and len(_chunks(session_store, "g8")) == 3


def test_reread_before_unlink_skips_a_file_the_head_started_naming(session_store, monkeypatch):
    """Fails if gc trusts the manifest it read at the start for the whole pass."""
    _, live = _orphaned(session_store, "g9")
    real = SM._live_manifest_files_from_prefix
    calls = {"n": 0}

    def flip(head):
        calls["n"] += 1
        files = real(head)
        return files | {"000003.json"} if calls["n"] > 1 else files      # after the first read, the head names 000003

    monkeypatch.setattr(SM, "_live_manifest_files_from_prefix", flip)
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    assert rep["reclaimed"] == 1 and "000003.json" in _chunks(session_store, "g9")


def test_gc_reads_heads_through_the_cheap_prefix_only(session_store, monkeypatch):
    """Fails if gc parses a head with read_bytes/json.loads instead of _read_metadata_json_prefix
    (the unbounded read that killed the webui twice)."""
    _orphaned(session_store, "ga")
    real = SM.Path.read_bytes

    def no_head_reads(self):
        if self.name == "ga.json":
            raise MemoryError("gc must not read a live head whole")
        return real(self)

    monkeypatch.setattr(SM.Path, "read_bytes", no_head_reads)
    rep = SM.gc_sessions(session_store)
    assert rep["would_reclaim"] == 2


def test_emptied_manifest_mid_pass_refuses_the_remaining_candidates(session_store, monkeypatch):
    """Fails if the per-unlink guard drops the `not live_now` clause: once the
    live manifest reads back EMPTY (head went manifested -> unmanifested
    between the top-of-loop read and this unlink -- a violated
    webui-stopped precondition, but this is the only deletion path, so it
    must hold anyway), that must not be read as "references nothing" and
    used to unlink every remaining candidate."""
    _, live = _orphaned(session_store, "gb")
    real = SM._live_manifest_files_from_prefix
    calls = {"n": 0}

    def truthful_once_then_empty(head):
        calls["n"] += 1
        return real(head) if calls["n"] == 1 else set()

    monkeypatch.setattr(SM, "_live_manifest_files_from_prefix", truthful_once_then_empty)
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    assert rep["reclaimed"] == 0
    assert sorted(_chunks(session_store, "gb")) == sorted([live, "000002.json", "000003.json"])


def test_emptied_manifest_mid_pass_is_refused_even_with_include_unmanifested(session_store, monkeypatch):
    """Fails if the mid-pass refusal is scoped to the whole-run
    `include_unmanifested` flag instead of the directory's ENTRY state: a
    directory that entered the loop manifested (non-empty live set at the
    top-of-loop read) and only goes empty mid-pass must be refused
    regardless of `include_unmanifested`, because that flag describes a
    directory that started unmanifested, not one that became so. Fails
    either by re-adding the `not include_unmanifested` exemption to the
    per-unlink guard, or by dropping `entered_manifested` and gating on the
    flag again."""
    _, live = _orphaned(session_store, "gc5")
    real = SM._live_manifest_files_from_prefix
    calls = {"n": 0}

    def truthful_once_then_empty(head):
        calls["n"] += 1
        return real(head) if calls["n"] == 1 else set()

    monkeypatch.setattr(SM, "_live_manifest_files_from_prefix", truthful_once_then_empty)
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True, include_unmanifested=True)
    assert rep["reclaimed"] == 0
    assert sorted(_chunks(session_store, "gc5")) == sorted([live, "000002.json", "000003.json"])


def test_hwm_write_failure_leaves_every_orphan_in_place(session_store, monkeypatch):
    """Fails if gc unlinks any candidate when `_write_seq_hwm` cannot durably
    write the mark -- the mark must land before anything is removed, and a
    write failure means it did not."""
    _, live = _orphaned(session_store, "gh")
    monkeypatch.setattr(M, "_write_seq_hwm", lambda sid, v: False)
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    assert rep["reclaimed"] == 0
    assert sorted(_chunks(session_store, "gh")) == sorted([live, "000002.json", "000003.json"])
    row = rep["sessions"][0]
    assert ".seq_hwm" in row.get("error", "")


def test_a_head_that_will_not_parse_is_reported_unreadable_and_untouched(session_store):
    """Fails if gc treats a head that fails to parse as an empty (== fully
    unmanifested) manifest instead of refusing the session as unreadable."""
    _, live = _orphaned(session_store, "gi")
    head = session_store / "gi.json"
    head.write_bytes(b"{not json")
    _age(head, 3600)
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    assert "gi" in rep["manifest_unreadable"]
    assert sorted(_chunks(session_store, "gi")) == sorted([live, "000002.json", "000003.json"])


def test_a_bak_that_will_not_parse_is_reported_unreadable_and_untouched(session_store):
    """Fails if gc computes candidates when `_bak_manifest_files` cannot rule
    out what an unreadable .bak still needs -- an unreadable .bak must block
    the whole session, not just the files it happens to name."""
    _, live = _orphaned(session_store, "gj")
    (session_store / "gj.json.bak").write_bytes(b"{corrupt")
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    entry = next((e for e in rep["manifest_unreadable"] if e.startswith("gj")), None)
    assert entry is not None and ".bak" in entry
    assert sorted(_chunks(session_store, "gj")) == sorted([live, "000002.json", "000003.json"])


def test_a_bak_only_number_that_is_already_gone_is_never_reissued(session_store):
    """Fails if gc raises `.seq_hwm` only to the highest seq still ON DISK: a
    number the `.bak` names but that is already deleted stays reissuable, the
    next seal hands it to different bytes, and the `.bak` becomes unrestorable
    in exactly the way the mark exists to prevent (ledger L107)."""
    _, live = _orphaned(session_store, "gm")
    bak = json.loads((session_store / "gm.json").read_bytes())
    bak["message_chunks"] = [dict(bak["message_chunks"][0], file="000009.json", seq=9)]
    (session_store / "gm.json.bak").write_text(json.dumps(bak), encoding="utf-8")
    assert not (session_store / "gm.msgs" / "000009.json").exists()
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True)
    assert rep["reclaimed"] == 2
    assert M._read_seq_hwm("gm") >= 9, "a bak-only, already-deleted number must raise the mark too"
    assert M._next_chunk_seq("gm") >= 10


def test_include_unmanifested_reclaims_an_unreferenced_chunk_dir(session_store):
    """Fails if `include_unmanifested=True` does not actually enable reclaiming
    a directory whose head no longer carries a message_chunks manifest --
    the same shape `test_archived_headless_and_unmanifested_dirs_are_skipped`
    proves is left alone WITHOUT the flag."""
    _orphaned(session_store, "gk")
    head_path = session_store / "gk.json"
    head = json.loads(head_path.read_bytes())
    head.pop("message_chunks")
    head["messages"] = [_msg(i) for i in range(30)]
    head_path.write_text(json.dumps(head), encoding="utf-8")
    _age(head_path, 3600)
    rep = SM.gc_sessions(session_store, apply=True, webui_stopped=True, include_unmanifested=True)
    assert rep["reclaimed"] == 3
    assert _chunks(session_store, "gk") == []
