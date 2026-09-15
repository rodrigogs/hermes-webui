"""Full parses of a session sidecar read bytes, not text.

Every remaining whole-file parse of a sidecar -- `Session.load()`,
`_load_session_from_path()`, and the recovery helpers in
`api/session_recovery.py` -- went through `json.loads(path.read_text())`.
`read_text` decodes through a TextIOWrapper in chunks and materialises a
`str` before json ever sees it. `json.loads()` accepts bytes directly (it
detects UTF-8 itself), so `read_bytes()` skips that whole stage.

Measured on the host against the 111,857,574-byte backup of the production
sidecar, best of 3, warm cache:

    read_text  + json.loads(str)     862 ms
    read_bytes + json.loads(bytes)   453 ms      1.9x

Same parse, same objects, same count. What must NOT change is the contract the
recovery path leans on: a torn file, a non-dict top level and an unreadable
path all still report -1, because `json.loads` on truncated bytes raises the
same `JSONDecodeError` and an undecodable byte still raises a `ValueError`
subclass.

One deliberate, benign difference is documented below: a sidecar that starts
with a UTF-8 BOM used to be "unreadable" (-1 -- json refuses the BOM after
`read_text('utf-8')` leaves it in), and is readable now, because `json.loads`
on bytes detects `utf-8-sig`. `save()` never writes a BOM, so this only ever
makes a hand-edited file MORE recoverable, never less.
"""
import json
import pathlib
from collections import OrderedDict

import pytest

import api.models as M
import api.session_recovery as R


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, "SESSION_DIR", sdir)
    monkeypatch.setattr(M, "SESSIONS", OrderedDict())
    return sdir


def _spy_read_text(monkeypatch, *targets):
    """Count `Path.read_text()` calls on any of the given paths."""
    calls = {"n": 0}
    real = pathlib.Path.read_text
    targets = {pathlib.Path(t) for t in targets}

    def counting(self, *a, **k):
        if pathlib.Path(self) in targets:
            calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(type(pathlib.Path()), "read_text", counting)
    return calls


NON_ASCII = "ação ✓ naïve — 🚀 é́ 中文"


def _make(session_store, sid, n=4):
    s = M.Session(session_id=sid, title=NON_ASCII, workspace=str(session_store.parent),
                  model="glm", messages=[{"role": "user", "content": f"{NON_ASCII} {i}"} for i in range(n)])
    s.save()
    return s


# ── the change: no text decode on the full-parse paths ──────────────────────

def test_session_load_does_not_go_through_read_text(session_store, monkeypatch):
    s = _make(session_store, "b1")
    calls = _spy_read_text(monkeypatch, s.path)

    loaded = M.Session.load("b1")

    assert loaded is not None and len(loaded.messages) == 4
    assert calls["n"] == 0, f"Session.load decoded the sidecar via read_text {calls['n']}x"


def test_recovery_msg_count_does_not_go_through_read_text(session_store, monkeypatch):
    s = _make(session_store, "b2", n=6)
    calls = _spy_read_text(monkeypatch, s.path)

    assert R._msg_count(s.path) == 6
    assert calls["n"] == 0, f"_msg_count decoded the sidecar via read_text {calls['n']}x"


def test_recovery_status_probe_does_not_go_through_read_text(session_store, monkeypatch):
    """inspect_session_recovery_status parses BOTH the live file and its .bak
    (twice each, across the shrink/clear helpers). None of it needs a str."""
    s = _make(session_store, "b3", n=5)
    bak = s.path.with_suffix(".json.bak")
    bak.write_bytes(s.path.read_bytes())
    calls = _spy_read_text(monkeypatch, s.path, bak)

    status = R.inspect_session_recovery_status(s.path)

    assert status["live_messages"] == 5 and status["bak_messages"] == 5
    assert calls["n"] == 0, f"recovery status decoded sidecars via read_text {calls['n']}x"


# ── what must not change ────────────────────────────────────────────────────

def test_non_ascii_content_round_trips_identically(session_store):
    """Bytes in, same characters out. The decode moved from TextIOWrapper to
    json's own UTF-8 detection; the result must be indistinguishable."""
    _make(session_store, "b4", n=2)

    loaded = M.Session.load("b4")

    assert loaded.title == NON_ASCII
    assert loaded.messages[1]["content"] == f"{NON_ASCII} 1"


def test_msg_count_contract_torn_non_dict_and_missing_still_report_minus_one(session_store):
    """The recovery contract: -1 is what makes a .bak win over a truncated live
    file. A bounded read cannot prove a file is not truncated, so this path
    stays a FULL parse -- just a cheaper one -- and the -1 cases are unchanged."""
    s = _make(session_store, "b5", n=3)

    torn = session_store / "torn.json"
    torn.write_bytes(s.path.read_bytes()[:-40])
    assert R._msg_count(torn) == -1, "a truncated sidecar must still read as unreadable"

    index = session_store / "_index.json"
    index.write_text(json.dumps([{"session_id": "x", "message_count": 9}]), encoding="utf-8")
    assert R._msg_count(index) == -1, "a top-level list is not a session"

    assert R._msg_count(session_store / "nope.json") == -1

    bad_utf8 = session_store / "bad.json"
    bad_utf8.write_bytes(b'{"session_id": "bad", "messages": ["\xff\xfe"]}')
    assert R._msg_count(bad_utf8) == -1, "undecodable bytes must still read as unreadable"


def test_utf8_bom_sidecar_is_readable_after_the_change(session_store):
    """Documents the one behavioural difference, and that it points the safe
    way. `read_text('utf-8')` keeps a BOM, json then refuses it, and the file
    read as -1 (unreadable). `json.loads(bytes)` detects utf-8-sig and parses
    it. save() never writes a BOM, so only a hand-edited file is affected --
    and it becomes recoverable rather than discarded."""
    doc = {"session_id": "bom", "title": "T", "created_at": 1.0, "updated_at": 2.0,
           "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    p = session_store / "bom.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(doc).encode("utf-8"))

    assert R._msg_count(p) == 2
