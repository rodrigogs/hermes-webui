"""Tests for the Fork Keeper API bridge.

The bridge answers three routes for a panel whose Sync action merges upstream
into a production git checkout. What matters is not that it returns JSON, but
that it cannot mislead the operator about that merge: a missing CLI must read as
"unavailable" rather than "up to date", an unparsable answer must not be reported
as success, and the False/None routing protocol must be honoured exactly, because
returning the wrong one puts two JSON bodies on the wire.

No test shells out to `hermes` or touches a real repository; `_run` is the single
seam and every case patches it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import urlparse

from api import fork_keeper_bridge as bridge


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeHandler:
    """Captures what the bridge writes, the way BaseHTTPRequestHandler would."""

    def __init__(self):
        self.status = None
        self.headers_sent = {}
        self.chunks = []
        self.ended = False

    def send_response(self, code):
        self.status = code

    def send_header(self, key, value):
        self.headers_sent[key] = value

    def end_headers(self):
        self.ended = True

    @property
    def wfile(self):
        outer = self

        class W:
            def write(self, data):
                outer.chunks.append(data)

        return W()

    def payload(self):
        return json.loads(b"".join(self.chunks).decode())


def _parsed(path):
    return urlparse(path)


def _run_returns(rc, stdout="", stderr=""):
    return lambda args, timeout: (rc, stdout, stderr)


# ---------------------------------------------------------------------------
# routing protocol — False means "not mine", None means "already answered"
# ---------------------------------------------------------------------------

def test_get_returns_false_for_an_unrelated_path():
    """False lets routes.py fall through to its own 404.

    Returning None here would claim the bridge answered when it wrote nothing,
    and the caller would emit no response at all.
    """
    h = FakeHandler()
    assert bridge.handle_fork_keeper_get(h, _parsed("/api/something-else")) is False
    assert h.status is None


def test_post_returns_false_for_an_unknown_fork_keeper_action():
    h = FakeHandler()
    assert bridge.handle_fork_keeper_post(h, _parsed("/api/fork-keeper/nope"), None) is False
    assert h.status is None


def test_get_returns_none_after_answering(monkeypatch):
    """None signals "response already written"; a 404 on top would concatenate
    two JSON bodies on the wire."""
    monkeypatch.setattr(bridge, "_run", _run_returns(0, '{"behind": 0}'))
    h = FakeHandler()
    assert bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status")) is None
    assert h.status == 200
    assert h.ended is True


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_passes_through_the_cli_json(monkeypatch):
    payload = {"behind": 42, "ahead": 3, "diverged": True, "dirty": False,
               "steps": 5, "conflict_prone": 2}
    monkeypatch.setattr(bridge, "_run", _run_returns(0, json.dumps(payload)))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    assert h.status == 200
    assert h.payload() == payload


def test_status_reads_the_last_line_when_the_cli_also_logs(monkeypatch):
    """`sync-fork --json` writes diagnostics to stderr, but a wrapper or a
    deprecation notice on stdout must not break parsing."""
    monkeypatch.setattr(bridge, "_run", _run_returns(
        0, 'find-peer: via cache\n{"behind": 7, "dirty": false}'))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    assert h.status == 200
    assert h.payload()["behind"] == 7


def test_status_reports_503_when_the_cli_is_missing(monkeypatch):
    """The panel must show "unavailable", never a fabricated "up to date": an
    operator who reads 0-behind from a missing CLI stops looking for updates."""
    monkeypatch.setattr(bridge, "_run", _run_returns(127, "", "hermes CLI not found"))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    assert h.status == 503
    assert "not found" in h.payload()["error"]
    assert "behind" not in h.payload()


def test_status_reports_503_when_the_cli_cannot_be_executed(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _run_returns(126, "", "Permission denied"))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    assert h.status == 503


def test_status_reports_502_on_unparsable_output(monkeypatch):
    """Garbage must not be laundered into a 200. 502 says "upstream spoke
    nonsense", which is the honest description."""
    monkeypatch.setattr(bridge, "_run", _run_returns(0, "Traceback (most recent call last):"))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    assert h.status == 502
    assert "could not parse" in h.payload()["error"]


def test_status_reports_502_on_empty_output(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _run_returns(0, ""))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    assert h.status == 502


# ---------------------------------------------------------------------------
# sync and dry-run
# ---------------------------------------------------------------------------

def test_sync_reports_ok_true_on_exit_zero(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _run_returns(0, "  Merged upstream in 3 step(s)."))
    h = FakeHandler()
    bridge.handle_fork_keeper_post(h, _parsed("/api/fork-keeper/sync"), None)
    assert h.status == 200
    assert h.payload()["ok"] is True
    assert "3 step" in h.payload()["reason"]


def test_sync_reports_ok_false_on_nonzero_exit(monkeypatch):
    """A failed merge reported as ok:true would tell the operator their fork
    advanced when it did not — the worst lie this bridge could tell."""
    monkeypatch.setattr(bridge, "_run", _run_returns(
        1, "  Conflict merging 9d4ef04ed. The fork was restored to 44c9871f8."))
    h = FakeHandler()
    bridge.handle_fork_keeper_post(h, _parsed("/api/fork-keeper/sync"), None)
    assert h.status == 200
    assert h.payload()["ok"] is False
    assert "restored" in h.payload()["reason"]


def test_dry_run_passes_the_flag_and_never_merges(monkeypatch):
    """Dry run must be incapable of mutating anything: if it ever omitted the
    flag it would merge while the operator believed they were previewing."""
    seen = {}

    def fake_run(args, timeout):
        seen["args"] = args
        return 0, "  42 commit(s) behind; would merge in 5 step(s).", ""

    monkeypatch.setattr(bridge, "_run", fake_run)
    h = FakeHandler()
    bridge.handle_fork_keeper_post(h, _parsed("/api/fork-keeper/dry-run"), None)
    assert "--dry-run" in seen["args"]
    assert h.payload()["ok"] is True


def test_sync_does_not_pass_the_dry_run_flag(monkeypatch):
    seen = {}

    def fake_run(args, timeout):
        seen["args"] = args
        return 0, "  Merged upstream in 1 step(s).", ""

    monkeypatch.setattr(bridge, "_run", fake_run)
    bridge.handle_fork_keeper_post(FakeHandler(), _parsed("/api/fork-keeper/sync"), None)
    assert "--dry-run" not in seen["args"]


def test_sync_gets_a_longer_timeout_than_status(monkeypatch):
    """A large backlog merges in many steps; a status timeout applied to a sync
    would kill a merge partway, which is exactly what the CLI works to avoid."""
    seen = {}

    def fake_run(args, timeout):
        seen.setdefault("timeouts", []).append(timeout)
        return 0, '{"behind": 0}' if "--json" in args else "  Merged.", ""

    monkeypatch.setattr(bridge, "_run", fake_run)
    bridge.handle_fork_keeper_get(FakeHandler(), _parsed("/api/fork-keeper/status"))
    bridge.handle_fork_keeper_post(FakeHandler(), _parsed("/api/fork-keeper/sync"), None)
    status_timeout, sync_timeout = seen["timeouts"]
    assert sync_timeout > status_timeout


def test_sync_reports_503_when_the_cli_is_missing(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _run_returns(127, "", "hermes CLI not found"))
    h = FakeHandler()
    bridge.handle_fork_keeper_post(h, _parsed("/api/fork-keeper/sync"), None)
    assert h.status == 503
    assert h.payload()["ok"] is False


def test_sync_surfaces_a_timeout_rather_than_claiming_success(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _run_returns(124, "", "timed out after 900s"))
    h = FakeHandler()
    bridge.handle_fork_keeper_post(h, _parsed("/api/fork-keeper/sync"), None)
    assert h.payload()["ok"] is False


# ---------------------------------------------------------------------------
# response shape
# ---------------------------------------------------------------------------

def test_responses_declare_json_and_a_correct_content_length(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _run_returns(0, '{"behind": 1}'))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    body = b"".join(h.chunks)
    assert h.headers_sent["Content-Type"] == "application/json"
    assert int(h.headers_sent["Content-Length"]) == len(body)


def test_a_response_is_written_exactly_once(monkeypatch):
    """Two writes would put two JSON documents in one body — the failure the
    False/None protocol exists to prevent."""
    monkeypatch.setattr(bridge, "_run", _run_returns(0, '{"behind": 0}'))
    h = FakeHandler()
    bridge.handle_fork_keeper_get(h, _parsed("/api/fork-keeper/status"))
    assert len(h.chunks) == 1
    json.loads(h.chunks[0].decode())  # a single valid document


# ---------------------------------------------------------------------------
# CLI resolution
# ---------------------------------------------------------------------------

def test_explicit_hermes_cli_env_wins(monkeypatch, tmp_path):
    exe = tmp_path / "hermes"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("HERMES_CLI", str(exe))
    assert bridge._hermes_cli() == [str(exe)]


def test_a_nonexistent_hermes_cli_env_is_ignored(monkeypatch, tmp_path):
    """A stale env var must not shadow a working install; it points at nothing,
    so resolution has to continue rather than fail."""
    monkeypatch.setenv("HERMES_CLI", str(tmp_path / "gone"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: None)
    assert bridge._hermes_cli() is None


def test_falls_back_to_the_venv_beside_hermes_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_CLI", raising=False)
    venv = tmp_path / "hermes-agent" / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    resolved = bridge._hermes_cli()
    assert resolved[0] == str(venv / "python")
    assert resolved[1:] == ["-m", "hermes_cli.main"]


def test_returns_none_when_no_cli_can_be_found(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_CLI", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: None)
    assert bridge._hermes_cli() is None
