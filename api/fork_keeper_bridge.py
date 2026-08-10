"""HTTP bridge for the Fork Keeper panel.

The panel shows whether this install is behind its upstream and offers to merge.
It never runs git itself — it calls ``hermes sync-fork``, the same command the
CLI and the cron job use, so all three surfaces cannot disagree.

Why the agent CLI rather than git here: the merge policy (incremental steps,
refuse a dirty worktree, restore ``HEAD`` on conflict) belongs in one place. A
second implementation in the webui would drift, and the drift would only show up
the day a merge goes wrong.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

_TIMEOUT_STATUS = 30
_TIMEOUT_SYNC = 900  # a large backlog merges in many steps


def _hermes_cli() -> list[str] | None:
    """Locate the agent CLI: explicit env, the venv beside it, then PATH."""
    explicit = os.environ.get("HERMES_CLI")
    if explicit and Path(explicit).exists():
        return [explicit]

    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    # ~/.hermes/hermes-agent is a symlink to the checkout on a standard install.
    venv = home / "hermes-agent" / "venv" / "bin" / "python"
    if venv.exists():
        return [str(venv), "-m", "hermes_cli.main"]

    found = shutil.which("hermes")
    return [found] if found else None


def _run(args: list[str], timeout: int) -> tuple[int, str, str]:
    cli = _hermes_cli()
    if cli is None:
        return 127, "", "hermes CLI not found (set HERMES_CLI)"
    try:
        proc = subprocess.run(
            cli + args, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except OSError as exc:
        return 126, "", str(exc)


def _status() -> tuple[int, dict]:
    rc, out, err = _run(["sync-fork", "--json"], _TIMEOUT_STATUS)
    if rc == 127 or rc == 126:
        return 503, {"error": err}
    if rc == 124:
        return 504, {"error": err}
    try:
        payload = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 502, {"error": "could not parse sync-fork output", "detail": (err or out)[:400]}
    if not isinstance(payload, dict):
        return 502, {"error": "sync-fork returned a non-object status"}
    # The CLI reports an unresolvable upstream ref as {"error": ...} with rc 2.
    # Forwarding that as a 200 would let a caller read a status object with no
    # commit count as though it were a healthy fork.
    if rc == 2 or "error" in payload:
        return 400, payload
    return 200, payload


def _sync(dry_run: bool) -> tuple[int, dict]:
    args = ["sync-fork"] + (["--dry-run"] if dry_run else [])
    rc, out, err = _run(args, _TIMEOUT_SYNC)
    if rc in (126, 127):
        return 503, {"ok": False, "reason": err}
    if rc == 124:
        return 504, {"ok": False, "reason": err}

    text = (out or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # The CLI prints the conflicted list under a "conflicted:" header and the
    # outcome sentence LAST. Parsing by that header rather than by guessing which
    # lines look like paths: the old heuristic ("has a /, has no space") took the
    # last conflicted path as the reason and then excluded it from the list, so on
    # the failure path that matters most the operator saw a bare filename as the
    # explanation and one file missing from the list.
    conflicted: list[str] = []
    reason = ""
    try:
        marker = lines.index("conflicted:")
    except ValueError:
        marker = -1
    if marker >= 0:
        for line in lines[marker + 1:]:
            # The outcome sentence ends the list; a path never contains a space.
            if " " in line:
                reason = line
                break
            conflicted.append(line)
        if not reason:
            reason = lines[-1] if lines else ""
    else:
        reason = lines[-1] if lines else (err or "").strip()

    payload = {"ok": rc == 0, "reason": reason, "conflicted": conflicted}
    # rc 2 is "the upstream ref does not resolve" — a configuration fault, not a
    # merge outcome. Say so distinctly so the panel does not render it as a
    # merge that simply failed.
    if rc == 2:
        return 400, payload
    return 200, payload


def handle_fork_keeper_get(handler, parsed) -> bool | None:
    """GET /api/fork-keeper/status. Returns False when no route matches."""
    if parsed.path != "/api/fork-keeper/status":
        return False
    code, payload = _status()
    _send(handler, code, payload)
    return None


def handle_fork_keeper_post(handler, parsed, body) -> bool | None:
    """POST /api/fork-keeper/{sync,dry-run}. Returns False when no route matches."""
    if parsed.path == "/api/fork-keeper/dry-run":
        code, payload = _sync(dry_run=True)
    elif parsed.path == "/api/fork-keeper/sync":
        code, payload = _sync(dry_run=False)
    else:
        return False
    _send(handler, code, payload)
    return None


def _send(handler, code: int, payload: dict) -> None:
    raw = json.dumps(payload).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)
