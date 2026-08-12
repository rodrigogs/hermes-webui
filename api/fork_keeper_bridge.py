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
import re
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
    # The CLI prints "conflicted (N):" and then exactly N indented paths, with the
    # outcome sentence last. Reading the COUNT is what makes this exact: an earlier
    # version ended the list at "the first line containing a space", and git prints
    # a conflicted path with a space unquoted (verified: `docs/my notes.md`), so
    # such a path was taken for the outcome and dropped from the list.
    conflicted: list[str] = []
    reason = ""
    marker = -1
    count = 0
    for i, line in enumerate(lines):
        m = re.match(r"^conflicted \((\d+)\):$", line)
        if m:
            marker, count = i, int(m.group(1))
            break
    if marker >= 0:
        conflicted = lines[marker + 1:marker + 1 + count]
        rest = lines[marker + 1 + count:]
        reason = rest[-1] if rest else (lines[-1] if lines else "")
    else:
        reason = lines[-1] if lines else (err or "").strip()

    payload = {"ok": rc == 0, "reason": reason, "conflicted": conflicted}
    # rc 2 is "the upstream ref does not resolve" — a configuration fault, not a
    # merge outcome. Say so distinctly so the panel does not render it as a
    # merge that simply failed.
    if rc == 2:
        return 400, payload
    return 200, payload


def _state_dir() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "cron" / "state"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _schedules() -> dict:
    """Per-job schedule and next run, read from the cron registry.

    Read from jobs.json rather than shelled out of `hermes cron list`: the panel
    polls this, and a subprocess per poll is a cost the operator pays for a value
    that changes twice a day. A missing or unreadable registry yields {} — the
    panel then says "unknown" rather than inventing a time.
    """
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    raw = _read_json(home / "cron" / "jobs.json")
    jobs = raw.get("jobs") if isinstance(raw.get("jobs"), list) else (raw if isinstance(raw, list) else [])
    out: dict = {}
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        name = str(job.get("name") or "")
        if "fork-keeper" not in name:
            continue
        which = "prs" if "pr" in name.lower() else "sync"
        sched = job.get("schedule") if isinstance(job.get("schedule"), dict) else {}
        out[which] = {
            "job_id": job.get("id") or job.get("job_id"),
            "name": name,
            "enabled": bool(job.get("enabled", True)),
            "schedule": sched.get("display") or sched.get("expr")
            or (f"every {sched.get('minutes')}m" if sched.get("minutes") else ""),
            "next_run": job.get("next_run") or "",
        }
    return out


def _history(job_ids: dict, limit: int = 6) -> list:
    """Recent completed runs per job, newest first.

    The cron's own execution table is the only per-run record that survives — the
    state file holds just the latest outcome. Reading it read-only keeps this a
    view: the panel must never mutate state a cron owns.
    """
    import sqlite3

    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    db = home / "cron" / "executions.db"
    if not db.exists() or not job_ids:
        return []
    rows: list = []
    try:
        # Read-only URI so a panel poll can never lock or alter the cron's db.
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
        try:
            for which, jid in job_ids.items():
                if not jid:
                    continue
                for status, started, finished, error in con.execute(
                    "SELECT status, started_at, finished_at, error FROM executions "
                    "WHERE job_id = ? ORDER BY started_at DESC LIMIT ?",
                    (jid, limit),
                ):
                    rows.append({
                        "job": which,
                        "status": status,
                        "started_at": started,
                        "finished_at": finished,
                        "error": (error or "")[:200] or None,
                    })
        finally:
            con.close()
    except sqlite3.Error:
        return []
    rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return rows[: limit * 2]


def _overview() -> tuple[int, dict]:
    """Everything the panel needs in ONE request.

    Deliberately one endpoint rather than four: the panel renders a single screen,
    and four polls would let it paint a status from one instant beside a schedule
    from another — the two disagreeing on screen with no way for the operator to
    tell which is stale.
    """
    code, status = _status()
    scheds = _schedules()
    job_ids = {k: v.get("job_id") for k, v in scheds.items()}

    marker = _state_dir() / "fork-keeper-restart-pending"
    try:
        pending_commit = marker.read_text().strip()
    except OSError:
        pending_commit = ""

    payload = {
        "status": status,
        "status_code": code,
        "sync": {**_read_json(_state_dir() / "fork-keeper.json"), **(scheds.get("sync") or {})},
        "prs": {**_read_json(_state_dir() / "fork-keeper-prs.json"), **(scheds.get("prs") or {})},
        "history": _history(job_ids),
        "restart_pending": pending_commit,
    }
    # The overview is a view: a failed status read is reported INSIDE it (as
    # status_code) rather than failing the whole screen, so the schedules and the
    # history still render when only the status probe is broken.
    return 200, payload


def _restart_gateway() -> tuple[int, dict]:
    """Restart the gateway user unit, clearing the pending-restart marker.

    This is the one write here that is not a merge. It is offered because the
    marker is otherwise a dead end: the cron cannot restart the gateway itself (a
    scheduled job restarting its own supervisor is the SIGTERM-respawn loop Hermes
    blocks, #30719), so without this the operator must leave the interface to act
    on what the interface told them.
    """
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "restart", "hermes-gateway"],
            capture_output=True, text=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 503, {"ok": False, "reason": f"could not run systemctl: {exc}"}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        return 502, {"ok": False, "reason": detail or "systemctl restart failed"}

    # Clear the marker only after the restart succeeded: a marker cleared on a
    # failed restart would hide staleness that is still true.
    try:
        (_state_dir() / "fork-keeper-restart-pending").unlink(missing_ok=True)
    except OSError:
        pass
    return 200, {"ok": True, "reason": "gateway restarted"}


def handle_fork_keeper_get(handler, parsed) -> bool | None:
    """GET /api/fork-keeper/{status,overview}. False when no route matches."""
    if parsed.path == "/api/fork-keeper/status":
        code, payload = _status()
    elif parsed.path == "/api/fork-keeper/overview":
        code, payload = _overview()
    else:
        return False
    _send(handler, code, payload)
    return None


def handle_fork_keeper_post(handler, parsed, body) -> bool | None:
    """POST /api/fork-keeper/{sync,dry-run,restart-gateway}. False when unmatched."""
    if parsed.path == "/api/fork-keeper/dry-run":
        code, payload = _sync(dry_run=True)
    elif parsed.path == "/api/fork-keeper/sync":
        code, payload = _sync(dry_run=False)
    elif parsed.path == "/api/fork-keeper/restart-gateway":
        code, payload = _restart_gateway()
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
