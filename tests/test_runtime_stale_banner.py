"""Contract tests for the runtime-stale banner (item 4).

The WebUI backend refuses actions with a typed ``agent_runtime_stale``
marker in two vocabularies (routes.py): ``type`` on HTTP 409 responses
(shared barrier + four hand-built sites) and ``error_type`` inside
background-job payloads with ``error_status: 409`` (manual-compression
worker). The frontend has ONE detector (static/runtime_stale.js) that reads
both vocabularies, a persistent banner with a restart action reusing the
existing profile-aware gateway restart flow (``/api/health/restart``), and a
localStorage bridge so the failed message survives the WebUI restart.

These source-presence tests pin the producer vocabulary to what the
detector reads, so a future producer that invents a third shape fails here
instead of silently going undetected in the browser. Mirrors the pattern of
``test_issue716_agent_heartbeat.py``.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent

INDEX_HTML = (REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
WORKSPACE_JS = (REPO_ROOT / "static" / "workspace.js").read_text(encoding="utf-8")
COMMANDS_JS = (REPO_ROOT / "static" / "commands.js").read_text(encoding="utf-8")
MESSAGES_JS = (REPO_ROOT / "static" / "messages.js").read_text(encoding="utf-8")
ROUTES_PY = (REPO_ROOT / "api" / "routes.py").read_text(encoding="utf-8")
RUNTIME_STALE_JS = (REPO_ROOT / "static" / "runtime_stale.js").read_text(encoding="utf-8")


def test_banner_markup_is_in_the_shell():
    assert 'id="runtimeStaleBanner"' in INDEX_HTML
    assert 'role="alert"' in INDEX_HTML
    assert 'aria-live="assertive"' in INDEX_HTML
    assert 'onclick="runtimeStaleRestart()"' in INDEX_HTML
    assert 'onclick="runtimeStaleDismiss()"' in INDEX_HTML
    # Reuses the agent-health banner visual pattern (persistent, sticky).
    assert 'class="agent-health-banner" id="runtimeStaleBanner"' in INDEX_HTML


def test_banner_script_is_loaded_before_workspace():
    stale_tag = 'src="static/runtime_stale.js?v=__WEBUI_VERSION__"'
    assert stale_tag in INDEX_HTML
    assert INDEX_HTML.index(stale_tag) < INDEX_HTML.index('src="static/workspace.js?v=__WEBUI_VERSION__"')


def test_detector_reads_both_producer_vocabularies():
    assert "body.type || body.error_type" in RUNTIME_STALE_JS
    assert "agent_runtime_stale" in RUNTIME_STALE_JS
    # Detection is marker-strict: only the type/error_type vocabulary. The
    # body.code vocabulary (session_profile_mismatch) must never match — the
    # behavior is pinned by tests/js/runtime_stale.test.js; here we pin that
    # the detector never falls back to matching on `code`.
    assert "kind !== 'agent_runtime_stale'" in RUNTIME_STALE_JS
    assert "body.code" not in RUNTIME_STALE_JS.split("function runtimeStaleInfo")[1]


def test_restart_action_posts_to_health_restart_itself():
    # The banner owns its restart: own button state, own POST to the
    # profile-aware /api/health/restart flow (api/gateway_restart.py ->
    # `hermes gateway restart`), hiding ITSELF on success. It deliberately
    # does NOT delegate to restartGatewayService() (ui.js), which guards on
    # the agent-health banner's button (silent no-op when that banner is not
    # rendered) and hides the WRONG banner on success. The fork-keeper
    # endpoint stays out: it hardcodes the unit name and clears fork-keeper
    # markers.
    assert "api('/api/health/restart', { method: 'POST' })" in RUNTIME_STALE_JS
    assert "runtimeStaleDismiss()" in RUNTIME_STALE_JS
    assert "/api/fork-keeper/restart-gateway" not in RUNTIME_STALE_JS
    # The shared health flow itself is untouched (still the health banner's).
    ui_js = (REPO_ROOT / "static" / "ui.js").read_text(encoding="utf-8")
    assert "api('/api/health/restart', {method: 'POST'})" in ui_js


def test_api_error_path_hooks_the_detector():
    # One hook covers every HTTP-409 producer (barrier consumers + manual sites).
    assert "runtimeStaleMaybeShow(err)" in WORKSPACE_JS


def test_compression_job_payload_hooks_the_detector():
    # The :24812 worker payload arrives via HTTP 200 — it never reaches the
    # api() error hook, so the consumer hooks it where the payload is read.
    assert "runtimeStaleMaybeShow(data)" in COMMANDS_JS
    assert "error_status" in COMMANDS_JS


def test_failed_send_persists_the_draft_for_the_restart():
    assert "runtimeStaleSaveDraft(" in MESSAGES_JS
    assert "runtimeStaleInfo(e)" in MESSAGES_JS
    assert "RUNTIME_STALE_DRAFT_KEY" in RUNTIME_STALE_JS


def test_backend_producers_speak_the_detector_vocabulary():
    # Every producer must use `type` or `error_type` — if a third shape
    # appears, the detector (and this test) must grow with it.
    assert '"type": "agent_runtime_stale"' in ROUTES_PY
    assert '"error_type": "agent_runtime_stale"' in ROUTES_PY
    # Barrier (:21502) + four hand-built sites (:23529, :24869, :25241,
    # :25906) + the worker job payload (:24812).
    assert ROUTES_PY.count("agent_runtime_stale") >= 6
