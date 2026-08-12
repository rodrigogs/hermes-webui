// Persistent, actionable banner for the agent_runtime_stale response family.
//
// WHY THIS EXISTS
// --------------
// When the Agent checkout changes while the WebUI process is alive, the
// backend refuses affected actions with a typed 409 (or, for background
// jobs, a status payload) carrying the marker `agent_runtime_stale`. The
// error text was already surfacing (error bubble in the chat), but it was
// inACTIONABLE: no button, and the user's message was at the mercy of the
// restart the condition demands. This module adds one detector that
// recognizes every producer shape, one persistent banner with a restart
// action, and a localStorage bridge so the failed message survives the
// WebUI restart (composer drafts are cleared by send() before the turn is
// refused, and are otherwise in-memory only).
//
// PRODUCER SHAPES (all in api/routes.py):
//   {type: "agent_runtime_stale", error, retryable}        HTTP 409
//       _agent_runtime_barrier_response (:21483) + its six consumers,
//       plus the four hand-built sites (:23529, :24869, :25241, :25906).
//   {error_type: "agent_runtime_stale", error_status: 409} HTTP 200 job payload
//       manual-compression worker (:24812), consumed by
//       _pollManualCompressionResult (static/commands.js).
//
// The 409 body is raw JSON text on api() Errors (workspace.js attaches
// err.status/err.statusText/err.body) — the detector accepts both an Error
// and a plain parsed payload, so it needs exactly one vocabulary: the
// `type` / `error_type` marker. session_profile_mismatch (which matches on
// body.code) is deliberately NOT detected: different vocabulary, different
// condition, no restart required.

const DEFAULT_RUNTIME_STALE_MESSAGE =
  'Hermes Agent was updated while Hermes WebUI was running. ' +
  'Restart Hermes WebUI before retrying this action.';

const RUNTIME_STALE_DRAFT_KEY = 'hermes-webui-stale-draft';

// Pure detector: returns {message, status} for a stale-runtime error/payload,
// else null. Never throws.
function runtimeStaleInfo(value) {
  if (!value) return null;
  let body = value;
  if (value.body !== undefined && typeof value.body === 'string') {
    try {
      body = JSON.parse(value.body);
    } catch (_) {
      return null;
    }
  }
  if (!body || typeof body !== 'object') return null;
  const kind = body.type || body.error_type;
  if (kind !== 'agent_runtime_stale') return null;
  // Status resolution: an api() Error carries .status (HTTP) AND .body (raw
  // text); a plain job/status payload carries .status as the JOB state
  // ('error'/'done'/'idle') — never treat that as an HTTP status. The
  // presence of string .body is the discriminator.
  const httpStatus =
    value.body !== undefined && typeof value.body === 'string' && value.status !== undefined
      ? value.status
      : null;
  return {
    message:
      typeof body.error === 'string' && body.error
        ? body.error
        : DEFAULT_RUNTIME_STALE_MESSAGE,
    status: Number(httpStatus || body.error_status || 409),
  };
}

// ── Banner (persistent until the restart or an explicit dismiss) ──────────
function runtimeStaleShow(info) {
  const banner = document.getElementById('runtimeStaleBanner');
  if (!banner) return;
  if (info && info.message) {
    const details = document.getElementById('runtimeStaleDetails');
    if (details) details.textContent = info.message;
  }
  banner.hidden = false;
  banner.classList.add('visible');
}

function runtimeStaleDismiss() {
  const banner = document.getElementById('runtimeStaleBanner');
  if (!banner) return;
  banner.classList.remove('visible');
  banner.hidden = true;
}

// Dismiss is session-scoped ON PURPOSE: unlike the agent-heartbeat banner
// (which persists its dismiss in localStorage until recovery), the stale
// condition is server truth that persists until the WebUI restarts. A
// permanent dismiss would recreate the exact silence this banner exists to
// break — the next detection brings it back.
function runtimeStaleMaybeShow(value) {
  const info = runtimeStaleInfo(value);
  if (info) runtimeStaleShow(info);
  return info;
}

// Restart action — OWN POST, deliberately NOT restartGatewayService().
//
// The shared health flow (ui.js) is entangled with the agent-health banner:
// it guards on $('btnRestartGateway') (so it is a silent no-op when that
// banner is not rendered — the button of THIS banner would be dead), and on
// success it hides agentHealthBanner and re-polls the agent health monitor
// (so THIS banner would survive the resolved condition, lying until reload).
// Parametrizing the shared function (button id, dismiss id, hide target,
// skip the health re-poll) would churn a battle-tested flow for one consumer;
// the banner's own needs are smaller and different. So: own button, own
// state, own POST to the same profile-aware endpoint (/api/health/restart ->
// api/gateway_restart.py -> `hermes gateway restart`), which drags the WebUI
// behind the gateway via PartOf. Returns the endpoint payload, or null on
// failure; never throws. Idempotent: a second call while the first is in
// flight is a no-op (button disabled guard).
async function runtimeStaleRestart() {
  const btn = document.getElementById('btnRuntimeStaleRestart');
  if (!btn || btn.disabled) return null;
  const dismissBtn = document.getElementById('runtimeStaleDismiss');
  const originalText = btn.textContent;
  btn.disabled = true;
  if (dismissBtn) dismissBtn.disabled = true;
  btn.textContent = 'Restarting…';
  if (typeof api !== 'function') {
    btn.disabled = false;
    if (dismissBtn) dismissBtn.disabled = false;
    btn.textContent = originalText;
    return null;
  }
  try {
    const res = await api('/api/health/restart', { method: 'POST' });
    if (res && res.ok) {
      // The gateway — and the WebUI behind it via PartOf — is restarting
      // now; the condition this banner reports is resolved by definition.
      // Hide so the banner cannot outlive the condition it reports (the
      // toast covers the short window before the page goes down).
      runtimeStaleDismiss();
      if (typeof showToast === 'function') {
        showToast('Hermes WebUI is restarting — refresh when it comes back.');
      }
    } else if (typeof showToast === 'function') {
      showToast(
        (res && res.error) || 'Failed to restart Hermes WebUI',
        20000,
        'error'
      );
    }
    return res || null;
  } catch (e) {
    if (typeof showToast === 'function') {
      showToast('Failed to restart Hermes WebUI: ' + (e && e.message ? e.message : e), 20000, 'error');
    }
    return null;
  } finally {
    btn.disabled = false;
    if (dismissBtn) dismissBtn.disabled = false;
    btn.textContent = originalText;
  }
}

// ── Draft bridge: the failed message survives the WebUI restart ───────────
// send() clears the composer AND the server-side draft before the turn is
// refused, and the in-memory draft restore dies with the page. Persist the
// original captured text so the user can re-send after the restart with one
// paste. Files cannot be re-staged from localStorage (they are Blob objects)
// — the banner copy says so.
function runtimeStaleSaveDraft(text) {
  if (!text) return;
  try {
    localStorage.setItem(
      RUNTIME_STALE_DRAFT_KEY,
      JSON.stringify({ text: String(text), ts: Date.now() })
    );
  } catch (_) {
    /* storage full/blocked — the in-memory restore still runs */
  }
}

function runtimeStaleRestoreDraft() {
  try {
    const raw = localStorage.getItem(RUNTIME_STALE_DRAFT_KEY);
    if (!raw) return '';
    localStorage.removeItem(RUNTIME_STALE_DRAFT_KEY);
    const data = JSON.parse(raw);
    return data && typeof data.text === 'string' ? data.text : '';
  } catch (_) {
    return '';
  }
}

function runtimeStaleInit() {
  const text = runtimeStaleRestoreDraft();
  if (!text) return;
  const inp = document.getElementById('msg');
  // Never clobber text the user typed (or a server draft restored) meanwhile.
  if (inp && !String(inp.value || '').trim()) {
    inp.value = text;
    if (typeof autoResize === 'function') autoResize();
    if (typeof updateSendBtn === 'function') updateSendBtn();
  }
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', runtimeStaleInit);
  } else {
    runtimeStaleInit();
  }
}

// Node test hook: the detector is pure and DOM-free, so it is unit-testable
// outside the browser (node --test tests/js/runtime_stale.test.js).
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    runtimeStaleInfo,
    runtimeStaleShow,
    runtimeStaleDismiss,
    runtimeStaleMaybeShow,
    runtimeStaleRestart,
    runtimeStaleSaveDraft,
    runtimeStaleRestoreDraft,
    RUNTIME_STALE_DRAFT_KEY,
    DEFAULT_RUNTIME_STALE_MESSAGE,
  };
}
