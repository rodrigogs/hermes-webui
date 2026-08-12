// Unit tests for the runtime-stale detector, draft bridge, and the
// actionable restart path.
//
// Run: node --test tests/js/runtime_stale.test.js
// (or: node --test tests/js/)
//
// The module under test is a browser script; its UMD guard exports the pure
// functions so Node can exercise them without a DOM. The restart-path tests
// install a MINIMAL fake DOM containing ONLY this banner's elements — the
// agent-health banner (btnRestartGateway / agentHealthBanner) is deliberately
// absent, because that absence is exactly what killed the previous delegated
// implementation (defect A: silent no-op), and the health banner being
// present was what caused defect B (success hid the wrong banner).

'use strict';

const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');

const stale = require(path.join(__dirname, '..', '..', 'static', 'runtime_stale.js'));

const DEFAULT_MSG = stale.DEFAULT_RUNTIME_STALE_MESSAGE;

function apiError(status, body) {
  const err = new Error('x');
  err.status = status;
  err.statusText = 'Conflict';
  err.body = typeof body === 'string' ? body : JSON.stringify(body);
  return err;
}

// ── Minimal fake DOM: only the runtime-stale banner exists ─────────────────

function installFakeDom() {
  const elements = new Map();
  function makeEl(id) {
    const el = {
      id,
      hidden: false,
      disabled: false,
      textContent: '',
      classList: {
        _classes: new Set(),
        add(c) {
          this._classes.add(c);
        },
        remove(c) {
          this._classes.delete(c);
        },
        contains(c) {
          return this._classes.has(c);
        },
      },
    };
    elements.set(id, el);
    return el;
  }
  const banner = makeEl('runtimeStaleBanner');
  banner.hidden = true; // matches the `hidden` attribute in index.html
  makeEl('runtimeStaleTitle');
  makeEl('runtimeStaleDetails');
  makeEl('btnRuntimeStaleRestart');
  makeEl('runtimeStaleDismiss');
  // NOTE: NO btnRestartGateway, NO agentHealthBanner — that is the point.
  global.document = {
    readyState: 'complete',
    addEventListener() {},
    getElementById(id) {
      return elements.get(id) || null;
    },
  };
  return {
    elements,
    banner,
    btn: elements.get('btnRuntimeStaleRestart'),
    dismissBtn: elements.get('runtimeStaleDismiss'),
  };
}

function installApiRecorder(result) {
  const calls = [];
  global.api = (path, opts) => {
    calls.push({ path, opts });
    return Promise.resolve(result);
  };
  return calls;
}

// ── Detection: every producer shape in the wild ────────────────────────────

test('detects the shared barrier shape: 409 + {type: agent_runtime_stale}', () => {
  const info = stale.runtimeStaleInfo(
    apiError(409, { error: 'Restart Hermes WebUI', type: 'agent_runtime_stale', retryable: true })
  );
  assert.ok(info);
  assert.equal(info.message, 'Restart Hermes WebUI');
  assert.equal(info.status, 409);
});

test('detects the hand-built manual sites: same shape, different producers', () => {
  const info = stale.runtimeStaleInfo(
    apiError(409, { error: 'restart', type: 'agent_runtime_stale', retryable: true })
  );
  assert.ok(info);
  assert.equal(info.status, 409);
});

test('detects the compression-worker job payload: error_type + error_status', () => {
  // routes.py:24812 — HTTP 200 job payload consumed by _pollManualCompressionResult.
  const info = stale.runtimeStaleInfo({
    status: 'error',
    error: 'Compression failed: restart',
    error_status: 409,
    error_type: 'agent_runtime_stale',
    retryable: true,
  });
  assert.ok(info);
  assert.equal(info.status, 409);
});

test('detects error_type without any status field (status falls back to 409)', () => {
  const info = stale.runtimeStaleInfo({ error: 'restart', error_type: 'agent_runtime_stale' });
  assert.ok(info);
  assert.equal(info.status, 409);
});

test('falls back to the default message when body.error is absent', () => {
  const info = stale.runtimeStaleInfo(apiError(409, { type: 'agent_runtime_stale' }));
  assert.ok(info);
  assert.equal(info.message, DEFAULT_MSG);
});

// ── Non-detection: the confusion cases ─────────────────────────────────────

test('does NOT detect session_profile_mismatch (the real 409 confusion case)', () => {
  const info = stale.runtimeStaleInfo(
    apiError(409, { code: 'session_profile_mismatch', profile: 'work', session_id: 's1' })
  );
  assert.equal(info, null);
});

test('does NOT detect a 409 with an unrelated type', () => {
  const info = stale.runtimeStaleInfo(apiError(409, { error: 'nope', type: 'something_else' }));
  assert.equal(info, null);
});

test('does NOT detect malformed JSON body text', () => {
  const info = stale.runtimeStaleInfo(apiError(409, '{not json'));
  assert.equal(info, null);
});

test('does NOT detect a plain-text (HTML) error body', () => {
  const info = stale.runtimeStaleInfo(apiError(502, '<html>Bad gateway</html>'));
  assert.equal(info, null);
});

test('does NOT detect null / undefined / primitive values', () => {
  assert.equal(stale.runtimeStaleInfo(null), null);
  assert.equal(stale.runtimeStaleInfo(undefined), null);
  assert.equal(stale.runtimeStaleInfo('agent_runtime_stale'), null);
});

test('detection never throws on exotic inputs', () => {
  for (const weird of [{}, [], 42, true, { body: 42 }, { body: null }, { type: null, error_type: null }]) {
    assert.doesNotThrow(() => stale.runtimeStaleInfo(weird));
  }
});

// ── Restart path: own POST, own button, hides itself on success ────────────
// These tests fail against the previous implementation, which delegated to
// restartGatewayService() (ui.js): that function guards on $('btnRestartGateway')
// — absent here — so no POST was ever made (defect A), and its success path
// hid agentHealthBanner, never runtimeStaleBanner (defect B).

test('click with ONLY this banner in the DOM makes the POST (defect A)', async () => {
  const dom = installFakeDom();
  const calls = installApiRecorder({ ok: true, message: 'Gateway service restarted successfully' });
  assert.equal(dom.elements.has('btnRestartGateway'), false); // the trap

  const result = await stale.runtimeStaleRestart();

  assert.ok(result);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, '/api/health/restart');
  assert.equal(calls[0].opts.method, 'POST');
});

test('success hides THIS banner (defect B)', async () => {
  const dom = installFakeDom();
  const calls = installApiRecorder({ ok: true });
  stale.runtimeStaleShow({ message: 'restart needed' });
  assert.equal(dom.banner.hidden, false); // banner is up before the click
  assert.equal(dom.banner.classList.contains('visible'), true);

  await stale.runtimeStaleRestart();

  assert.equal(calls.length, 1);
  assert.equal(dom.banner.hidden, true); // the fix: it must go down
  assert.equal(dom.banner.classList.contains('visible'), false);
});

test('button shows in-flight state and re-enables after the request', async () => {
  const dom = installFakeDom();
  let resolveApi;
  global.api = () =>
    new Promise((res) => {
      resolveApi = res;
    });

  const pending = stale.runtimeStaleRestart();
  // The async body runs synchronously up to the first await: state is set.
  assert.equal(dom.btn.disabled, true);
  assert.equal(dom.btn.textContent, 'Restarting…');

  resolveApi({ ok: true });
  await pending;
  assert.equal(dom.btn.disabled, false);
  assert.equal(dom.btn.textContent, '');
});

test('second click while in flight is a no-op (no double POST)', async () => {
  const dom = installFakeDom();
  let resolveApi;
  let callCount = 0;
  global.api = () => {
    callCount += 1;
    return new Promise((res) => {
      resolveApi = res;
    });
  };

  const p1 = stale.runtimeStaleRestart();
  const p2 = stale.runtimeStaleRestart(); // disabled guard must swallow it
  resolveApi({ ok: true });
  await Promise.all([p1, p2]);

  assert.equal(callCount, 1);
});

test('failure keeps the banner up and re-enables the button', async () => {
  const dom = installFakeDom();
  const toasts = [];
  global.showToast = (...args) => toasts.push(args);
  global.api = () => Promise.reject(new Error('boom'));
  stale.runtimeStaleShow({ message: 'restart needed' });

  const result = await stale.runtimeStaleRestart();

  assert.equal(result, null);
  assert.equal(dom.banner.hidden, false); // NOT hidden on failure — it still reports the truth
  assert.equal(dom.btn.disabled, false);
  assert.ok(toasts.length >= 1);
  delete global.showToast;
});

test('is a safe no-op when the button is not rendered', async () => {
  const dom = installFakeDom();
  const calls = installApiRecorder({ ok: true });
  dom.elements.delete('btnRuntimeStaleRestart');

  const result = await stale.runtimeStaleRestart();

  assert.equal(result, null);
  assert.equal(calls.length, 0);
});

// ── Draft bridge: the failed message survives the WebUI restart ────────────

function installFakeLocalStorage() {
  const store = new Map();
  const fake = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    _store: store,
  };
  global.localStorage = fake;
  return fake;
}

test('draft roundtrip: save then restore, key consumed on restore', () => {
  const ls = installFakeLocalStorage();
  stale.runtimeStaleSaveDraft('my important message');
  assert.ok(ls.getItem(stale.RUNTIME_STALE_DRAFT_KEY));

  const restored = stale.runtimeStaleRestoreDraft();
  assert.equal(restored, 'my important message');
  assert.equal(ls.getItem(stale.RUNTIME_STALE_DRAFT_KEY), null);
});

test('draft restore returns empty string when nothing was saved', () => {
  installFakeLocalStorage();
  assert.equal(stale.runtimeStaleRestoreDraft(), '');
});

test('draft save/restore never throw without localStorage', () => {
  delete global.localStorage;
  assert.doesNotThrow(() => stale.runtimeStaleSaveDraft('x'));
  assert.doesNotThrow(() => stale.runtimeStaleRestoreDraft());
  assert.equal(stale.runtimeStaleRestoreDraft(), '');
});
