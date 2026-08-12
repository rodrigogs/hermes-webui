// Tests for defects C, D and E in the stale-runtime banner.
//
// These are written to FAIL against the current module and pass once each defect
// is fixed — that is the only property that makes them worth adding. Each test
// names the user-visible failure, not the implementation detail.
//
// Run: node --test tests/js/runtime_stale_copy.test.js

const fs = require('fs');
const path = require('path');
const assert = require('assert');
const { test, beforeEach } = require('node:test');

const SRC = path.join(__dirname, '..', '..', 'static', 'runtime_stale.js');

// ── DOM ───────────────────────────────────────────────────────────────────
class El {
  constructor(id) {
    this.id = id || '';
    this.hidden = false;
    this.disabled = false;
    this.textContent = '';
    this.value = '';
    this._cls = new Set();
    this.classList = {
      add: (c) => this._cls.add(c),
      remove: (c) => this._cls.delete(c),
      contains: (c) => this._cls.has(c),
    };
  }
}

// The static copy index.html ships inside #runtimeStaleDetails. Kept verbatim so
// a copy edit in the markup shows up here as a failure rather than a silent drift.
const STATIC_COPY =
  'The running WebUI still executes the old code. Your message is saved — ' +
  'restart to load the update, then re-send. Re-attach files after the restart.';

let DOM, store, RS, bootInit;

function load({ withComposer = true, withDetails = true } = {}) {
  DOM = {};
  const mk = (id) => (DOM[id] = new El(id));
  mk('runtimeStaleBanner').hidden = true;
  mk('btnRuntimeStaleRestart');
  mk('runtimeStaleDismiss');
  if (withDetails) mk('runtimeStaleDetails').textContent = STATIC_COPY;
  if (withComposer) mk('msg');

  store = {
    _s: {},
    getItem(k) { return Object.prototype.hasOwnProperty.call(this._s, k) ? this._s[k] : null; },
    setItem(k, v) { this._s[k] = String(v); },
    removeItem(k) { delete this._s[k]; },
  };

  // runtimeStaleInit is not exported; the module invokes it itself. Load with
  // readyState 'loading' so the module registers a DOMContentLoaded listener we
  // can fire on demand — otherwise init runs at load time, before a test can
  // arrange the draft, and a `typeof init === 'function'` guard would skip
  // silently instead of failing.
  const pending = [];
  const doc = {
    readyState: 'loading',
    getElementById: (id) => DOM[id] || null,
    addEventListener: (ev, fn) => { if (ev === 'DOMContentLoaded') pending.push(fn); },
  };
  bootInit = () => {
    assert.ok(pending.length, 'the module registered no DOMContentLoaded handler');
    pending.forEach((fn) => fn());
  };

  const src = fs.readFileSync(SRC, 'utf8');
  const mod = { exports: {} };
  new Function('module', 'exports', 'document', 'localStorage', '$',
               'autoResize', 'updateSendBtn', 'api', 'showToast', src)(
    mod, mod.exports, doc, store, (id) => DOM[id] || null,
    () => {}, () => {}, async () => ({ ok: true }), () => {});
  RS = mod.exports;
  return RS;
}

const SERVER_MSG =
  'Hermes Agent was updated while Hermes WebUI was running. ' +
  'Restart Hermes WebUI before retrying this action.';

function staleError() {
  const e = new Error(SERVER_MSG);
  e.status = 409;
  e.body = JSON.stringify({ error: SERVER_MSG, type: 'agent_runtime_stale', retryable: true });
  return e;
}

beforeEach(() => load());

// ── Defect C ──────────────────────────────────────────────────────────────
test('C: showing the banner keeps the recovery instructions', () => {
  // The static copy carries the only two facts the server text cannot know:
  // that the message survived, and that files did not. Overwriting it with the
  // server message trades those facts for a repetition of the banner title.
  RS.runtimeStaleMaybeShow(staleError());
  const shown = DOM.runtimeStaleDetails.textContent;
  assert.ok(/message is saved/i.test(shown),
    `the banner no longer says the message was saved. It reads: ${JSON.stringify(shown)}`);
  assert.ok(/re-attach files/i.test(shown),
    `the banner no longer tells the user to re-attach files. It reads: ${JSON.stringify(shown)}`);
});

test('C: the banner does not merely repeat its own title', () => {
  // #runtimeStaleTitle already reads "Hermes Agent was updated while Hermes
  // WebUI was running"; the details line must add information, not echo it.
  RS.runtimeStaleMaybeShow(staleError());
  const shown = DOM.runtimeStaleDetails.textContent;
  assert.notStrictEqual(shown.trim(), SERVER_MSG.trim(),
    'the details line was replaced by the server message, which repeats the title');
});

// ── Defect D ──────────────────────────────────────────────────────────────
test('D: a saved message survives when the composer is not empty', () => {
  // init refuses to clobber text typed after the reload — correct. But the draft
  // must then still be recoverable; today the key is consumed by the read.
  RS.runtimeStaleSaveDraft('the long message I do not want to lose');
  DOM.msg.value = 'a few words typed after the reload';
  bootInit();

  const stillThere = store.getItem(RS.RUNTIME_STALE_DRAFT_KEY) !== null;
  const inComposer = /do not want to lose/.test(DOM.msg.value);
  assert.ok(stillThere || inComposer,
    'the saved message was neither delivered nor retained — it was destroyed, ' +
    'while the banner promised "Your message is saved"');
});

test('D: a saved message survives when there is no composer to deliver into', () => {
  load({ withComposer: false });
  RS.runtimeStaleSaveDraft('the long message I do not want to lose');
  bootInit();
  assert.notStrictEqual(store.getItem(RS.RUNTIME_STALE_DRAFT_KEY), null,
    'the key was consumed even though #msg was absent, so the text went nowhere');
});

test('D: a delivered message DOES clear the key', () => {
  // The complement of the two above: once delivery succeeds, the draft must not
  // linger and reappear on a later, unrelated boot.
  RS.runtimeStaleSaveDraft('deliver me');
  DOM.msg.value = '';
  bootInit();
  assert.match(DOM.msg.value, /deliver me/, 'the draft was not delivered into an empty composer');
  assert.strictEqual(store.getItem(RS.RUNTIME_STALE_DRAFT_KEY), null,
    'a delivered draft must be consumed, or it will reappear later');
});

// ── Defect E ──────────────────────────────────────────────────────────────
test('E: a second stale failure does not silently destroy the first message', () => {
  RS.runtimeStaleSaveDraft('FIRST unsent message');
  RS.runtimeStaleSaveDraft('SECOND unsent message');
  const raw = store.getItem(RS.RUNTIME_STALE_DRAFT_KEY) || '';
  assert.ok(raw.includes('FIRST'),
    'the first unsent message was overwritten and is unrecoverable');
});

test('E: a stale draft is not restored into an unrelated later session', () => {
  // ts is written today but never read. Either honour it or drop it — writing a
  // timestamp nobody checks implies a freshness guarantee that does not exist.
  store.setItem(RS.RUNTIME_STALE_DRAFT_KEY,
    JSON.stringify({ text: 'something typed last week', ts: 1 }));
  const restored = RS.runtimeStaleRestoreDraft();
  assert.strictEqual(restored, '',
    'a draft from 1970 was restored; ts is written but never checked');
});
