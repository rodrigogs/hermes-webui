"""Maintenance for segmented session sidecars: rollback and inspection.

A storage format without a way back is a trap, so `unchunk_session` folds a
segmented session into the single-file layout it had before segmenting.
`fsck_sessions` reports what is on disk and NEVER modifies it -- orphan
chunks are expected after a crash or a re-seal, and deleting them
automatically (even implicitly, even for chunks a successful unchunk just
finished reading) is how a bug turns into data loss: a `<sid>.json.bak` head
can reference chunk files the LIVE head no longer names, and destroying one
of those is exactly what makes that `.bak` unrestorable. An earlier round on
this branch proved it: doing so turned a 30-message `.bak` into a 4-message
one.
"""
import json
import logging
import os
import threading
from pathlib import Path

import api.models as M

logger = logging.getLogger(__name__)


def unchunk_session(sid) -> dict:
    """Fold a segmented session back into one file. Idempotent.

    Refuses (and changes nothing) when any sealed chunk cannot be read --
    the whole point of unchunking is to end up with a file that holds
    everything, so a partial read must never be accepted as "done".

    What gets folded is `doc['messages']` -- what `_read_sidecar_document`
    ACTUALLY reassembled and verified (sha256 + count per chunk) -- not a sum
    over the manifest. A manifest can claim a count a damaged chunk did not
    deliver; folding from the claim instead of the verified read would write
    a "complete" file that silently disagrees with what was checked.

    Chunk cleanup removes only the FILES named in the manifest that was just
    folded, one at a time, and removes the directory only once it is empty.
    A blind `rmtree` of the whole `.msgs` directory would also destroy any
    OTHER file sitting in it -- most importantly an orphan that only this
    session's `.bak` still references (see module docstring). Any such
    leftover simply keeps the directory (and itself) on disk; that is not a
    failure of unchunking, which has already succeeded once the head is
    rewritten to hold everything.
    """
    out = {'session_id': sid, 'unchunked': False, 'messages': 0, 'removed_chunks': 0}
    if not M.is_safe_session_id(sid):
        out['error'] = 'unsafe session id'
        return out
    head = M.SESSION_DIR / f'{sid}.json'
    doc = M._read_sidecar_document(head, sid)
    if doc is None:
        out['error'] = 'head unreadable'
        return out
    out['messages'] = len(doc.get('messages') or [])
    if doc.get('chunk_errors'):
        # Refuse BEFORE touching anything: a hole in the reassembled history
        # must never be accepted as the new single-file truth.
        out['error'] = f"refusing: {'; '.join(doc['chunk_errors'])}"
        return out
    manifest = M._normalised_manifest(doc.get('message_chunks'))
    if not manifest:
        return out  # already a single file; nothing to do
    doc.pop('message_chunks', None)
    payload = json.dumps(doc, ensure_ascii=False, indent=2)
    tmp = head.with_suffix(f'.tmp.{os.getpid()}.{threading.current_thread().ident}')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        M._safe_replace(tmp, head)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        out['error'] = f'write failed: {exc}'
        return out
    # The fold is done and durable: the head now holds everything, no matter
    # what happens to the (now-redundant) chunk files below.
    out['unchunked'] = True
    d = M._session_chunk_dir(sid)
    if d is not None and d.is_dir():
        removed = 0
        for entry in manifest:
            fname = entry.get('file')
            if not fname or Path(fname).name != fname:
                continue  # never touch anything that isn't a plain chunk filename
            try:
                (d / fname).unlink()
                removed += 1
            except OSError:
                logger.debug('unchunk %s: could not remove chunk %s', sid, fname, exc_info=True)
        out['removed_chunks'] = removed
        try:
            d.rmdir()
        except OSError:
            # Not empty -- an orphan the manifest never named (a crash
            # residue, a re-seal residue, or a `.bak`'s referent) is
            # deliberately left alone -- or some other best-effort failure.
            # Either way the fold already succeeded and must not be reported
            # as failed because of this.
            logger.debug('unchunk %s: chunk dir not removed (not empty?)', sid, exc_info=True)
    return out


def fsck_sessions(session_dir=None) -> dict:
    """Report segmented sidecars, their integrity, and orphan chunks.

    Read-only by design: nothing here writes, unlinks, or truncates anything.
    An orphan chunk is the EXPECTED residue of a crash between sealing a
    chunk and writing the head, and of every re-seal (which orphans the old
    chunks ON PURPOSE, per `Session.save`) -- it is a thing to report, not to
    reap.

    Orphan detection consults BOTH the live head's manifest and its `.bak`'s
    manifest (when a `.bak` exists) before calling a chunk file orphaned: a
    `.bak` head can name a chunk the live head no longer does (e.g. the live
    head re-sealed after a shrink, orphaning the `.bak`'s chunks from the
    live head's point of view while the `.bak` still needs them), and a
    chunk only the `.bak` still references is exactly what makes that `.bak`
    restorable. So: this function's orphan list IS cross-checked against the
    `.bak` -- an entry only ends up in `orphans` when NEITHER the live head
    NOR its `.bak` names it.

    Orphan chunk directories are discovered by walking `*.msgs` DIRECTORIES
    on disk, not by walking the heads found above: a `.msgs` directory can
    outlive its head (e.g. a crash after the head was removed but before its
    chunk directory was), and such a directory's entire contents would be
    invisible to a walk that only ever looks at heads.
    """
    session_dir = session_dir or M.SESSION_DIR
    report = {'sessions': [], 'orphans': [], 'errors': []}
    referenced_by_sid: dict[str, set] = {}
    for head in sorted(session_dir.glob('*.json')):
        if head.name.startswith('_'):
            continue
        sid = head.stem
        doc = M._read_sidecar_document(head, sid)
        live_referenced = set()
        if doc is None:
            report['errors'].append({'session_id': sid, 'error': 'head unreadable'})
        else:
            manifest = M._normalised_manifest(doc.get('message_chunks'))
            live_referenced = {e['file'] for e in manifest}
            if manifest:
                report['sessions'].append({
                    'session_id': sid,
                    'chunks': len(manifest),
                    'total_messages': len(doc.get('messages') or []),
                    'claimed_messages': doc.get('message_count'),
                    'errors': list(doc.get('chunk_errors') or []),
                })
        bak_referenced = set()
        bak_path = head.with_suffix('.json.bak')
        if bak_path.exists():
            try:
                bak_doc = json.loads(bak_path.read_bytes())
                if isinstance(bak_doc, dict):
                    bak_referenced = {e['file'] for e in
                                       M._normalised_manifest(bak_doc.get('message_chunks'))}
            except Exception:
                report['errors'].append({
                    'session_id': sid,
                    'error': '.bak unreadable; its chunks could not be excluded from orphan detection',
                })
        referenced_by_sid[sid] = live_referenced | bak_referenced

    for d in sorted(session_dir.glob('*.msgs')):
        if not d.is_dir():
            continue
        sid = d.name[:-len('.msgs')]
        if not M.is_safe_session_id(sid):
            report['errors'].append({'session_id': sid, 'error': 'unsafe session id in chunk dir name'})
            continue
        referenced = referenced_by_sid.get(sid, set())
        for f in sorted(d.glob('*.json')):
            if f.name not in referenced:
                report['orphans'].append({'session_id': sid, 'file': str(f), 'bytes': f.stat().st_size})
    return report
