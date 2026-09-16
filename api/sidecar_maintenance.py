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


def _bak_manifest_files(head) -> tuple[set, str | None]:
    """Chunk filenames the sibling ``<head>.bak`` manifest still references.

    ONE definition of "still referenced by the .bak", shared by both
    `unchunk_session` (which must not delete a file this returns) and
    `fsck_sessions` (which must not call a file this returns an orphan). Two
    near-identical loops computing this independently is exactly how the two
    functions drifted: `fsck_sessions` cross-checked the `.bak` from the
    start, `unchunk_session` did not, and the gap was invisible until a
    tail-only shrink (the ORDINARY case -- it does not force a re-seal) left
    the live head and the `.bak` naming the SAME sealed chunk file.

    Returns ``(files, error)``. ``error`` is ``None`` when there is no
    `.bak`, or one exists and parses; it carries a message when a `.bak`
    EXISTS but could not be read or parsed. A caller must not treat "no
    `.bak`" and "an unreadable `.bak`" the same way: the first means nothing
    needs excluding, the second means a chunk's need cannot be RULED OUT and
    must be treated as still referenced.
    """
    bak_path = head.with_suffix('.json.bak')
    if not bak_path.exists():
        return set(), None
    try:
        bak_doc = json.loads(bak_path.read_bytes())
    except Exception as exc:
        return set(), f'.bak unreadable ({exc.__class__.__name__})'
    if not isinstance(bak_doc, dict):
        return set(), '.bak did not parse to a JSON object'
    manifest = M._normalised_manifest(bak_doc.get('message_chunks'))
    return {e['file'] for e in manifest}, None


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
    folded, one at a time, MINUS any of those filenames the sibling `.bak`
    still references (via `_bak_manifest_files`) -- and removes the
    directory only once it is empty. A blind `rmtree` of the whole `.msgs`
    directory, or removing every live-manifest filename without the `.bak`
    check, would destroy a chunk the `.bak` needs: a tail-only shrink does
    NOT force a re-seal, so the live head and the `.bak` routinely name the
    exact SAME sealed chunk file, not just different ones. If the `.bak`
    can't be read at all, nothing named in the live manifest is removed --
    an unreadable `.bak` means its needs cannot be ruled out, not that it has
    none. Any file left behind this way simply keeps the directory (and
    itself) on disk; that is not a failure of unchunking, which has already
    succeeded once the head is rewritten to hold everything.
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
        bak_referenced, bak_err = _bak_manifest_files(head)
        if bak_err:
            # Cannot rule out the .bak needing one of these files -- remove
            # NONE of them rather than guess. The fold already succeeded;
            # only cleanup is skipped.
            logger.error('unchunk %s: not removing any chunk file -- %s; its needs could not be ruled out',
                          sid, bak_err)
        else:
            removed = 0
            for entry in manifest:
                fname = entry.get('file')
                if not fname or Path(fname).name != fname:
                    continue  # never touch anything that isn't a plain chunk filename
                if fname in bak_referenced:
                    # The .bak's manifest still names this exact file -- a
                    # tail-only shrink does not force a re-seal, so the live
                    # head and the .bak can share a sealed chunk. Removing it
                    # would make that .bak unrestorable. Leave it.
                    continue
                try:
                    (d / fname).unlink()
                    removed += 1
                except OSError:
                    logger.debug('unchunk %s: could not remove chunk %s', sid, fname, exc_info=True)
            out['removed_chunks'] = removed
            try:
                d.rmdir()
            except OSError:
                # Not empty -- a chunk the .bak still needs, an orphan the
                # manifest never named (a crash residue or a re-seal
                # residue), or some other best-effort failure. Either way the
                # fold already succeeded and must not be reported as failed
                # because of this.
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
        bak_referenced, bak_err = _bak_manifest_files(head)
        if bak_err:
            report['errors'].append({
                'session_id': sid,
                'error': f'{bak_err}; its chunks could not be excluded from orphan detection',
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
