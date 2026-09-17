"""Maintenance for segmented session sidecars: rollback and inspection.

A storage format without a way back is a trap, so `unchunk_session` folds a
segmented session into the single-file layout it had before segmenting, and
`unchunk_all` does the same for the whole store, one session at a time.
`fsck_sessions` reports what is on disk and NEVER modifies it -- orphan
chunks are expected after a crash or a re-seal, and deleting them
automatically (even implicitly, even for chunks a successful unchunk just
finished reading) is how a bug turns into data loss: a `<sid>.json.bak` head
can reference chunk files the LIVE head no longer names, and destroying one
of those is exactly what makes that `.bak` unrestorable. An earlier round on
this branch proved it: doing so turned a 30-message `.bak` into a 4-message
one.
"""
import hashlib
import json
import logging
import os
import threading
import time
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
    must be treated as still referenced. A `.bak` whose METADATA parses is
    ruled out even if its `messages` array is torn: the manifest is the whole
    question here, and the prefix read below answers it without the array.
    """
    bak_path = head.with_suffix('.json.bak')
    if not bak_path.exists():
        return set(), None
    try:
        # Through the CHEAP PREFIX first. `message_chunks` is written before
        # `messages` in every head this code writes, and a `.bak` IS a head, so
        # the 1 MiB prefix carries the whole manifest -- everything this function
        # needs. The full parse was the one unbounded read left in the
        # maintenance tools: the restored `.bak` of the archived production
        # session is 203 MB, and on a 5.9 GiB swapless VM reading it whole is an
        # OOM kill. Fall back to the full parse only when the prefix cannot be
        # had (no top-level `messages` key, or metadata past the budget).
        prefix = M._read_metadata_json_prefix(bak_path)
        bak_doc = json.loads(prefix if prefix is not None else bak_path.read_bytes())
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

    The chunk directory is deliberately left behind holding `.seq_hwm` even
    when every chunk file was removed: deleting the mark along with the
    directory would let a later re-segmentation of this same session reuse a
    released chunk number (spec 2026-09-17 §3.6).

    Run with the webui STOPPED: a cached Session in a live webui can republish
    the old manifest after this rewrote the head (same message count, so the
    cache-freshness check does not notice); spec 2026-09-17 §4.2.
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
            # Spec 2026-09-17 §3.6: raise the mark BEFORE the first unlink so no
            # number released here is ever reissued to a different chunk. The
            # `.bak`'s own seqs count too: a number it names that is ALREADY gone
            # from disk is still reissuable by disk-max alone, and handing it to
            # different bytes is precisely what makes that `.bak` unrestorable.
            highest = max([e['seq'] for e in manifest] +
                          [int(m.group(1)) for m in (M._CHUNK_FILENAME_RE.match(p.name) for p in d.iterdir()) if m] +
                          [int(n[:6]) for n in bak_referenced if M._CHUNK_FILENAME_RE.match(n)])
            if not M._write_seq_hwm(sid, highest):
                logger.error('unchunk %s: could not write .seq_hwm; leaving every chunk in place', sid)
                return out
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
                # §4.2: the head was rewritten above, but a cached Session in a
                # LIVE webui can republish a manifest at any moment (this tool
                # is documented webui-stopped for that reason); re-read the live
                # manifest right before each unlink and skip a file it names.
                live_now = _live_manifest_files_from_prefix(head)
                if live_now is None or fname in live_now:
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
                # residue), the just-written `.seq_hwm` mark (expected: it is
                # never deleted, so the directory stays non-empty for good
                # once it exists), or some other best-effort failure. Either
                # way the fold already succeeded and must not be reported as
                # failed because of this.
                logger.debug('unchunk %s: chunk dir not removed (not empty?)', sid, exc_info=True)
    return out


def unchunk_all(session_dir=None) -> dict:
    """Fold every segmented session back into one file. The whole-store rollback.

    The work list comes from the DIRECTORY LISTING (`*.msgs`), never from
    parsing heads: enumerating what to roll back must not cost what rolling it
    back costs, and a `.msgs` directory is exactly the set of sessions that
    have something to fold -- including the ones a `*.json` walk cannot see.

    One session at a time, via `unchunk_session`, so the peak cost of the whole
    store is the cost of its largest session and never a sum. Nothing is read
    or held here beyond one directory listing.

    A sid whose `<sid>.json` is absent is SKIPPED with a reason and left
    completely alone. Archiving is a rename of the head, so "no head" means
    either an archived session (whose chunks hold its history) or one that is
    gone; there is nothing to fold into, and folding would either resurrect it
    or write a file for a session that no longer exists.

    A failure on one sid is reported and the walk continues. A rollback that
    stops at the first problem leaves the store half-converted, which is worse
    than either end state and invisible to the caller.

    Returns ``{'unchunked': [sid, ...], 'skipped': [{'session_id', 'reason'}, ...]}``.
    """
    session_dir = session_dir or M.SESSION_DIR
    out = {'unchunked': [], 'skipped': []}
    # `unchunk_session` resolves its own paths under M.SESSION_DIR. Enumerating
    # a DIFFERENT directory here would hand it session ids that name unrelated
    # files in the live store -- and this function rewrites heads. Refuse per
    # sid rather than silently fold the wrong ones.
    wrong_dir = Path(session_dir) != Path(M.SESSION_DIR)
    for d in sorted(session_dir.glob('*.msgs')):
        if not d.is_dir():
            continue
        sid = d.name[:-len('.msgs')]
        if not M.is_safe_session_id(sid):
            out['skipped'].append({'session_id': sid, 'reason': 'unsafe session id in chunk dir name'})
            continue
        if wrong_dir:
            out['skipped'].append({
                'session_id': sid,
                'reason': 'session_dir is not the active SESSION_DIR; unchunk_session writes only inside it',
            })
            continue
        if not (M.SESSION_DIR / f'{sid}.json').exists():
            out['skipped'].append({'session_id': sid, 'reason': 'head missing (archived or removed)'})
            continue
        try:
            res = unchunk_session(sid)
        except Exception as exc:
            logger.error('unchunk_all: %s failed (%s)', sid, exc.__class__.__name__, exc_info=True)
            out['skipped'].append({'session_id': sid, 'reason': f'failed ({exc.__class__.__name__})'})
            continue
        if res.get('unchunked'):
            out['unchunked'].append(sid)
        else:
            # A refusal (an unreadable chunk, a torn head) or nothing to do.
            # Either way `unchunk_session` changed nothing.
            out['skipped'].append({'session_id': sid, 'reason': res.get('error') or 'nothing to fold'})
    return out


def _chunk_dir_footprint(d) -> tuple[int, int]:
    """``(files, bytes)`` for one chunk directory, from `os.stat` alone.

    Never opens a chunk. The operator runs these helpers INSIDE the container
    so the environment matches, and one directory on the deployed box holds the
    bulk of a 203 MB session; describing it must not allocate it.
    """
    files = total = 0
    try:
        entries = list(os.scandir(d))
    except OSError:
        return (0, 0)
    for e in entries:
        try:
            if e.is_file():
                files += 1
                total += e.stat().st_size
        except OSError:
            continue  # vanished mid-walk; a report must not fail over one file
    return (files, total)


def _chunk_sha256(path, block=1024 * 1024) -> str | None:
    """sha256 of one chunk, fed the file a megabyte at a time. None if unreadable.

    Blockwise, not `read_bytes()`: the whole point of `verify=True` being opt-in
    is that it stays bounded when it IS asked for. A chunk is only as small as
    the tail limits made it, and this runs on a 5.9 GiB VM with no swap.
    """
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while True:
                buf = f.read(block)
                if not buf:
                    break
                h.update(buf)
    except OSError:
        return None
    return h.hexdigest()


def fsck_sessions(session_dir=None, *, verify=False) -> dict:
    """Report segmented sidecars, their integrity, and orphan chunks.

    Read-only by design: nothing here writes, unlinks, or truncates anything.
    An orphan chunk is the EXPECTED residue of a crash between sealing a
    chunk and writing the head, and of every re-seal (which orphans the old
    chunks ON PURPOSE, per `Session.save`) -- it is a thing to report, not to
    reap.

    BOUNDED BY DEFAULT. Each head is parsed once and each manifest entry is
    checked with `os.stat`: a missing chunk is reported, a present one is
    accepted. No chunk is read. The previous version called
    `_read_sidecar_document` per head, which reads, sha256-es and
    `json.loads`-es every chunk of every session -- on the deployed box (5.9
    GiB, no swap, one archived head of 203,876,949 bytes) that is precisely
    the allocation that got the webui OOM-killed twice. `verify=True` asks for
    the expensive half: each chunk is additionally hashed a megabyte at a time
    and compared against the manifest's `sha256`. So the default answers "is
    anything the manifest names gone?" and `verify=True` answers "and does
    what is there still match?".

    `total_messages` is therefore what the HEAD describes -- the counts its
    manifest claims plus the tail it carries -- not a reassembled count.

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

    A directory whose `<sid>.json` is gone is NOT reported as orphan files.
    The operator archives a session by RENAMING its head to
    `<sid>.json.archived` (one such file is on the deployed box), so a
    `<sid>.json.*` sibling means the session is archived, its chunks still
    hold its history, and they belong under `archived_chunk_dirs`. Only when
    no sibling exists at all is the directory headless
    (`headless_chunk_dirs`). Both are reported as ONE row per directory, from
    `os.stat`: an archived session can have thousands of chunks, and thousands
    of orphan lines describe one decision as if it were thousands.
    """
    session_dir = session_dir or M.SESSION_DIR
    report = {'sessions': [], 'orphans': [], 'headless_chunk_dirs': [],
              'archived_chunk_dirs': [], 'errors': [], 'other_files': []}
    referenced_by_sid: dict[str, set] = {}
    for head in sorted(session_dir.glob('*.json')):
        if head.name.startswith('_'):
            continue
        sid = head.stem
        live_referenced = set()
        chunk_dir = M._session_chunk_dir(sid)
        try:
            doc = json.loads(head.read_bytes())
        # Not a bare `except`: the biggest live head on the deployed box is
        # ~200 MB, so a MemoryError here is a real event, and reporting it as
        # "head unreadable" would send the operator looking for corruption --
        # possibly at a `.bak` -- over a transient allocation failure. Let it
        # escape; only "could not open it" and "not JSON" are findings.
        except (OSError, ValueError):
            doc = None
        if not isinstance(doc, dict):
            report['errors'].append({'session_id': sid, 'error': 'head unreadable'})
        elif chunk_dir is None:
            report['errors'].append({'session_id': sid, 'error': 'unsafe session id; chunks not checked'})
        else:
            raw_manifest = doc.get('message_chunks')
            manifest = M._normalised_manifest(raw_manifest)
            live_referenced = {e['file'] for e in manifest}
            if manifest:
                errors = []
                # Same visibility `_read_sidecar_document` owes a reader: a
                # manifest whose entries stop chaining describes a hole, and the
                # dropped tail must not vanish from the report just because the
                # prefix normalised cleanly.
                dropped = len(raw_manifest) - len(manifest) if isinstance(raw_manifest, list) else 0
                if dropped:
                    errors.append(f'manifest truncated after {len(manifest)} of {len(raw_manifest)} '
                                  f'entries: continuity broke or an entry was malformed')
                for entry in manifest:
                    fname = entry['file']
                    # The manifest is read off disk, not generated -- refuse
                    # anything that is not a bare filename before joining it.
                    if Path(fname).name != fname:
                        errors.append(f'{fname}: refused, not a plain chunk filename')
                        continue
                    f = chunk_dir / fname
                    try:
                        os.stat(f)
                    except FileNotFoundError:
                        errors.append(f'{fname}: missing')
                        continue
                    except OSError as exc:
                        errors.append(f'{fname}: unstattable ({exc.__class__.__name__})')
                        continue
                    if verify:
                        digest = _chunk_sha256(f)
                        if digest is None:
                            errors.append(f'{fname}: unreadable')
                        elif digest != entry['sha256']:
                            errors.append(f'{fname}: sha256 mismatch')
                report['sessions'].append({
                    'session_id': sid,
                    'chunks': len(manifest),
                    'total_messages': M._sealed_total(manifest) + len(doc.get('messages') or []),
                    'claimed_messages': doc.get('message_count'),
                    'errors': errors,
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
        if not (session_dir / f'{sid}.json').exists():
            files, nbytes = _chunk_dir_footprint(d)
            row = {'session_id': sid, 'files': files, 'bytes': nbytes}
            # A `<sid>.json.*` sibling (`.archived`, `.bak`, `.bak.archived`...)
            # means a head still exists under another name -- archiving IS a
            # rename here -- so these chunks are that session's history, not
            # residue. Calling them orphans invites the one action that loses
            # it. Only a directory with no sibling at all is truly headless.
            if any(session_dir.glob(f'{sid}.json.*')):
                report['archived_chunk_dirs'].append(row)
            else:
                report['headless_chunk_dirs'].append(row)
            continue
        referenced = referenced_by_sid.get(sid, set())
        try:
            other = sum(1 for f in d.iterdir() if not M._CHUNK_FILENAME_RE.match(f.name))
        except OSError:
            other = 0
        if other:
            report['other_files'].append({'session_id': sid, 'count': other})
        for f in sorted(d.glob('*.json')):
            if f.name not in referenced:
                report['orphans'].append({'session_id': sid, 'file': str(f), 'bytes': f.stat().st_size})
    return report


def _live_manifest_files_from_prefix(head) -> set | None:
    """Chunk filenames the live head names, read from the CHEAP PREFIX only.

    `message_chunks` sits before `messages` in every head this code writes, so
    `_read_metadata_json_prefix` (1 MiB budget) yields it without touching the
    array. None when the prefix cannot be read -- the caller must then treat
    the session as unknown, never as "references nothing".
    """
    try:
        prefix = M._read_metadata_json_prefix(head)
        if prefix is None:
            return None
        doc = json.loads(prefix)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    return {e['file'] for e in M._normalised_manifest(doc.get('message_chunks'))}


def gc_sessions(session_dir=None, *, apply=False, min_age_s=900, webui_stopped=False,
                include_unmanifested=False) -> dict:
    """Reclaim chunk files that no head names. Operator-invoked, webui STOPPED.

    A candidate is a `NNNNNN.json` in `<sid>.msgs/` that neither the live head
    nor its `.bak` names, whose own mtime AND the head's mtime are older than
    `min_age_s`. Everything else is reported, never touched: directories whose
    head is archived (`<sid>.json.*` sibling) or gone, directories whose live
    manifest is empty while files remain (the post-unchunk / materialize shape
    -- those can be the only copy of messages no head names; pass
    `include_unmanifested=True` to treat them as candidates), sessions whose
    head or `.bak` cannot be read.

    `apply=False` (default) reports `would_reclaim`. `apply=True` requires
    `webui_stopped=True` -- the operator's assertion that no cached Session
    can publish a manifest while this runs; it is not verifiable from here,
    it exists so a script cannot omit the precondition by accident. Before the
    first unlink for a sid the `.seq_hwm` mark is raised to the highest seq in
    the directory OR named by the `.bak` (a released number is never reissued,
    including one the `.bak` names that is already gone); before EACH unlink
    the live head's manifest is re-read and a file it now names is skipped.
    Never removes a head, a `.bak`, `.seq_hwm`, or any non-conforming name.

    `_write_seq_hwm` is read-then-write, not locked: two `apply=True` runs
    racing each other could clobber a higher mark with a lower one. That is
    acceptable here because gc is a webui-stopped operator tool meant to be
    run one at a time, never concurrently with itself; this does not add
    locking.
    """
    session_dir = Path(session_dir or M.SESSION_DIR)
    if apply and not webui_stopped:
        raise ValueError('gc_sessions(apply=True) requires webui_stopped=True: stop the webui '
                         '(docker compose stop webui) so no cached Session can publish while chunks are removed')
    if apply and session_dir != Path(M.SESSION_DIR):
        raise ValueError('gc_sessions(apply=True) only operates on the active SESSION_DIR')
    now = time.time()
    report = {'sessions': [], 'would_reclaim': 0, 'reclaimed': 0, 'bytes_reclaimed': 0,
              'archived_chunk_dirs': [], 'headless_chunk_dirs': [], 'unmanifested_chunk_dirs': [],
              'manifest_unreadable': [], 'skipped_busy': []}
    for d in sorted(session_dir.glob('*.msgs')):
        if not d.is_dir():
            continue
        sid = d.name[:-len('.msgs')]
        if not M.is_safe_session_id(sid):
            report['manifest_unreadable'].append(sid)
            continue
        head = session_dir / f'{sid}.json'
        if not head.exists():
            files, nbytes = _chunk_dir_footprint(d)
            row = {'session_id': sid, 'files': files, 'bytes': nbytes}
            (report['archived_chunk_dirs'] if any(session_dir.glob(f'{sid}.json.*'))
             else report['headless_chunk_dirs']).append(row)
            continue
        live = _live_manifest_files_from_prefix(head)
        if live is None:
            report['manifest_unreadable'].append(sid)
            continue
        # Captured now, at the entry read, so the per-unlink guard below can
        # tell "this directory entered the loop unmanifested (its live set
        # was already empty, e.g. via include_unmanifested=True)" from "this
        # directory entered manifested and went empty MID-PASS" -- the flag
        # alone cannot make that distinction, and conflating the two is
        # exactly what let a mid-pass emptied manifest slip past the guard.
        entered_manifested = bool(live)
        bak_referenced, bak_err = _bak_manifest_files(head)
        if bak_err:
            report['manifest_unreadable'].append(f'{sid} (.bak: {bak_err})')
            continue
        try:
            conforming = [f for f in d.iterdir() if M._CHUNK_FILENAME_RE.match(f.name)]
            head_age = now - head.stat().st_mtime
        except OSError:
            report['manifest_unreadable'].append(sid)
            continue
        if not live and conforming and not include_unmanifested:
            report['unmanifested_chunk_dirs'].append({'session_id': sid, 'files': len(conforming)})
            continue
        if head_age < min_age_s:
            report['skipped_busy'].append(sid)
            continue
        referenced = live | bak_referenced
        candidates = []
        for f in sorted(conforming):
            if f.name in referenced:
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            if now - st.st_mtime < min_age_s:
                continue
            candidates.append((f, st.st_size))
        row = {'session_id': sid, 'candidates': [f.name for f, _ in candidates],
               'bytes': sum(s for _, s in candidates), 'reclaimed': []}
        report['would_reclaim'] += len(candidates)
        if apply and candidates:
            # The `.bak`'s own seqs count too: a number it names that is ALREADY
            # gone from disk is still reissuable by disk-max alone, and handing it
            # to different bytes is what makes that `.bak` unrestorable.
            highest = max([int(M._CHUNK_FILENAME_RE.match(f.name).group(1)) for f in conforming] +
                          [int(n[:6]) for n in bak_referenced if M._CHUNK_FILENAME_RE.match(n)])
            if not M._write_seq_hwm(sid, highest):
                row['error'] = 'could not write .seq_hwm; nothing removed'
                report['sessions'].append(row)
                continue
            for f, size in candidates:
                live_now = _live_manifest_files_from_prefix(head)
                # An empty `live_now` here means the head went from
                # manifested to unmanifested BETWEEN the top-of-loop read
                # and this unlink (a concurrent unchunk_session/materialize
                # -- a violated webui-stopped precondition, but this is the
                # only deletion path, so it must hold anyway). Reading that
                # as "references nothing" would unlink every remaining
                # candidate -- exactly the unmanifested directory shape
                # this function otherwise refuses to touch -- so it is
                # refused here too, but ONLY when this directory entered the
                # loop manifested: a directory that entered UNMANIFESTED (its
                # live set was already empty, via include_unmanifested=True)
                # stays reclaimable -- the refusal is scoped to the
                # directory's entry state, never to the whole-run flag.
                if live_now is None or f.name in live_now or (not live_now and entered_manifested):
                    continue
                try:
                    f.unlink()
                except OSError:
                    logger.debug('gc %s: could not remove %s', sid, f.name, exc_info=True)
                    continue
                row['reclaimed'].append(f.name)
                report['reclaimed'] += 1
                report['bytes_reclaimed'] += size
        report['sessions'].append(row)
    return report
