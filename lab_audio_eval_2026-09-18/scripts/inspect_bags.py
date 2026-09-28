#!/usr/bin/env python3
"""Inventory every lab bag: topics, types, message counts, duration.

Verifies rather than assumes which topics survived -- in particular /tf,
/tf_static and /joint_states, which decide whether anything beyond audio
is recoverable from this session.
"""
import os
import pathlib
import sqlite3
import sys

from paths import BAGS


def bag_dirs():
    return sorted(d for d in BAGS.iterdir() if d.is_dir())


def db3_of(d):
    hits = sorted(d.glob('*.db3'))
    return hits[0] if hits else None


def inspect(d):
    db = db3_of(d)
    if db is None:
        return {'name': d.name, 'error': 'no .db3 file'}
    meta = sorted(d.glob('metadata*.yaml'))
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    cur = con.cursor()
    cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
    tables = {r[0] for r in cur.fetchall()}
    if not {'topics', 'messages'} <= tables:
        con.close()
        return {'name': d.name, 'error': f'unexpected schema: {sorted(tables)}'}

    cur.execute('SELECT id, name, type FROM topics')
    topics = cur.fetchall()
    rows = []
    for tid, tname, ttype in topics:
        cur.execute(
            'SELECT COUNT(*), MIN(timestamp), MAX(timestamp) '
            'FROM messages WHERE topic_id=?', (tid,))
        n, tmin, tmax = cur.fetchone()
        rows.append({
            'topic': tname, 'type': ttype, 'count': n,
            'span_s': (tmax - tmin) / 1e9 if n and n > 1 else 0.0,
        })
    cur.execute('SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages')
    tmin, tmax, total = cur.fetchone()
    con.close()
    return {
        'name': d.name,
        'db3': db.name,
        'db3_mb': db.stat().st_size / 1e6,
        'metadata_files': [m.name for m in meta],
        'topics': sorted(rows, key=lambda r: -r['count']),
        'total_msgs': total,
        'bag_span_s': (tmax - tmin) / 1e9 if total and total > 1 else 0.0,
    }


def main():
    dirs = bag_dirs()
    print(f'{len(dirs)} bag directories under {BAGS}\n')
    all_topics = {}
    for d in dirs:
        info = inspect(d)
        if 'error' in info:
            print(f'{info["name"]:34s}  ERROR: {info["error"]}')
            continue
        meta_note = ''
        if info['metadata_files'] != ['metadata.yaml']:
            meta_note = f'   [metadata: {info["metadata_files"]}]'
        print(f'{info["name"]:34s}  {info["db3_mb"]:6.2f} MB  '
              f'{info["total_msgs"]:5d} msgs  {info["bag_span_s"]:6.2f} s{meta_note}')
        for r in info['topics']:
            print(f'      {r["topic"]:28s} {r["type"]:44s} n={r["count"]:5d} span={r["span_s"]:6.2f}s')
            all_topics.setdefault(r['topic'], []).append(info['name'])
        print()

    print('=' * 100)
    print('TOPIC PRESENCE ACROSS ALL BAGS')
    print('=' * 100)
    for t, names in sorted(all_topics.items()):
        print(f'{t:30s} present in {len(names)}/{len(dirs)} bags')
    print()
    print('EXPLICIT CHECK -- robot-state topics:')
    for t in ('/tf', '/tf_static', '/joint_states'):
        n = len(all_topics.get(t, []))
        print(f'  {t:16s} {"PRESENT in %d bags" % n if n else "ABSENT from every bag"}')


if __name__ == '__main__':
    sys.exit(main())
