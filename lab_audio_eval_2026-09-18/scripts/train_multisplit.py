#!/usr/bin/env python3
"""Repeat the report's action-prediction experiment over eight episode splits.

All variants use the same episodes within each split. The saved experiment
scales inputs and targets with the full dataset and selects the lowest
validation loss. It has no independent test set. Overlapping splits are not
independent trials, so their spread is descriptive, not a significance test.
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_head import ActionHead  # noqa: E402

from paths import DER
from paths import OUT


def run_split(E, L, Y, AN, offs, anchors, ep_of_anchor, sample_ids, n_ep,
              seed, args, dev, dims):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb_dim, low_dim, chunk_len, act_dim = dims

    val_eps = set(rng.permutation(n_ep)[
        :max(1, int(round(args.val_frac * n_ep)))].tolist())
    is_val = np.array([ep_of_anchor[m] in val_eps for m in sample_ids])
    tr = torch.tensor(sample_ids[~is_val], dtype=torch.long, device=dev)
    va = torch.tensor(sample_ids[is_val], dtype=torch.long, device=dev)

    def batch(ids):
        a = AN[ids]
        return E[a[:, None] + offs[None, :]], L[a], Y[ids]

    out = {}
    with torch.no_grad():
        out['mean'] = float((batch(va)[2] ** 2).mean())

    for name, ua, up in (('pose', False, True), ('audio', True, False),
                         ('both', True, True)):
        torch.manual_seed(seed)
        model = ActionHead(emb_dim, low_dim, chunk_len, act_dim,
                           use_audio=ua, use_pose=up).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best = float('inf')
        for _ in range(args.epochs):
            model.train()
            perm = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(perm), args.batch):
                ids = perm[i:i + args.batch]
                a, p, y = batch(ids)
                loss = ((model(a, p) - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                preds = [model(*batch(va[i:i + 1024])[:2])
                         for i in range(0, len(va), 1024)]
                va_loss = float(((torch.cat(preds) - Y[va]) ** 2).mean())
            best = min(best, va_loss)
        out[name] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, default=8)
    ap.add_argument('--hist', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)

    d = np.load(DER / 'wipe_audio_dataset.npz')
    emb, low = d['embedding'], d['low_dim']
    chunks, anchors, ends = d['action_chunks'], d['anchors'], d['episode_ends']
    starts = np.concatenate([[0], ends[:-1]])
    n_ep, chunk_len, act_dim = len(ends), chunks.shape[1], chunks.shape[2]

    ep_of_anchor = np.searchsorted(ends, anchors, side='right')
    ok = anchors - args.hist + 1 >= starts[ep_of_anchor]
    sample_ids = np.where(ok)[0]

    # Keep the full-dataset scaling used for the published results.
    # This includes validation data and limits how these scores can be used.
    e_mu, e_sd = emb.mean(0), emb.std(0) + 1e-6
    l_mu, l_sd = low.mean(0), low.std(0) + 1e-6
    flat = chunks.reshape(-1, act_dim)
    a_mu, a_sd = flat.mean(0), flat.std(0) + 1e-6

    E = torch.tensor((emb - e_mu) / e_sd, dtype=torch.float32, device=dev)
    L = torch.tensor((low - l_mu) / l_sd, dtype=torch.float32, device=dev)
    Y = torch.tensor((chunks - a_mu) / a_sd, dtype=torch.float32, device=dev)
    AN = torch.tensor(anchors, dtype=torch.long, device=dev)
    offs = torch.arange(-args.hist + 1, 1, device=dev)
    dims = (emb.shape[1], low.shape[1], chunk_len, act_dim)

    print(f'{n_ep} episodes, {len(sample_ids)} samples, {args.splits} splits, '
          f'{args.epochs} epochs each')
    print(f'{"split":>5s} {"mean":>9s} {"pose":>9s} {"audio":>9s} {"both":>9s} '
          f'{"pose-both":>10s} {"pose-audio":>11s}')

    rows = []
    for s in range(args.splits):
        r = run_split(E, L, Y, AN, offs, anchors, ep_of_anchor, sample_ids,
                      n_ep, s, args, dev, dims)
        r['split'] = s
        r['pose_minus_both'] = r['pose'] - r['both']
        r['pose_minus_audio'] = r['pose'] - r['audio']
        rows.append(r)
        print(f'{s:5d} {r["mean"]:9.4f} {r["pose"]:9.4f} {r["audio"]:9.4f} '
              f'{r["both"]:9.4f} {r["pose_minus_both"]:+10.4f} '
              f'{r["pose_minus_audio"]:+11.4f}', flush=True)

    def stat(key):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.std(ddof=1)

    print()
    print('=' * 74)
    print(f'{args.splits} splits, mean +/- sd')
    print('=' * 74)
    for k in ('mean', 'pose', 'audio', 'both'):
        m, sd = stat(k)
        print(f'  {k:8s} val MSE {m:.4f} +/- {sd:.4f}')

    print()
    dm, dsd = stat('pose_minus_both')
    wins = sum(1 for r in rows if r['pose_minus_both'] > 0)
    print(f'  pose - both   {dm:+.4f} +/- {dsd:.4f}   '
          f'(audio+pose better in {wins}/{args.splits} splits)')
    da, dasd = stat('pose_minus_audio')
    winsa = sum(1 for r in rows if r['pose_minus_audio'] > 0)
    print(f'  pose - audio  {da:+.4f} +/- {dasd:.4f}   '
          f'(audio alone better in {winsa}/{args.splits} splits)')

    print()
    # A difference smaller than its own spread is not a finding.
    if abs(dm) < dsd:
        print('  Adding audio changes mean validation MSE by less than the')
        print('  spread of the paired differences across these splits.')
        print('  This does not establish a reliable benefit from audio.')
    elif dm > 0:
        print(f'  VERDICT: audio+pose beats pose in {wins}/{args.splits} splits, '
              f'by more than the split spread.')
    else:
        print('  VERDICT: adding audio makes it worse, beyond the split spread.')

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'maniwav_multisplit.json', 'w') as f:
        json.dump({'rows': rows, 'config': vars(args)}, f, indent=2)
    print(f'\nwrote {OUT}/maniwav_multisplit.json')


if __name__ == '__main__':
    main()
