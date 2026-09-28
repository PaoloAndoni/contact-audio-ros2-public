#!/usr/bin/env python3
"""Single-split action prediction with frozen AST features.

Compare a mean target, current pose plus gripper state, audio history, and both.
This is a small follow-up using ManiWAV data, not a reproduction of its policy.
This script fits scaling on training samples. train_multisplit.py keeps the
full-dataset scaling used for the eight-split results in the report.
"""
import argparse
import json
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn

from paths import DER
from paths import OUT
OUT.mkdir(parents=True, exist_ok=True)


class ActionHead(nn.Module):
    """GRU over the audio-embedding history, fused with pose, then an MLP."""

    def __init__(self, emb_dim, low_dim, chunk, action_dim,
                 hidden=256, use_audio=True, use_pose=True):
        super().__init__()
        self.use_audio, self.use_pose = use_audio, use_pose
        fused = 0
        if use_audio:
            self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
            fused += hidden
        if use_pose:
            fused += low_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, chunk * action_dim))
        self.chunk, self.action_dim = chunk, action_dim

    def forward(self, audio, pose):
        parts = []
        if self.use_audio:
            parts.append(self.gru(audio)[1][-1])
        if self.use_pose:
            parts.append(pose)
        x = torch.cat(parts, dim=1)
        return self.head(x).view(-1, self.chunk, self.action_dim)


def main():
    ap = argparse.ArgumentParser()
    # 30 one-second windows at 15 Hz have endpoints about 2 s apart.
    # Their combined audio span is about 2.93 s.
    ap.add_argument('--hist', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dataset', default='wipe_audio_dataset')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    d = np.load(DER / f'{args.dataset}.npz')
    meta = json.loads((DER / f'{args.dataset}.json').read_text())
    emb = d['embedding']            # (N, 768) at 15 Hz
    low = d['low_dim']              # (N, 10)
    chunks = d['action_chunks']     # (M, 16, 10)
    anchors = d['anchors']          # (M,) index into the 15 Hz arrays
    ends = d['episode_ends']        # (E,) in 15 Hz steps
    starts = np.concatenate([[0], ends[:-1]])
    n_ep, chunk_len, act_dim = len(ends), chunks.shape[1], chunks.shape[2]

    print(f'dataset   emb{emb.shape} low{low.shape} chunks{chunks.shape} '
          f'{n_ep} episodes @ {meta["embedding_hz"]} Hz')

    # A sample is usable only if its whole history lies inside its episode.
    ep_of_anchor = np.searchsorted(ends, anchors, side='right')
    ok = anchors - args.hist + 1 >= starts[ep_of_anchor]
    sample_ids = np.where(ok)[0]
    print(f'samples   {len(sample_ids)} of {len(anchors)} '
          f'(history {args.hist} @ 15 Hz = {args.hist/15:.1f} s)')

    val_eps = set(rng.permutation(n_ep)[:max(1, int(round(args.val_frac*n_ep)))].tolist())
    is_val = np.array([ep_of_anchor[m] in val_eps for m in sample_ids])
    tr, va = sample_ids[~is_val], sample_ids[is_val]
    print(f'split     {n_ep-len(val_eps)} train / {len(val_eps)} val episodes '
          f'({len(tr)} / {len(va)} samples)')

    # Normalise from TRAIN only.
    e_mu, e_sd = emb[anchors[tr]].mean(0), emb[anchors[tr]].std(0) + 1e-6
    l_mu, l_sd = low[anchors[tr]].mean(0), low[anchors[tr]].std(0) + 1e-6
    flat = chunks[tr].reshape(-1, act_dim)
    a_mu, a_sd = flat.mean(0), flat.std(0) + 1e-6

    # Flat tables on the GPU; histories are gathered per batch.
    E = torch.tensor((emb - e_mu) / e_sd, dtype=torch.float32, device=dev)
    L = torch.tensor((low - l_mu) / l_sd, dtype=torch.float32, device=dev)
    Y = torch.tensor((chunks - a_mu) / a_sd, dtype=torch.float32, device=dev)
    AN = torch.tensor(anchors, dtype=torch.long, device=dev)
    offs = torch.arange(-args.hist + 1, 1, device=dev)
    print(f'gpu       {(E.nelement()*4 + Y.nelement()*4)/2**20:.0f} MB resident')

    tr_t = torch.tensor(tr, dtype=torch.long, device=dev)
    va_t = torch.tensor(va, dtype=torch.long, device=dev)

    def batch(ids):
        a = AN[ids]
        return E[a[:, None] + offs[None, :]], L[a], Y[ids]

    results = {}
    with torch.no_grad():
        mean_mse = float((batch(va_t)[2] ** 2).mean())
    results['mean'] = {'val_mse': mean_mse}
    print(f'\nbaseline "mean": val MSE {mean_mse:.4f}')

    for name, ua, up in (('pose', False, True), ('audio', True, False),
                         ('both', True, True)):
        model = ActionHead(emb.shape[1], low.shape[1], chunk_len, act_dim,
                           use_audio=ua, use_pose=up).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        nparam = sum(p.numel() for p in model.parameters())
        print(f'\n--- {name} ({nparam/1e6:.2f}M params) ---')
        best, best_pred, hist = float('inf'), None, []
        for ep in range(args.epochs):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            tot = 0.0
            for i in range(0, len(perm), args.batch):
                ids = perm[i:i + args.batch]
                a, p, y = batch(ids)
                loss = ((model(a, p) - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(ids)
            tr_loss = tot / len(perm)
            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(va_t), 1024):
                    a, p, _ = batch(va_t[i:i + 1024])
                    preds.append(model(a, p))
                pred = torch.cat(preds)
                va_loss = float(((pred - Y[va_t]) ** 2).mean())
            if va_loss < best:
                best, best_pred = va_loss, pred.cpu().numpy()
            hist.append({'epoch': ep, 'train': tr_loss, 'val': va_loss})
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f'  epoch {ep:3d}  train {tr_loss:.4f}  val {va_loss:.4f}',
                      flush=True)

        p_denorm = best_pred * a_sd + a_mu
        t_denorm = chunks[va]
        trans = float(np.linalg.norm(p_denorm[:, :, :3] - t_denorm[:, :, :3],
                                     axis=-1).mean())
        rot = float(np.abs(p_denorm[:, :, 3:9] - t_denorm[:, :, 3:9]).mean())
        grip = float(np.abs(p_denorm[:, :, 9] - t_denorm[:, :, 9]).mean())
        results[name] = {
            'val_mse': best, 'history': hist, 'params': int(nparam),
            'trans_err_m': trans, 'rot6d_err': rot, 'grip_err': grip,
            'improvement_vs_mean': (mean_mse - best) / mean_mse,
        }
        print(f'  best val MSE {best:.4f} ({100*(mean_mse-best)/mean_mse:+.1f}% vs floor)')
        print(f'  translation {trans*1000:.2f} mm   rot6d {rot:.4f}   '
              f'gripper {grip:.5f}')

    results['_config'] = vars(args) | {
        'dataset_meta': meta, 'n_train': int(len(tr)), 'n_val': int(len(va)),
        'val_episodes': sorted(val_eps)}
    with open(OUT / 'maniwav_wipe_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 66)
    print('SUMMARY -- does contact audio beat pose alone?')
    print('=' * 66)
    print(f'{"variant":8s} {"val MSE":>10s} {"vs floor":>11s} {"translation":>13s}')
    for k in ('mean', 'pose', 'audio', 'both'):
        r = results[k]
        imp = r.get('improvement_vs_mean')
        te = r.get('trans_err_m')
        print(f'{k:8s} {r["val_mse"]:10.4f} '
              f'{"" if imp is None else f"{100*imp:+10.1f}%"} '
              f'{"" if te is None else f"{te*1000:11.2f} mm"}')
    pose_mse = results['pose']['val_mse']
    for k in ('audio', 'both'):
        delta = 100 * (pose_mse - results[k]['val_mse']) / pose_mse
        verdict = 'BETTER than pose' if delta > 0 else 'no better than pose'
        print(f'\n{k} vs pose: {delta:+.1f}%  -> {verdict}')
    print(f'\nwrote {OUT}/maniwav_wipe_results.json')


if __name__ == '__main__':
    main()
