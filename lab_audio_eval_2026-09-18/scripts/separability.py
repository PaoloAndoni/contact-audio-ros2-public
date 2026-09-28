#!/usr/bin/env python3
"""Do the three actions separate in embedding space -- and can 17 clips say so?

Three things are measured, because the first one alone is misleading:

  1. Silhouette of the true action labels in each encoder's embedding space.
  2. Leave-one-CLIP-out nearest-centroid accuracy. Grouping by clip matters:
     windows from one clip overlap and share a recording, so a random window
     split would leak and inflate accuracy towards 100%.
  3. A label permutation test at the CLIP level. This is the part that decides
     whether any number above chance means anything. With 17 clips over 3
     classes the null distribution is wide, so the observed accuracy must be
     compared against it rather than against 1/3.

A fourth diagnostic settles what the embedding is actually keyed on: if clip
identity is more predictable than action, the encoder is reading a per-recording
signature (noise floor, gain) rather than the action.
"""
import json
import os
import pathlib

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from paths import OUT
RNG = np.random.default_rng(0)
N_PERM = 2000


def load():
    meta = json.loads((OUT / 'window_meta.json').read_text())
    clips = np.array([m['clip'] for m in meta])
    actions = np.array([m['action'] for m in meta])
    return meta, clips, actions


def loclo_nearest_centroid(E, y, groups):
    """Leave-one-clip-out nearest-centroid accuracy, evaluated per window."""
    correct = 0
    total = 0
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        classes = np.unique(y[tr])
        cents = np.stack([E[tr][y[tr] == c].mean(axis=0) for c in classes])
        # Cosine nearest centroid, matching the pipeline's classifier.
        cn = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
        Xn = E[te] / (np.linalg.norm(E[te], axis=1, keepdims=True) + 1e-12)
        pred = classes[np.argmax(Xn @ cn.T, axis=1)]
        correct += int((pred == y[te]).sum())
        total += int(te.sum())
    return correct / total


def loclo_clip_vote(E, y, groups):
    """Same, but one majority-vote prediction per clip -- 17 independent calls."""
    correct = 0
    per_clip = {}
    uniq = np.unique(groups)
    for g in uniq:
        te = groups == g
        tr = ~te
        classes = np.unique(y[tr])
        cents = np.stack([E[tr][y[tr] == c].mean(axis=0) for c in classes])
        cn = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
        Xn = E[te] / (np.linalg.norm(E[te], axis=1, keepdims=True) + 1e-12)
        pred = classes[np.argmax(Xn @ cn.T, axis=1)]
        vals, cnts = np.unique(pred, return_counts=True)
        winner = vals[np.argmax(cnts)]
        truth = y[te][0]
        per_clip[str(g)] = {'pred': str(winner), 'true': str(truth)}
        correct += int(winner == truth)
    return correct / len(uniq), per_clip


def permutation_null(E, y, groups, n_perm=N_PERM):
    """Shuffle the action label BETWEEN CLIPS, keeping clip structure intact."""
    uniq = np.unique(groups)
    clip_label = {g: y[groups == g][0] for g in uniq}
    labels = np.array([clip_label[g] for g in uniq])
    null = []
    for _ in range(n_perm):
        perm = RNG.permutation(labels)
        mapping = dict(zip(uniq, perm))
        y_perm = np.array([mapping[g] for g in groups])
        acc, _ = loclo_clip_vote(E, y_perm, groups)
        null.append(acc)
    return np.array(null)


def main():
    meta, clips, actions = load()
    results = {}
    for enc in ('cnn', 'melstats', 'ast'):
        p = OUT / f'emb_{enc}.npy'
        if not p.exists():
            print(f'{enc}: embeddings missing, skipping')
            continue
        E = np.load(p).astype(np.float64)
        print(f'\n{"="*78}\n{enc.upper()}  embeddings {E.shape}\n{"="*78}')

        sil = silhouette_score(E, actions)
        win_acc = loclo_nearest_centroid(E, actions, clips)
        clip_acc, per_clip = loclo_clip_vote(E, actions, clips)
        # Is clip identity easier to read than action?
        sil_clip = silhouette_score(E, clips)

        print(f'  silhouette (action labels)      {sil:+.3f}   '
              f'(0 = no structure, 1 = perfect)')
        print(f'  silhouette (clip identity)      {sil_clip:+.3f}')
        print(f'  LOCO accuracy, per window       {win_acc:.3f}')
        print(f'  LOCO accuracy, per clip (n=17)  {clip_acc:.3f}  '
              f'({int(round(clip_acc*len(np.unique(clips))))}/{len(np.unique(clips))})')

        null = permutation_null(E, actions, clips)
        p_val = float((null >= clip_acc).mean())
        print(f'  permutation null (clip-level label shuffle, {N_PERM} draws):')
        print(f'      mean {null.mean():.3f}   95th pct {np.percentile(null,95):.3f}   '
              f'max {null.max():.3f}')
        print(f'  p = {p_val:.3f}  -> '
              f'{"NOT distinguishable from chance" if p_val > 0.05 else "above chance"}')

        tsne = TSNE(n_components=2, perplexity=30, init='pca',
                    learning_rate='auto', random_state=0)
        Z = tsne.fit_transform(E)
        np.save(OUT / f'tsne_{enc}.npy', Z)

        results[enc] = {
            'silhouette_action': float(sil), 'silhouette_clip': float(sil_clip),
            'loco_window_acc': float(win_acc), 'loco_clip_acc': float(clip_acc),
            'perm_mean': float(null.mean()),
            'perm_p95': float(np.percentile(null, 95)),
            'perm_max': float(null.max()), 'p_value': p_val,
            'per_clip': per_clip,
        }

    with open(OUT / 'separability.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {OUT}/separability.json')


if __name__ == '__main__':
    main()
