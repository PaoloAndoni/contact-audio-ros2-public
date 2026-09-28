#!/usr/bin/env python3
"""Step 4a: turn ManiWAV `wipe` into per-step audio embeddings + derived actions.

Runs through OUR pipeline, not a fork of theirs. contact_audio_perception is
ROS-free -- spectrogram.py, encoders/, pose_math.py and action.py need only
numpy, torch, torchaudio, scipy and transformers -- so it all imports unchanged
inside the venv. This makes the experiment an extension of our own code that
happens to be fed by their data.

ASSUMPTIONS USED IN THIS EXPERIMENT:
  * 800 samples per stored step are read as 48 kHz audio at 60 Hz.
    probe_rates.py checks motion under this choice; this is not a direct
    check against sensor timestamps.
  * mic_0 is used as the contact channel. The report notes that the source
    sensor setup still needs a direct check.
  * Our action targets are consecutive pose changes over 16 stored steps.
    They differ from ManiWAV's future poses relative to the current pose.

MEMORY STRATEGY: one embedding per TIMESTEP, not per observation. An
observation is a history of consecutive steps, so per-observation would repeat
each AST forward H times. Per-step means N forwards and an (N, 768) array;
an observation is then just a slice. This is also how the live pipeline works.
The 15 GB RGB array is never opened.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation

try:
    import imagecodecs.numcodecs as inc
    inc.register_codecs()
except Exception:                                          # noqa: BLE001
    pass

from paths import REPO
sys.path.insert(0, str(REPO))

from contact_audio_perception import spectrogram as S            # noqa: E402
from contact_audio_perception.action import ActionChunker        # noqa: E402
from contact_audio_perception.contact_model import ContactModel  # noqa: E402
from contact_audio_perception.encoders import build_encoder_for_model  # noqa: E402
from contact_audio_perception.encoders.ast_fbank import pad_and_normalise  # noqa: E402
from contact_audio_perception.pose_math import rotation_to_6d, quaternion_to_matrix  # noqa: E402

from paths import ZIP
from paths import DER as OUT
OUT.mkdir(parents=True, exist_ok=True)

AUDIO_RATE = 48000          # measured: 800 samples/step at 60 Hz steps
STEP_HZ = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=0, help='0 = all')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--window-s', type=float, default=1.0)
    ap.add_argument('--chunk', type=int, default=16)
    # 60 Hz / 4 = 15 Hz, which IS this pipeline's spectrogram contract -- and
    # gives 30 embedding timestamps over about 2 s; their audio spans 2.93 s.
    # It is also a 4x speedup, but the rate is chosen to match, not to be fast.
    ap.add_argument('--emb-stride', type=int, default=4)
    ap.add_argument('--mic', default='mic_0')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='wipe_audio_dataset')
    args = ap.parse_args()

    store = zarr.ZipStore(str(ZIP), mode='r')
    root = zarr.open(store, mode='r')

    mic = root[f'data/{args.mic}']
    pos_a = root['data/robot0_eef_pos']
    rot_a = root['data/robot0_eef_rot_axis_angle']
    grip_a = root['data/robot0_gripper_width']
    ends = np.asarray(root['meta/episode_ends'][:]).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])

    samples_per_step = int(mic.shape[1])
    win_steps = int(round(args.window_s * STEP_HZ))
    win_samples = win_steps * samples_per_step
    n_ep = len(ends) if args.episodes <= 0 else min(args.episodes, len(ends))

    print(f'mic            data/{args.mic} {mic.shape}')
    print(f'audio rate     {AUDIO_RATE} Hz, {STEP_HZ} Hz steps, '
          f'{samples_per_step} samples/step')
    print(f'window         {args.window_s}s = {win_steps} steps = {win_samples} samples')
    print(f'episodes       {n_ep} of {len(ends)}')

    bundle = REPO / 'models' / 'contact_model_ast'
    model = ContactModel.load(bundle)
    enc = build_encoder_for_model(model)
    print(f'encoder        {enc.encoder_id}')
    print(f'preprocessing  {enc.required_preprocessing_id}')
    print(f'device         {args.device}')

    # Parity reference is taken on the CPU, BEFORE the model moves to the GPU:
    # enc.embed() builds its tensor on the CPU, so calling it against a CUDA
    # model raises. Take the reference first, then move, then compare.
    probe = np.asarray(mic[0:win_steps], dtype=np.float64).reshape(-1)
    probe_i16 = np.clip(probe * 32768.0, -32768, 32767).astype(np.int16)
    probe_fb, _ = S.buffer_to_fbank(probe_i16, 1, AUDIO_RATE)
    if probe_fb.shape[1] != 128:
        raise SystemExit(f'fbank has {probe_fb.shape[1]} mels, expected 128')
    reference = enc.embed(probe_fb)

    ast = enc.model.to(args.device).eval()

    def embed_batch(fbs):
        arr = np.stack([pad_and_normalise(fb, enc.max_length, enc.mean, enc.std)
                        for fb in fbs])
        t = torch.from_numpy(arr).to(args.device)
        with torch.no_grad():
            cls = ast(input_values=t).last_hidden_state[:, 0, :]
        return cls.cpu().numpy().astype(np.float32)

    # The batched GPU path must agree with the encoder's own single-item path,
    # or the embeddings below are not the pipeline's. CPU and CUDA kernels
    # differ in the last few bits, hence a tolerance rather than equality.
    max_diff = float(np.max(np.abs(reference - embed_batch([probe_fb])[0])))
    print(f'parity         batched GPU vs enc.embed() CPU: max_diff = {max_diff:.3e}')
    if max_diff > 1e-3:
        raise SystemExit(f'batched path diverges ({max_diff:.3e}) -- refusing to run')
    print(f'fbank shape    {probe_fb.shape}')

    chunker = ActionChunker(chunk_length=args.chunk, frame='ee')

    # Per-episode checkpoints. An interrupted run (the 21 Sept shutdown killed
    # one at episode 24 of 119 and lost every one of them) must cost a single
    # episode, not the whole pass -- so each episode is written the moment it
    # is done, and a restart skips what is already on disk.
    shards = OUT / f'{args.out}_shards'
    shards.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for e in range(n_ep):
        shard = shards / f'ep_{e:04d}.npz'
        if shard.exists():
            print(f'  ep {e:3d}/{n_ep}  cached, skipping', flush=True)
            continue
        s, t = int(starts[e]), int(ends[e])
        n = t - s
        te = time.time()

        audio = np.asarray(mic[s:t], dtype=np.float64).reshape(-1)
        audio = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

        # Embed at every emb_stride-th step (15 Hz), not every step.
        steps = np.arange(0, n, args.emb_stride, dtype=np.int64)
        fbanks = []
        for k in steps:
            hi = (int(k) + 1) * samples_per_step
            lo = max(0, hi - win_samples)
            buf = audio[lo:hi]
            if len(buf) < win_samples:          # left-pad the opening steps
                buf = np.concatenate(
                    [np.zeros(win_samples - len(buf), np.int16), buf])
            fb, _ = S.buffer_to_fbank(buf, 1, AUDIO_RATE)
            fbanks.append(fb)

        embs = [embed_batch(fbanks[i:i + args.batch])
                for i in range(0, len(fbanks), args.batch)]
        E = np.concatenate(embs).astype(np.float32)

        # Poses in OUR convention: [x, y, z, qx, qy, qz, qw].
        # scipy's as_quat() is already XYZW, so no reordering is needed -- the
        # classic silent transposition this package's docstring warns about.
        p = np.asarray(pos_a[s:t], dtype=np.float64)
        rv = np.asarray(rot_a[s:t], dtype=np.float64)
        q = Rotation.from_rotvec(rv).as_quat()
        poses = np.concatenate([p, q], axis=1)
        grip = np.asarray(grip_a[s:t], dtype=np.float64).reshape(-1)

        # Low-dim observation at the SAME 15 Hz instants as the embeddings:
        # position + 6-D rotation + gripper.
        r6 = np.stack([rotation_to_6d(quaternion_to_matrix(qq)) for qq in q[steps]])
        L = np.concatenate([p[steps], r6, grip[steps, None]], axis=1).astype(np.float32)

        # Our targets are 16 consecutive pose changes at the assumed 60 Hz.
        # These differ from ManiWAV targets; only observation anchors are strided.
        keep = steps[steps + args.chunk < n]
        A = np.stack([chunker.chunk(poses, grip, int(k)) for k in keep]
                     ).astype(np.float32) if len(keep) else np.empty(
                         (0, args.chunk, 10), np.float32)

        # Write to a temp file then rename: a crash mid-write must not leave a
        # truncated shard that a later run would happily treat as cached.
        tmp = shard.with_suffix('.tmp.npz')
        np.savez(tmp, embedding=E, low_dim=L, action_chunks=A,
                 n_steps=np.int64(len(steps)), n_chunks=np.int64(len(keep)))
        tmp.rename(shard)

        print(f'  ep {e:3d}/{n_ep}  {n:5d} steps -> {len(steps):4d} @15Hz  '
              f'emb{E.shape}  act{A.shape}  {time.time()-te:6.1f}s', flush=True)

    # ---- assemble the shards in episode order ----------------------------
    print('\nassembling shards...')
    emb_parts, low_parts, act_parts, anchor_parts, ep_end_parts = [], [], [], [], []
    written = 0
    for e in range(n_ep):
        shard = shards / f'ep_{e:04d}.npz'
        if not shard.exists():
            raise SystemExit(f'shard missing for episode {e}: {shard}')
        z = np.load(shard)
        emb_parts.append(z['embedding'])
        low_parts.append(z['low_dim'])
        act_parts.append(z['action_chunks'])
        # Anchors are global indices into the 15 Hz arrays, so they can only be
        # numbered once every preceding episode's length is known.
        anchor_parts.append(np.arange(int(z['n_chunks']), dtype=np.int64) + written)
        written += int(z['n_steps'])
        ep_end_parts.append(written)

    emb = np.concatenate(emb_parts)
    low = np.concatenate(low_parts)
    act = np.concatenate(act_parts)
    anchors = np.concatenate(anchor_parts)
    ep_ends = np.asarray(ep_end_parts, dtype=np.int64)

    np.savez(
        OUT / f'{args.out}.npz',
        embedding=emb, low_dim=low, action_chunks=act, anchors=anchors,
        episode_ends=ep_ends)
    meta = {
        'source': str(ZIP), 'task': 'wipe', 'mic': args.mic,
        'encoder_id': enc.encoder_id,
        'preprocessing_id': enc.required_preprocessing_id,
        'audio_rate': AUDIO_RATE, 'step_hz': STEP_HZ,
        'samples_per_step': samples_per_step,
        'window_s': args.window_s, 'window_steps': win_steps,
        'emb_stride': args.emb_stride,
        'embedding_hz': STEP_HZ / args.emb_stride,
        'chunk_length': args.chunk, 'action_frame': 'ee',
        'episodes': n_ep, 'steps_15hz': int(written),
        'embedding_dim': int(emb.shape[1]), 'low_dim': int(low.shape[1]),
        'n_action_chunks': int(act.shape[0]),
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(OUT / f'{args.out}.json', 'w') as f:
        json.dump(meta, f, indent=2)

    size = (OUT / f'{args.out}.npz').stat().st_size / 2**20
    print()
    print(f'embedding {emb.shape}  low_dim {low.shape}  chunks {act.shape}')
    print(f'on disk   {size:.1f} MB     elapsed {meta["elapsed_s"]}s')
    print(f'wrote     {OUT}/{args.out}.npz')
    store.close()


if __name__ == '__main__':
    main()
