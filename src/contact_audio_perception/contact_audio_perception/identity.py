"""
Encoder identity, canonical hashing, and safe bundle path resolution.

An encoder's identity is not its weights. Identical weights read under a
different architecture config, pooling, activation or output layer land in a
different embedding space, and centroids built in one space are meaningless in
another. So the fingerprint covers the whole identity document -- type, config,
weight provenance and weight digest -- rather than the weight file alone.

Checkpoint names are mutable: 'MIT/ast-finetuned-audioset-10-10-0.4593' can be
repointed at new weights upstream. The immutable commit revision is therefore
part of the identity, resolved from the local cache so it works offline.
"""
import hashlib
import json
import os
from pathlib import Path

WEIGHTS_SOURCES = ('checkpoint', 'bundled', 'none')
METADATA_NAME = 'model.npz'
DEFAULT_WEIGHTS_NAME = 'encoder.safetensors'


def canonical_json(obj):
    """Serialise to bytes deterministically: sorted, compact, UTF-8, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')


def sha256_hex(data):
    """Return the SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def identity_document(encoder_type, encoder_config, weights_source,
                      checkpoint_id='', checkpoint_revision='', weights_sha256=''):
    """Build the document whose hash is the encoder's fingerprint."""
    if weights_source not in WEIGHTS_SOURCES:
        raise ValueError(
            f'weights_source must be one of {WEIGHTS_SOURCES}, got {weights_source!r}')
    return {
        'encoder_type': str(encoder_type),
        'encoder_config': encoder_config or {},
        'weights_source': str(weights_source),
        'checkpoint_id': str(checkpoint_id),
        'checkpoint_revision': str(checkpoint_revision),
        'weights_sha256': str(weights_sha256),
    }


def derive_encoder_id(document):
    """Derive the namespaced encoder id from an identity document."""
    digest = sha256_hex(canonical_json(document))
    return f"{document['encoder_type']}:{document['weights_source']}:{digest[:16]}"


def validate_weight_fields(weights_source, weights_file, weights_sha256,
                           weights_format, checkpoint_id, checkpoint_revision='',
                           require_revision=True):
    """Raise ValueError unless the weight fields are consistent with the source."""
    has_file = bool(weights_file)
    has_hash = bool(weights_sha256)
    if weights_source == 'checkpoint':
        if has_file or has_hash or weights_format:
            raise ValueError(
                "weights_source 'checkpoint' must not carry weights_file, "
                'weights_sha256 or weights_format; the checkpoint is the source')
        if not checkpoint_id:
            raise ValueError("weights_source 'checkpoint' requires a checkpoint_id")
        if require_revision and not checkpoint_revision:
            raise ValueError(
                "weights_source 'checkpoint' requires a checkpoint_revision: a "
                'model id alone is a mutable pointer, so without the commit hash '
                'the bundle cannot say which weights it was built from')
    elif weights_source == 'bundled':
        if not (has_file and has_hash):
            raise ValueError(
                "weights_source 'bundled' requires both weights_file and "
                'weights_sha256')
        if weights_format != 'safetensors':
            raise ValueError(
                f"weights_source 'bundled' requires weights_format 'safetensors', "
                f'got {weights_format!r}')
    else:  # 'none'
        if has_file or has_hash or weights_format:
            raise ValueError(
                "weights_source 'none' must not carry any weight fields")


def safe_bundle_file(bundle_dir, name):
    """
    Resolve a bundle-relative filename, refusing anything that escapes it.

    A model bundle is a file people copy between machines, so the weights
    pointer is treated as untrusted: it must be a plain filename (no separators,
    no '..', not absolute), and after resolution it must still sit directly
    inside the bundle, which also rejects a symlink pointing elsewhere.
    """
    if not name:
        raise ValueError('weights_file is empty')
    if name != os.path.basename(name) or name in ('.', '..'):
        raise ValueError(
            f'weights_file must be a plain relative filename, got {name!r}')
    base = Path(bundle_dir).resolve(strict=True)
    target = base / name
    if target.is_symlink():
        raise ValueError(f'weights_file {name!r} is a symlink; refusing to follow it')
    if not target.is_file():
        raise FileNotFoundError(f'{target} does not exist')
    resolved = target.resolve(strict=True)
    if resolved.parent != base:
        raise ValueError(
            f'weights_file {name!r} resolves to {resolved}, outside the bundle {base}')
    return resolved


def resolve_hf_revision(model_id):
    """
    Return the immutable commit hash a cached checkpoint name points at.

    Read from the local cache so it works without network. A model id alone is
    a mutable pointer and therefore not a complete identity.
    """
    from huggingface_hub import constants

    folder = 'models--' + model_id.replace('/', '--')
    root = Path(constants.HF_HUB_CACHE) / folder
    ref = root / 'refs' / 'main'
    if ref.is_file():
        revision = ref.read_text().strip()
        if revision:
            return revision
    snapshots = root / 'snapshots'
    if snapshots.is_dir():
        names = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
        if len(names) == 1:
            return names[0]
        raise ValueError(
            f'{model_id}: {len(names)} snapshots cached, cannot pick a revision '
            f'unambiguously: {names}')
    raise FileNotFoundError(
        f'{model_id}: no cached revision under {root}. Download the checkpoint '
        f'once so its immutable commit hash can be recorded.')


def dataset_digest(paths, root):
    """
    Return a digest of the CONTENT and the LABEL of every dataset file.

    Content, because clip names cannot detect a WAV being replaced while keeping
    its filename. Label, because the path is recorded class-relative
    ('tap_table/tap_table_001.wav'): hashing basenames alone would let identical
    audio move between classes without changing the digest, which is exactly a
    relabelled dataset.
    """
    root = Path(root).resolve()
    entries = []
    for path in sorted(Path(p).resolve() for p in paths):
        entries.append({
            'name': path.relative_to(root).as_posix(),
            'sha256': sha256_file(path),
        })
    return sha256_hex(canonical_json(entries))[:16]
