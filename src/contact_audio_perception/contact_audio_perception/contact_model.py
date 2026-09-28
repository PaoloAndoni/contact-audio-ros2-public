"""
Model bundle: centroids, the feature contract, and the encoder's identity.

A bundle is a DIRECTORY:

    contact_model_cnn/
      model.npz            metadata + centroids + weights pointer and digest
      encoder.safetensors  present only when weights_source == 'bundled'

A bare (class_names, centroids) pair cannot say what features or what encoder it
was trained on, so applying it elsewhere produces confident, wrong labels and no
error. Everything needed to detect that is recorded here and checked before the
first message is classified.

Weights are loaded through safetensors after a SHA-256 check. torch.load is
never used: it executes pickle, and a bundle is exactly the kind of file that
gets copied between machines.

Format history:
  3  directory bundle, identity-document encoder ids, classifier config
  2  single tagged .npz, encoder_id = checkpoint name       (needs migration)
  -  untagged two-array .npz                                (needs migration)
"""
import json
from pathlib import Path

from contact_audio_perception.contact_classifier import CentroidClassifier
from contact_audio_perception.identity import (
    derive_encoder_id,
    identity_document,
    METADATA_NAME,
    safe_bundle_file,
    sha256_file,
    validate_weight_fields,
)
import numpy as np

BUNDLE_FORMAT = 3
DEFAULT_CLASSIFIER_CONFIG = {
    'type': 'cosine_centroid',
    'temperature': 0.1,
    'unknown_threshold': 0.0,
}


def _text(data, key, default=''):
    """Read a string field from an npz."""
    return str(data[key]) if key in data else default


def _json_field(data, key, default):
    """Read a JSON-encoded field from an npz."""
    if key not in data:
        return default
    raw = str(data[key])
    return json.loads(raw) if raw else default


class ContactModel:
    """Class centroids plus the feature and encoder contract behind them."""

    def __init__(self, class_names, centroids, encoder_type, encoder_config,
                 weights_source, preprocessing_id, expected_n_frames,
                 expected_n_mels, window_duration_sec, checkpoint_id='',
                 checkpoint_revision='', weights_file='', weights_sha256='',
                 weights_format='', encoder_seed=-1, classifier_config=None,
                 fold_id='all', trained_on=None, dataset_digest='',
                 embedding_dim=None, bundle_dir=None, needs_migration=False):
        """Store and validate every field of the bundle."""
        self.class_names = [str(name) for name in class_names]
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.encoder_type = str(encoder_type)
        self.encoder_config = dict(encoder_config or {})
        self.weights_source = str(weights_source)
        self.checkpoint_id = str(checkpoint_id)
        self.checkpoint_revision = str(checkpoint_revision)
        self.weights_file = str(weights_file)
        self.weights_sha256 = str(weights_sha256)
        self.weights_format = str(weights_format)
        self.encoder_seed = int(encoder_seed)
        self.preprocessing_id = str(preprocessing_id)
        self.expected_n_frames = int(expected_n_frames)
        self.expected_n_mels = int(expected_n_mels)
        self.window_duration_sec = float(window_duration_sec)
        self.classifier_config = dict(classifier_config or DEFAULT_CLASSIFIER_CONFIG)
        self.fold_id = str(fold_id)
        self.trained_on = dict(trained_on or {})
        self.dataset_digest = str(dataset_digest)
        self.bundle_dir = Path(bundle_dir) if bundle_dir else None
        self.needs_migration = bool(needs_migration)
        self.embedding_dim = int(
            embedding_dim if embedding_dim is not None else self.centroids.shape[-1])

        if self.centroids.ndim != 2:
            raise ValueError(f'centroids must be 2-D, got {self.centroids.shape}')
        if self.centroids.shape[1] != self.embedding_dim:
            raise ValueError(
                f'centroid width {self.centroids.shape[1]} does not match '
                f'embedding_dim {self.embedding_dim}')
        if len(self.class_names) != self.centroids.shape[0]:
            raise ValueError(
                f'{len(self.class_names)} class names but '
                f'{self.centroids.shape[0]} centroid rows')
        validate_weight_fields(self.weights_source, self.weights_file,
                               self.weights_sha256, self.weights_format,
                               self.checkpoint_id, self.checkpoint_revision,
                               require_revision=not self.needs_migration)

    # ---- identity -------------------------------------------------------

    @property
    def identity(self):
        """Return the identity document this model's encoder_id hashes."""
        return identity_document(
            self.encoder_type, self.encoder_config, self.weights_source,
            self.checkpoint_id, self.checkpoint_revision, self.weights_sha256)

    @property
    def encoder_id(self):
        """Return the namespaced fingerprint of the encoder that built these centroids."""
        return derive_encoder_id(self.identity)

    # ---- load / save ----------------------------------------------------

    @classmethod
    def load(cls, path, allow_fold=False, allow_needs_migration=False):
        """
        Load a bundle directory, or an older single-file model for migration.

        Older formats are readable only with allow_needs_migration, which the
        migration tooling sets and production code does not: silently filling in
        the identity fields they lack would hand back exactly the unverifiable
        model this format exists to reject.
        """
        path = Path(path)
        if path.is_dir():
            model = cls._load_bundle(path)
        elif path.is_file():
            model = cls._load_pre_v3(path)
        else:
            raise FileNotFoundError(f'{path} does not exist')

        if model.needs_migration and not allow_needs_migration:
            raise ValueError(
                f'{path}: pre-v3 model without an encoder identity. Run '
                f'scripts/migrate_centroids.py to produce a bundle, or pass '
                f'allow_needs_migration for diagnosis only.')
        if model.fold_id != 'all' and not allow_fold:
            raise ValueError(
                f'{path}: this is a cross-validation artifact (fold_id='
                f'{model.fold_id!r}), not a deployable model. Fold models are '
                f'trained without one trial and must never be shipped.')
        return model

    @classmethod
    def _load_bundle(cls, bundle_dir):
        """Read a v3 directory bundle."""
        meta = bundle_dir / METADATA_NAME
        if not meta.is_file():
            raise FileNotFoundError(f'{bundle_dir}: no {METADATA_NAME}')
        data = np.load(meta, allow_pickle=False)
        version = int(data['format_version']) if 'format_version' in data else 0
        if version != BUNDLE_FORMAT:
            raise ValueError(
                f'{meta}: format_version {version}, expected {BUNDLE_FORMAT}')
        return cls(
            class_names=data['class_names'], centroids=data['centroids'],
            encoder_type=_text(data, 'encoder_type'),
            encoder_config=_json_field(data, 'encoder_config', {}),
            weights_source=_text(data, 'weights_source'),
            checkpoint_id=_text(data, 'checkpoint_id'),
            checkpoint_revision=_text(data, 'checkpoint_revision'),
            weights_file=_text(data, 'weights_file'),
            weights_sha256=_text(data, 'weights_sha256'),
            weights_format=_text(data, 'weights_format'),
            encoder_seed=int(data['encoder_seed']) if 'encoder_seed' in data else -1,
            preprocessing_id=_text(data, 'preprocessing_id'),
            expected_n_frames=int(data['expected_n_frames']),
            expected_n_mels=int(data['expected_n_mels']),
            window_duration_sec=float(data['window_duration_sec']),
            classifier_config=_json_field(data, 'classifier_config',
                                          DEFAULT_CLASSIFIER_CONFIG),
            fold_id=_text(data, 'fold_id', 'all'),
            trained_on=_json_field(data, 'trained_on', {}),
            dataset_digest=_text(data, 'dataset_digest'),
            embedding_dim=int(data['embedding_dim']),
            bundle_dir=bundle_dir)

    @classmethod
    def _load_pre_v3(cls, path):
        """Read a v2 tagged .npz or an untagged legacy one, flagged for migration."""
        data = np.load(path, allow_pickle=True)
        if 'centroids' not in data or 'class_names' not in data:
            raise ValueError(f"{path}: expected 'class_names' and 'centroids' arrays")
        version = int(data['format_version']) if 'format_version' in data else 0
        if version > BUNDLE_FORMAT:
            raise ValueError(f'{path}: format_version {version} is from the future')
        return cls(
            class_names=data['class_names'], centroids=data['centroids'],
            encoder_type=_text(data, 'encoder_type', 'ast'),
            encoder_config={}, weights_source='checkpoint',
            checkpoint_id=_text(data, 'encoder_id', 'unknown-pre-v3'),
            checkpoint_revision='',
            preprocessing_id=_text(data, 'preprocessing_id', 'unknown-pre-v3'),
            expected_n_frames=int(data['expected_n_frames'])
            if 'expected_n_frames' in data else -1,
            expected_n_mels=int(data['expected_n_mels'])
            if 'expected_n_mels' in data else -1,
            window_duration_sec=float(data['window_duration_sec'])
            if 'window_duration_sec' in data else -1.0,
            embedding_dim=int(data['centroids'].shape[1]),
            needs_migration=True)

    def save(self, bundle_dir, weights_bytes=None, allow_fold=False):
        """
        Write the bundle directory, including weights when they are bundled.

        Layer one of fold isolation: a cross-validation artifact is refused
        unless the caller says it means it, and is refused outright anywhere
        under a models/ tree, which is what gets installed.
        """
        bundle_dir = Path(bundle_dir)
        if self.fold_id != 'all':
            if not allow_fold:
                raise ValueError(
                    f'refusing to write a fold artifact (fold_id={self.fold_id!r}) '
                    f'without allow_fold=True: it is trained without one trial and '
                    f'is an evaluation output, not a model')
            resolved = bundle_dir.resolve()
            if any(parent.name == 'models' for parent in resolved.parents):
                raise ValueError(
                    f'refusing to write fold artifact {self.fold_id!r} under a '
                    f'models/ tree ({resolved}): only the final all-data model is '
                    f'installed. Write cross-validation outputs to eval_runs/.')
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if self.weights_source == 'bundled':
            if weights_bytes is None:
                raise ValueError('bundled weights require weights_bytes')
            (bundle_dir / self.weights_file).write_bytes(weights_bytes)
            written = sha256_file(bundle_dir / self.weights_file)
            if written != self.weights_sha256:
                raise ValueError(
                    f'weights digest mismatch after write: {written} != '
                    f'{self.weights_sha256}')
        elif weights_bytes is not None:
            raise ValueError(
                f'weights_bytes given but weights_source is {self.weights_source!r}')

        np.savez(
            bundle_dir / METADATA_NAME,
            format_version=np.array(BUNDLE_FORMAT),
            encoder_type=np.array(self.encoder_type),
            encoder_config=np.array(json.dumps(self.encoder_config, sort_keys=True)),
            weights_source=np.array(self.weights_source),
            checkpoint_id=np.array(self.checkpoint_id),
            checkpoint_revision=np.array(self.checkpoint_revision),
            weights_file=np.array(self.weights_file),
            weights_sha256=np.array(self.weights_sha256),
            weights_format=np.array(self.weights_format),
            encoder_seed=np.array(self.encoder_seed),
            embedding_dim=np.array(self.embedding_dim),
            preprocessing_id=np.array(self.preprocessing_id),
            expected_n_frames=np.array(self.expected_n_frames),
            expected_n_mels=np.array(self.expected_n_mels),
            window_duration_sec=np.array(self.window_duration_sec),
            classifier_config=np.array(
                json.dumps(self.classifier_config, sort_keys=True)),
            fold_id=np.array(self.fold_id),
            trained_on=np.array(json.dumps(self.trained_on, sort_keys=True)),
            dataset_digest=np.array(self.dataset_digest),
            class_names=np.array(self.class_names),
            centroids=self.centroids.astype(np.float32),
        )
        self.bundle_dir = bundle_dir
        return bundle_dir

    # ---- weights --------------------------------------------------------

    def load_weights(self):
        """Verify the digest, then load the bundled tensors via safetensors."""
        if self.weights_source != 'bundled':
            return None
        if self.bundle_dir is None:
            raise ValueError('model was not loaded from a bundle directory')
        path = safe_bundle_file(self.bundle_dir, self.weights_file)
        actual = sha256_file(path)
        if actual != self.weights_sha256:
            raise ValueError(
                f'{path}: SHA-256 {actual} does not match the {self.weights_sha256} '
                f'recorded in {METADATA_NAME}. The weights and the centroids are '
                f'not the pair this model was built from.')
        from safetensors.torch import load_file
        return load_file(str(path))

    # ---- compatibility --------------------------------------------------

    def classifier(self, unknown_threshold=None, temperature=None):
        """Build the classifier, defaulting to the settings stored in the bundle."""
        config = self.classifier_config
        if config.get('type', 'cosine_centroid') != 'cosine_centroid':
            raise ValueError(f"unsupported classifier type {config.get('type')!r}")
        return CentroidClassifier(
            self.class_names, self.centroids,
            config['unknown_threshold'] if unknown_threshold is None
            else unknown_threshold,
            config['temperature'] if temperature is None else temperature)

    def incompatibility(self, preprocessing_id=None, n_frames=None, n_mels=None,
                        encoder_type=None):
        """Return why the given input does not fit this model, or None if it does."""
        if self.needs_migration:
            return 'model predates the identity format and declares no contract'
        if encoder_type is not None and encoder_type != self.encoder_type:
            return (f'encoder_type {encoder_type!r} but the model was trained with '
                    f'{self.encoder_type!r}')
        if preprocessing_id is not None:
            if not preprocessing_id:
                return ('message carries no preprocessing_id, so its features cannot '
                        f'be checked against the model ({self.preprocessing_id})')
            if preprocessing_id != self.preprocessing_id:
                return (f'preprocessing_id {preprocessing_id!r} but the model was '
                        f'trained on {self.preprocessing_id!r}')
        if n_mels is not None and n_mels != self.expected_n_mels:
            return f'{n_mels} mel bins but the model expects {self.expected_n_mels}'
        if n_frames is not None and n_frames != self.expected_n_frames:
            return (f'{n_frames} frames but the model expects {self.expected_n_frames} '
                    f'({self.window_duration_sec:g} s buffers)')
        return None

    def encoder_mismatch(self, encoder_type, encoder_id, embedding_dim):
        """
        Return why the given encoder cannot produce this model's space, or None.

        encoder_id is the check that matters: it fingerprints the architecture
        config and weight provenance together, so two checkpoints -- or the same
        weights read under a different config -- are distinguishable even though
        they share a type and an output dimension.
        """
        if self.needs_migration:
            return 'model predates the identity format; its encoder cannot be checked'
        if encoder_type != self.encoder_type:
            return (f'encoder_type {encoder_type!r} but the model was trained with '
                    f'{self.encoder_type!r}')
        if encoder_id != self.encoder_id:
            return (f'encoder_id {encoder_id!r} but the model was trained with '
                    f'{self.encoder_id!r}. Same family, different embedding space.')
        if int(embedding_dim) != self.embedding_dim:
            return (f'encoder produces {embedding_dim}-D embeddings but the model '
                    f'expects {self.embedding_dim}-D')
        return None

    def describe(self):
        """Return a one-line summary for startup logs."""
        weights = self.weights_source
        if weights == 'checkpoint':
            weights = f'{self.checkpoint_id}@{self.checkpoint_revision[:12]}'
        elif weights == 'bundled':
            weights = f'{self.weights_file} sha {self.weights_sha256[:12]}'
        return (f'{self.encoder_id} dim={self.embedding_dim} classes='
                f'{self.class_names} expects {self.expected_n_frames}x'
                f'{self.expected_n_mels} ({self.window_duration_sec:g} s) '
                f'features={self.preprocessing_id} weights={weights} '
                f'fold={self.fold_id} data={self.dataset_digest or "unrecorded"}')


def bundle_paths(models_root):
    """List every bundle directory under a models root."""
    root = Path(models_root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / METADATA_NAME).is_file())


def refuse_fold_artifact_in_models(bundle_dir, fold_id, models_root):
    """Raise if a fold artifact is about to be written into the install tree."""
    if fold_id == 'all':
        return
    bundle = Path(bundle_dir).resolve()
    root = Path(models_root).resolve()
    if root == bundle or root in bundle.parents:
        raise ValueError(
            f'refusing to write fold artifact (fold_id={fold_id!r}) under '
            f'{root}: only the final all-data model is installed with the '
            f'package. Write cross-validation outputs to eval_runs/ instead.')
