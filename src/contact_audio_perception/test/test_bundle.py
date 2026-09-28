"""Tests for encoder identity, bundle safety, fold isolation and packaging."""
import json
import os
from pathlib import Path
import tempfile

from ament_index_python.packages import get_package_share_directory
from contact_audio_perception.contact_model import (
    BUNDLE_FORMAT,
    bundle_paths,
    ContactModel,
    refuse_fold_artifact_in_models,
)
from contact_audio_perception.identity import (
    canonical_json,
    derive_encoder_id,
    identity_document,
    METADATA_NAME,
    safe_bundle_file,
    sha256_file,
    sha256_hex,
    validate_weight_fields,
)
import numpy as np
import pytest


def _kwargs(**overrides):
    """Build valid ContactModel arguments."""
    base = {
        'class_names': ['a', 'b'], 'centroids': np.eye(2, 4, dtype=np.float32),
        'encoder_type': 'ast', 'encoder_config': {'pooling': 'cls'},
        'weights_source': 'checkpoint', 'checkpoint_id': 'org/model',
        'checkpoint_revision': 'abc123def456', 'preprocessing_id': 'kaldi_fbank_v1-aa',
        'expected_n_frames': 98, 'expected_n_mels': 128, 'window_duration_sec': 1.0,
    }
    base.update(overrides)
    return base


# ---- canonical identity ----

def test_canonical_json_is_stable_across_key_order():
    """Two dicts differing only in insertion order hash identically."""
    a = canonical_json({'b': 1, 'a': {'d': 2, 'c': 3}})
    b = canonical_json({'a': {'c': 3, 'd': 2}, 'b': 1})
    assert a == b
    assert b' ' not in a          # compact separators
    assert json.loads(a.decode('utf-8'))['b'] == 1


def test_canonical_json_refuses_nan():
    """Refuse NaN, which has no canonical JSON form and must not serialise."""
    with pytest.raises(ValueError):
        canonical_json({'x': float('nan')})


def test_config_changes_the_encoder_id_even_with_identical_weights():
    """The same weights read under a different config are a different encoder."""
    base = identity_document('cnn', {'pooling': 'mean'}, 'bundled',
                             weights_sha256='deadbeef')
    other = identity_document('cnn', {'pooling': 'max'}, 'bundled',
                              weights_sha256='deadbeef')
    assert derive_encoder_id(base) != derive_encoder_id(other)


def test_checkpoint_revision_changes_the_encoder_id():
    """A mutable checkpoint name repointed at new weights is a different encoder."""
    a = identity_document('ast', {}, 'checkpoint', 'org/m', 'rev-one')
    b = identity_document('ast', {}, 'checkpoint', 'org/m', 'rev-two')
    assert derive_encoder_id(a) != derive_encoder_id(b)


def test_encoder_id_is_namespaced():
    """The id names its type and weight source before the digest."""
    doc = identity_document('melstats', {'stats': ['mean']}, 'none')
    assert derive_encoder_id(doc).startswith('melstats:none:')


# ---- weight-field consistency ----

def test_weight_field_combinations_are_enforced():
    """Each weights_source permits exactly one shape of weight metadata."""
    validate_weight_fields('checkpoint', '', '', '', 'org/m', 'rev-one')
    validate_weight_fields('bundled', 'encoder.safetensors', 'aa', 'safetensors', '')
    validate_weight_fields('none', '', '', '', '')
    with pytest.raises(ValueError, match='must not carry'):
        validate_weight_fields('checkpoint', 'w.safetensors', 'aa', 'safetensors',
                               'x', 'rev-one')
    with pytest.raises(ValueError, match='requires a checkpoint_id'):
        validate_weight_fields('checkpoint', '', '', '', '', 'rev-one')
    with pytest.raises(ValueError, match='requires a checkpoint_revision'):
        validate_weight_fields('checkpoint', '', '', '', 'org/m', '')
    # Pre-v3 models predate the revision field, so migration tooling can still
    # read them; production loading refuses them for exactly that reason.
    validate_weight_fields('checkpoint', '', '', '', 'org/m', '',
                           require_revision=False)
    with pytest.raises(ValueError, match='requires both'):
        validate_weight_fields('bundled', 'w.safetensors', '', 'safetensors', '')
    with pytest.raises(ValueError, match='safetensors'):
        validate_weight_fields('bundled', 'w.bin', 'aa', 'pickle', '')
    with pytest.raises(ValueError, match='must not carry'):
        validate_weight_fields('none', 'w.safetensors', 'aa', 'safetensors', '')


# ---- bundle path safety ----

def test_safe_bundle_file_rejects_escapes():
    """Absolute paths, separators, '..' and escaping symlinks are all refused."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'bundle'
        bundle.mkdir()
        (bundle / 'encoder.safetensors').write_bytes(b'weights')
        outside = Path(tmp) / 'outside.safetensors'
        outside.write_bytes(b'not mine')

        assert safe_bundle_file(bundle, 'encoder.safetensors').is_file()
        for bad in ('/etc/passwd', '../outside.safetensors', 'sub/w.safetensors',
                    '..', '.', ''):
            with pytest.raises((ValueError, FileNotFoundError)):
                safe_bundle_file(bundle, bad)

        link = bundle / 'link.safetensors'
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip('symlinks unavailable')
        with pytest.raises(ValueError, match='symlink'):
            safe_bundle_file(bundle, 'link.safetensors')


# ---- bundle round trip ----

def test_bundle_round_trip_preserves_identity():
    """Saving and loading a bundle keeps every identity field and the encoder id."""
    model = ContactModel(**_kwargs())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'contact_model_ast'
        model.save(path)
        assert (path / METADATA_NAME).is_file()
        loaded = ContactModel.load(path)
    assert loaded.encoder_id == model.encoder_id
    assert loaded.checkpoint_revision == 'abc123def456'
    assert loaded.encoder_config == {'pooling': 'cls'}
    assert loaded.classifier_config['type'] == 'cosine_centroid'
    assert not loaded.needs_migration
    assert np.array_equal(loaded.centroids, model.centroids)


def test_bundled_weights_are_checksum_verified():
    """A tampered weight file is refused rather than loaded."""
    payload = b'\x00' * 64
    digest = sha256_hex(payload)
    model = ContactModel(**_kwargs(
        weights_source='bundled', checkpoint_id='', checkpoint_revision='',
        weights_file='encoder.safetensors', weights_sha256=digest,
        weights_format='safetensors'))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'contact_model_cnn'
        model.save(path, weights_bytes=payload)
        assert sha256_file(path / 'encoder.safetensors') == digest
        (path / 'encoder.safetensors').write_bytes(b'\x01' * 64)
        loaded = ContactModel.load(path)
        with pytest.raises(ValueError, match='does not match'):
            loaded.load_weights()


def test_real_safetensors_round_trip_through_a_bundle():
    """
    Save, verify, load and apply a genuine safetensors artifact.

    The tamper test proves a corrupted file is caught; this proves an intact one
    is actually parseable and loadable, which arbitrary bytes never exercise.
    """
    import torch
    from safetensors.torch import save_file

    module = torch.nn.Linear(4, 3)
    state = {k: v.contiguous() for k, v in module.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / 'staged.safetensors'
        save_file(state, str(raw))
        payload = raw.read_bytes()
        digest = sha256_hex(payload)

        model = ContactModel(**_kwargs(
            encoder_type='cnn', weights_source='bundled', checkpoint_id='',
            checkpoint_revision='', weights_file='encoder.safetensors',
            weights_sha256=digest, weights_format='safetensors'))
        bundle = Path(tmp) / 'contact_model_cnn'
        model.save(bundle, weights_bytes=payload)

        loaded = ContactModel.load(bundle)
        tensors = loaded.load_weights()
        assert set(tensors) == set(state)
        rebuilt = torch.nn.Linear(4, 3)
        rebuilt.load_state_dict(tensors, strict=True)
        assert torch.allclose(rebuilt.weight, module.weight)


def test_checkpoint_models_must_pin_a_revision():
    """A mutable model name alone is not an identity."""
    with pytest.raises(ValueError, match='checkpoint_revision'):
        ContactModel(**_kwargs(checkpoint_revision=''))


def test_pre_v3_model_is_refused_by_production_load():
    """A v2 file must not load silently; it is readable only for migration."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'old.npz'
        np.savez(path, format_version=np.array(2),
                 encoder_type=np.array('ast'), encoder_id=np.array('org/model'),
                 embedding_dim=np.array(4), preprocessing_id=np.array('kaldi_fbank_v1-aa'),
                 expected_n_frames=np.array(98), expected_n_mels=np.array(128),
                 window_duration_sec=np.array(1.0),
                 class_names=np.array(['a', 'b']),
                 centroids=np.eye(2, 4, dtype=np.float32))
        with pytest.raises(ValueError, match='pre-v3'):
            ContactModel.load(path)
        old = ContactModel.load(path, allow_needs_migration=True)
    assert old.needs_migration
    assert old.incompatibility() is not None
    assert old.encoder_mismatch('ast', 'anything', 4) is not None


# ---- fold isolation ----

def test_fold_artifact_is_not_loadable_as_a_deployable_model():
    """A cross-validation model must never be shipped by accident."""
    model = ContactModel(**_kwargs(fold_id='004'))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'fold_004'
        model.save(path, allow_fold=True)
        with pytest.raises(ValueError, match='cross-validation artifact'):
            ContactModel.load(path)
        assert ContactModel.load(path, allow_fold=True).fold_id == '004'


def test_saving_a_fold_artifact_needs_an_explicit_opt_in():
    """Layer one: save() refuses a fold model unless the caller means it."""
    model = ContactModel(**_kwargs(fold_id='004'))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match='without allow_fold'):
            model.save(Path(tmp) / 'fold_004')


def test_fold_artifacts_cannot_be_written_into_the_install_tree():
    """Even with allow_fold, a fold model may not be written under models/."""
    model = ContactModel(**_kwargs(fold_id='004'))
    final = ContactModel(**_kwargs(fold_id='all'))
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp) / 'models'
        models.mkdir()
        with pytest.raises(ValueError, match='under a models/ tree'):
            model.save(models / 'fold_004', allow_fold=True)
        final.save(models / 'contact_model_cnn')
        model.save(Path(tmp) / 'eval_runs' / 'fold_004', allow_fold=True)
        with pytest.raises(ValueError, match='refusing to write fold artifact'):
            refuse_fold_artifact_in_models(models / 'fold_004', '004', models)


# ---- packaging ----

def test_installed_bundles_are_complete():
    """
    Check every installed bundle keeps its metadata and its weights together.

    data_files cannot install a directory, so a bundle silently arriving without
    encoder.safetensors is a realistic packaging failure. Check the real install
    tree through the ament index rather than the source tree.
    """
    share = Path(get_package_share_directory('contact_audio_perception'))
    installed = bundle_paths(share / 'models')
    assert installed, f'no model bundles installed under {share / "models"}'

    # Compare against the SOURCE tree: checking only what is installed would
    # still pass if packaging dropped an entire bundle, since the surviving one
    # looks fine on its own.
    source = bundle_paths(Path('models'))
    assert [p.name for p in source] == [p.name for p in installed], (
        f'source bundles {[p.name for p in source]} != installed '
        f'{[p.name for p in installed]}')

    for path in installed:
        model = ContactModel.load(path)
        assert int(BUNDLE_FORMAT) == 3
        assert os.path.isfile(path / METADATA_NAME)
        assert model.fold_id == 'all', f'{path} is a fold artifact'
        if model.weights_source == 'bundled':
            weights = safe_bundle_file(path, model.weights_file)
            assert sha256_file(weights) == model.weights_sha256, path
            assert model.load_weights(), f'{path}: weights did not load'
