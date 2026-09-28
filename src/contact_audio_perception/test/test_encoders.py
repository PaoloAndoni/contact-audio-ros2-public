"""Tests for the extractors that do not need an 87M-parameter checkpoint."""
from pathlib import Path
import tempfile

from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders import (
    build_encoder,
    build_encoder_for_model,
    ENCODER_TYPES,
)
from contact_audio_perception.identity import sha256_hex
import numpy as np
import pytest


def _fbank(seed=0):
    """Deterministic (98, 128) filterbank-shaped array."""
    rng = np.random.RandomState(seed)
    return (rng.randn(98, 128).astype(np.float32) * 2.0) - 4.0


def _training_set():
    """Build a tiny two-class set with a separable offset."""
    fbanks, labels = [], []
    for i in range(8):
        fbanks.append(_fbank(i))
        labels.append(0)
        fbanks.append(_fbank(100 + i) + 3.0)
        labels.append(1)
    return fbanks, labels


# ---- registry ----

def test_every_declared_encoder_type_is_buildable_by_name():
    """ENCODER_TYPES and build_encoder do not drift apart."""
    assert set(ENCODER_TYPES) == {'ast', 'ast_random', 'cnn', 'melstats'}
    with pytest.raises(ValueError, match='unknown encoder_type'):
        build_encoder('does_not_exist')


# ---- melstats ----

def test_melstats_shape_and_determinism():
    """One statistic per mel bin, and the same input always gives the same vector."""
    encoder = build_encoder('melstats')
    out = encoder.embed(_fbank())
    assert out.shape == (5 * 128,)
    assert out.dtype == np.float32
    assert np.array_equal(out, encoder.embed(_fbank()))
    assert encoder.embedding_dim == 640


def test_melstats_statistics_are_actually_computed():
    """Mean and percentile entries match numpy on the same axis."""
    encoder = build_encoder('melstats', stats=('mean', 'p50'))
    data = _fbank()
    out = encoder.embed(data)
    assert np.allclose(out[:128], data.mean(axis=0), atol=1e-5)
    assert np.allclose(out[128:], np.percentile(data, 50, axis=0), atol=1e-5)


def test_melstats_identity_tracks_its_statistic_list():
    """Changing the statistics changes the embedding space and so the id."""
    a = build_encoder('melstats', stats=('mean', 'std'))
    b = build_encoder('melstats', stats=('mean', 'p90'))
    assert a.encoder_id != b.encoder_id
    assert a.encoder_id.startswith('melstats:none:')


def test_melstats_rejects_an_unknown_statistic():
    """A typo must fail at construction, not silently drop a feature."""
    with pytest.raises(ValueError, match='unknown statistics'):
        build_encoder('melstats', stats=('mean', 'medain'))


# ---- cnn ----

def test_cnn_embedding_shape_and_eval_determinism():
    """The CNN returns a fixed-width embedding and is deterministic in eval mode."""
    encoder = build_encoder('cnn', n_classes=3)
    out = encoder.embed(_fbank())
    assert out.shape == (128,)
    assert np.array_equal(out, encoder.embed(_fbank()))


def test_cnn_training_reduces_loss_and_changes_the_embedding():
    """Fitting actually learns: loss drops and the representation moves."""
    encoder = build_encoder('cnn', n_classes=2)
    before = encoder.embed(_fbank())
    loss = encoder.fit(*_training_set(), epochs=15)
    assert loss < 0.5
    assert not np.allclose(before, encoder.embed(_fbank()), atol=1e-4)


def test_cnn_weights_survive_a_bundle_round_trip():
    """A trained CNN reloaded from its bundle produces identical embeddings."""
    encoder = build_encoder('cnn', n_classes=2)
    encoder.fit(*_training_set(), epochs=5)
    payload = encoder.state_bytes()
    encoder.weights_sha256 = sha256_hex(payload)
    expected = encoder.embed(_fbank())

    model = ContactModel(
        class_names=['a', 'b'], centroids=np.eye(2, 128, dtype=np.float32),
        encoder_type='cnn', encoder_config=encoder.encoder_config,
        weights_source='bundled', weights_file='encoder.safetensors',
        weights_sha256=encoder.weights_sha256, weights_format='safetensors',
        preprocessing_id='kaldi_fbank_v1-aa', expected_n_frames=98,
        expected_n_mels=128, window_duration_sec=1.0)
    assert model.encoder_id == encoder.encoder_id

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'contact_model_cnn'
        model.save(bundle, weights_bytes=payload)
        loaded = ContactModel.load(bundle)
        rebuilt = build_encoder_for_model(loaded)
    assert rebuilt.encoder_id == encoder.encoder_id
    assert np.allclose(rebuilt.embed(_fbank()), expected, atol=1e-6)


def test_cnn_identity_changes_with_its_weights():
    """Two differently trained CNNs are different encoders."""
    a = build_encoder('cnn', n_classes=2)
    a.fit(*_training_set(), epochs=3)
    a.weights_sha256 = sha256_hex(a.state_bytes())
    b = build_encoder('cnn', n_classes=2)
    b.fit(*_training_set(), epochs=12)
    b.weights_sha256 = sha256_hex(b.state_bytes())
    assert a.encoder_id != b.encoder_id


def test_encoders_reject_a_transposed_filterbank():
    """A mel-major array must not be silently accepted by either extractor."""
    transposed = _fbank().T
    for encoder in (build_encoder('melstats'), build_encoder('cnn', n_classes=2)):
        with pytest.raises(ValueError, match='mel bins'):
            encoder.embed(transposed)
