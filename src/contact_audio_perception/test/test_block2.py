"""Unit tests for the frozen feature contract, the encoder, and the model."""
from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders.ast_fbank import pad_and_normalise
from contact_audio_perception.encoders.base import Encoder
from contact_audio_perception.spectrogram import (
    ANALYSIS_SAMPLE_RATE,
    compute_fbank,
    FbankConfig,
    KALDI_V1_OPTIONS,
)
import numpy as np
import pytest

AST_MEAN = -4.2677393
AST_STD = 4.5689974


def _audio(seconds=1.0):
    """Deterministic test waveform at the analysis rate."""
    n = int(seconds * ANALYSIS_SAMPLE_RATE)
    return np.sin(np.arange(n, dtype=np.float32) * 0.01)


def _model(**overrides):
    """Build a small valid ContactModel."""
    kwargs = {
        'class_names': ['a', 'b'], 'centroids': np.eye(2, 4, dtype=np.float32),
        'encoder_type': 'ast', 'encoder_config': {}, 'weights_source': 'checkpoint',
        'checkpoint_id': 'org/model', 'checkpoint_revision': 'rev-one',
        'preprocessing_id': 'kaldi_fbank_v1-abc123', 'expected_n_frames': 98,
        'expected_n_mels': 128, 'window_duration_sec': 1.0,
    }
    kwargs.update(overrides)
    return ContactModel(**kwargs)


# ---- Gate 0: the frozen preprocessing contract ----

def test_explicit_options_match_ast_defaults():
    """Pinning all 25 kaldi options must not change the features AST computes."""
    import torch
    import torchaudio.compliance.kaldi as ta_kaldi

    audio = _audio()
    pinned = compute_fbank(audio, KALDI_V1_OPTIONS)
    # Exactly the call ASTFeatureExtractor makes: three arguments, 22 defaults.
    upstream = ta_kaldi.fbank(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0),
        sample_frequency=16000, window_type='hanning', num_mel_bins=128,
    ).numpy()
    assert pinned.shape == upstream.shape == (98, 128)
    assert np.array_equal(pinned, upstream)


def test_unknown_kaldi_option_is_rejected():
    """A typo in an option name must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match='unknown kaldi options'):
        FbankConfig(options={'num_mel_binz': 128})


def test_preprocessing_id_covers_every_kaldi_option():
    """Changing any pinned option must change the digest."""
    base = FbankConfig().preprocessing_id
    for key, value in (('dither', 0.1), ('low_freq', 50.0), ('snip_edges', False),
                       ('preemphasis_coefficient', 0.5), ('window_type', 'povey')):
        assert FbankConfig(options={key: value}).preprocessing_id != base, key


def test_preprocessing_id_carries_the_version_marker():
    """The digest is namespaced so a redefinition invalidates old models."""
    assert FbankConfig().preprocessing_id.startswith('kaldi_fbank_v1-')


# ---- Gate 1: padding and normalisation ----

def test_padding_happens_before_normalisation():
    """Padded rows must reach the model as 0.4670, not as zeros."""
    fbank = np.full((98, 128), -4.0, dtype=np.float32)
    out = pad_and_normalise(fbank, 1024, AST_MEAN, AST_STD)
    assert out.shape == (1024, 128)
    expected_pad = (0.0 - AST_MEAN) / (AST_STD * 2.0)
    assert np.allclose(out[98:], expected_pad, atol=1e-6)
    assert abs(expected_pad - 0.4670323625) < 1e-9
    assert not np.allclose(out[:98], expected_pad)


def test_longer_input_is_truncated_not_padded():
    """More frames than max_length are cut, matching upstream behaviour."""
    fbank = np.zeros((2000, 128), dtype=np.float32)
    assert pad_and_normalise(fbank, 1024, AST_MEAN, AST_STD).shape == (1024, 128)


def test_padding_fraction_is_90_percent():
    """A 1 s window really is mostly padding -- the number quoted everywhere."""
    assert abs((1024 - 98) / 1024 - 0.904) < 0.001


# ---- Encoder input validation ----

def test_encoder_rejects_malformed_filterbanks():
    """Wrong rank, wrong mel count, empty and non-finite input all raise."""
    encoder = Encoder()
    with pytest.raises(ValueError, match='2-D'):
        encoder.validate_fbank(np.zeros(98, dtype=np.float32))
    with pytest.raises(ValueError, match='mel bins'):
        encoder.validate_fbank(np.zeros((128, 98), dtype=np.float32), expected_n_mels=128)
    with pytest.raises(ValueError, match='non-empty'):
        encoder.validate_fbank(np.zeros((0, 128), dtype=np.float32))
    nan = np.zeros((98, 128), dtype=np.float32)
    nan[3, 3] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        encoder.validate_fbank(nan)


# ---- Gate 3: the model contract ----

def test_incompatibility_names_the_mismatch():
    """Each kind of mismatch is reported, and a matching input passes."""
    model = _model()
    good = {'preprocessing_id': 'kaldi_fbank_v1-abc123', 'n_frames': 98,
            'n_mels': 128, 'encoder_type': 'ast'}
    assert model.incompatibility(**good) is None
    assert 'preprocessing_id' in model.incompatibility(
        **{**good, 'preprocessing_id': 'kaldi_fbank_v1-other'})
    assert 'mel bins' in model.incompatibility(**{**good, 'n_mels': 64})
    assert 'frames' in model.incompatibility(**{**good, 'n_frames': 49})
    assert 'encoder_type' in model.incompatibility(**{**good, 'encoder_type': 'cnn'})


def test_empty_preprocessing_id_is_rejected():
    """A message with no id must fail the guard, not slip past it."""
    model = _model()
    reason = model.incompatibility(preprocessing_id='', n_frames=98, n_mels=128,
                                   encoder_type='ast')
    assert reason is not None and 'no preprocessing_id' in reason


def test_same_encoder_type_and_dim_but_different_checkpoint_is_rejected():
    """Two AST checkpoints share type and output width but not embedding space."""
    model = _model(checkpoint_revision='rev-one')
    other = _model(checkpoint_revision='rev-two')
    assert model.encoder_id != other.encoder_id
    assert model.encoder_mismatch('ast', model.encoder_id, 4) is None
    reason = model.encoder_mismatch('ast', other.encoder_id, 4)
    assert reason is not None and 'encoder_id' in reason
    assert 'different embedding space' in reason


def test_encoder_mismatch_still_catches_type_and_dimension():
    """The cheaper checks remain in place alongside the encoder_id check."""
    model = _model()
    assert 'encoder_type' in model.encoder_mismatch('cnn', model.encoder_id, 4)
    assert '768-D' in model.encoder_mismatch('ast', model.encoder_id, 768)


def test_window_duration_mismatch_is_caught_even_when_features_match():
    """A half-second buffer must be refused by a model trained on one second."""
    model = _model()
    reason = model.incompatibility(preprocessing_id='kaldi_fbank_v1-abc123',
                                   n_frames=48, n_mels=128, encoder_type='ast')
    assert reason is not None and '1 s buffers' in reason
