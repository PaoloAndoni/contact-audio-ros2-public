"""Unit tests for the ROS-free filterbank front end."""
from contact_audio_perception.spectrogram import (
    ANALYSIS_SAMPLE_RATE,
    buffer_to_fbank,
    compute_fbank,
    downmix_to_mono,
    expected_frame_count,
    FbankConfig,
    peak_normalise,
)
import numpy as np
import pytest


def _tone(n_frames, channels=1, sample_rate=48000, freq=440.0, amplitude=8000.0):
    """Build a deterministic interleaved int16 buffer."""
    t = np.arange(n_frames, dtype=np.float32) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * freq * t)
    if channels == 1:
        return mono.astype(np.int16)
    # Give the channels different content so a down-mix bug cannot pass silently.
    stacked = np.stack([mono * (1.0 - 0.3 * c) for c in range(channels)], axis=1)
    return stacked.astype(np.int16).reshape(-1)


def test_one_second_is_98_by_128():
    """A 1 s buffer at 16 kHz yields 98 time frames by 128 mel bins."""
    audio = np.sin(np.arange(ANALYSIS_SAMPLE_RATE, dtype=np.float32) * 0.01)
    fbank = compute_fbank(audio)
    assert fbank.shape == (98, 128)
    assert expected_frame_count(ANALYSIS_SAMPLE_RATE) == 98


def test_fbank_is_time_major():
    """Doubling the buffer length must grow axis 0, never axis 1."""
    short = compute_fbank(np.sin(np.arange(ANALYSIS_SAMPLE_RATE, dtype=np.float32) * 0.01))
    long = compute_fbank(np.sin(np.arange(2 * ANALYSIS_SAMPLE_RATE, dtype=np.float32) * 0.01))
    assert long.shape[1] == short.shape[1] == 128
    assert long.shape[0] > short.shape[0]


def test_buffer_to_fbank_resamples_and_reports_gain():
    """A 48 kHz stereo buffer produces the same shape as a 16 kHz mono one."""
    buf = _tone(48000, channels=2, sample_rate=48000)
    fbank, gain = buffer_to_fbank(buf, 2, 48000)
    assert fbank.shape == (98, 128)
    assert fbank.dtype == np.float32
    assert 0.0 < gain <= 1.0
    assert np.isfinite(fbank).all()


def test_peak_gain_recovers_absolute_amplitude():
    """Multiplying by the reported gain restores the original scale."""
    audio = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    normalised, gain = peak_normalise(audio)
    assert normalised.max() <= 1.0
    assert np.allclose(normalised * gain, audio, atol=1e-6)


def test_legacy_downmix_truncates_to_int16():
    """The legacy path truncates the channel mean, the modern one does not."""
    interleaved = np.array([1, 2, 3, 4], dtype=np.int16)  # frames (1,2) and (3,4)
    legacy = downmix_to_mono(interleaved, 2, legacy_int16=True)
    modern = downmix_to_mono(interleaved, 2, legacy_int16=False)
    assert legacy.dtype == np.int16
    assert legacy.tolist() == [1, 3]      # 1.5 and 3.5 truncated toward zero
    assert modern.tolist() == [1.5, 3.5]


def test_matches_legacy_preprocessing_chain():
    """buffer_to_fbank must reproduce audio_window_node + window_to_model_input."""
    from contact_audio_perception.ast_encoder import window_to_model_input

    buf = _tone(48000, channels=2, sample_rate=48000)
    mine, _ = buffer_to_fbank(buf, 2, 48000)

    # Exactly what the four-node pipeline did: int16 down-mix in the window node,
    # then the encoder's own preprocessing, then AST's filterbank.
    legacy_mono = downmix_to_mono(buf, 2, legacy_int16=True)
    legacy_audio = window_to_model_input(legacy_mono, 1, 48000)
    legacy = compute_fbank(legacy_audio)

    assert mine.shape == legacy.shape
    assert np.max(np.abs(mine - legacy)) < 1e-5


def test_preprocessing_id_is_stable_and_discriminating():
    """The digest is deterministic but changes when any feature setting changes."""
    a = FbankConfig()
    b = FbankConfig()
    assert a.preprocessing_id == b.preprocessing_id
    assert a.preprocessing_id.startswith('kaldi_fbank_v1-')
    assert FbankConfig(n_mels=64).preprocessing_id != a.preprocessing_id
    assert FbankConfig(frame_shift_ms=20.0).preprocessing_id != a.preprocessing_id
    assert FbankConfig(peak_normalise=False).preprocessing_id != a.preprocessing_id


def test_frames_for_matches_actual_output():
    """The advertised frame count matches what compute_fbank really returns."""
    config = FbankConfig()
    for duration in (0.5, 1.0, 2.0):
        audio = np.sin(np.arange(int(duration * ANALYSIS_SAMPLE_RATE), dtype=np.float32) * 0.01)
        assert compute_fbank(audio).shape[0] == config.frames_for(duration)


def test_rejects_invalid_config():
    """Nonsense feature settings fail at construction, not at 15 Hz."""
    with pytest.raises(ValueError):
        FbankConfig(n_mels=0)
    with pytest.raises(ValueError):
        FbankConfig(frame_shift_ms=0.0)
    with pytest.raises(ValueError):
        FbankConfig(analysis_sample_rate=0)
