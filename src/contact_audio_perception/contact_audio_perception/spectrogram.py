"""
Mel filterbank features for the contact-audio pipeline.

Kept ROS-free so it can be unit-tested, reused by training scripts, and compared
against the legacy pipeline. Produces exactly the features that
transformers' ASTFeatureExtractor computes internally, minus its padding and
normalisation -- those belong to the encoder, not to data acquisition.

The full chain reproduced here is the one the four-node pipeline performed
across audio_window_node + ast_encoder.window_to_model_input:

    int16 interleaved
      -> mono downmix                (audio_window_node._to_out_frames)
      -> float32 / 32768             (window_to_model_input)
      -> peak normalisation          (window_to_model_input)
      -> resample to 16 kHz          (window_to_model_input)
      -> kaldi fbank                 (ASTFeatureExtractor._extract_fbank_features)

Measured against transformers 5.13.1 and MIT/ast-finetuned-audioset-10-10-0.4593:
a 1.0 s buffer at 16 kHz with 25/10 ms framing yields (98, 128) -- 98 TIME frames
by 128 mel bins, time-major. AST then pads that to 1024 frames and normalises,
so 90.4% of what the model sees is padding.
"""
import hashlib
import wave

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

ANALYSIS_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 128
DEFAULT_FRAME_LENGTH_MS = 25.0
DEFAULT_FRAME_SHIFT_MS = 10.0

# Version marker, not just a name: it is part of preprocessing_id, so bumping it
# invalidates every model trained under the previous definition. Bump whenever
# the meaning of these features changes.
FEATURE_TYPE = 'kaldi_fbank_v1'

# ASTFeatureExtractor calls ta_kaldi.fbank with only sample_frequency,
# window_type and num_mel_bins, inheriting 22 further options from torchaudio's
# defaults. Those defaults are not a contract: a torchaudio upgrade could change
# one and silently move the features underneath a trained model. Pinning the
# full set here -- and hashing it into preprocessing_id -- turns that class of
# change into a loud mismatch instead of a silent one.
#
# Values below are torchaudio 2.11.0's defaults, with AST's three overrides
# applied. test_explicit_options_match_ast_defaults guards the equivalence.
KALDI_V1_OPTIONS = {
    'blackman_coeff': 0.42,
    'channel': -1,
    'dither': 0.0,
    'energy_floor': 1.0,
    'frame_length': DEFAULT_FRAME_LENGTH_MS,
    'frame_shift': DEFAULT_FRAME_SHIFT_MS,
    'high_freq': 0.0,
    'htk_compat': False,
    'low_freq': 20.0,
    'min_duration': 0.0,
    'num_mel_bins': DEFAULT_N_MELS,       # AST override (torchaudio default 23)
    'preemphasis_coefficient': 0.97,
    'raw_energy': True,
    'remove_dc_offset': True,
    'round_to_power_of_two': True,
    'sample_frequency': float(ANALYSIS_SAMPLE_RATE),
    'snip_edges': True,
    'subtract_mean': False,
    'use_energy': False,
    'use_log_fbank': True,
    'use_power': True,
    'vtln_high': -500.0,
    'vtln_low': 100.0,
    'vtln_warp': 1.0,
    'window_type': 'hanning',             # AST override (torchaudio default povey)
}


def read_wav_int16(path):
    """
    Read a 16-bit PCM WAV, returning (interleaved samples, channels, rate).

    Lives here so scripts stop growing private copies of it.
    """
    with wave.open(str(path), 'rb') as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(
                f'{path}: only 16-bit PCM is supported '
                f'(got {handle.getsampwidth() * 8}-bit)')
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.int16), channels, sample_rate


def expected_frame_count(n_samples, sample_rate=ANALYSIS_SAMPLE_RATE,
                         frame_length_ms=DEFAULT_FRAME_LENGTH_MS,
                         frame_shift_ms=DEFAULT_FRAME_SHIFT_MS):
    """Return the number of fbank frames kaldi produces for n_samples."""
    win = int(round(sample_rate * frame_length_ms / 1000.0))
    hop = int(round(sample_rate * frame_shift_ms / 1000.0))
    if n_samples < win:
        return 0
    return 1 + (n_samples - win) // hop


def downmix_to_mono(samples, channels, legacy_int16=True):
    """
    Down-mix interleaved int16 samples to a single channel.

    legacy_int16 reproduces audio_window_node, which averaged in float and then
    truncated back to int16 before anything else touched the data. That
    truncation is worth up to half an LSB, which is ~1.5e-5 after scaling --
    far above the 1e-6 tolerance the parity gate uses -- so it must be kept
    until the legacy pipeline is retired, even though averaging in float is
    the better behaviour. Flip the flag once parity no longer matters.
    """
    samples = np.asarray(samples)
    if channels <= 1:
        return samples.astype(np.int16, copy=False) if legacy_int16 else \
            samples.astype(np.float32)
    frames = samples.reshape(-1, channels)
    if legacy_int16:
        return frames.astype(np.float32).mean(axis=1).astype(np.int16)
    return frames.astype(np.float32).mean(axis=1)


def peak_normalise(audio):
    """Scale to unit peak; return (audio, gain) where gain is the divisor used."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    gain = peak + 1e-6
    return audio / gain, gain


def resample(audio, sample_rate, target_rate=ANALYSIS_SAMPLE_RATE):
    """Resample a 1-D float32 array (no-op when already at target_rate)."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if sample_rate == target_rate:
        return audio
    tensor = torch.from_numpy(audio)
    return torchaudio.functional.resample(tensor, sample_rate, target_rate).numpy()


def compute_fbank(audio, options=None):
    """Compute a time-major (n_frames, n_mels) kaldi filterbank."""
    options = dict(KALDI_V1_OPTIONS if options is None else options)
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    fbank = ta_kaldi.fbank(waveform.unsqueeze(0), **options)
    return fbank.numpy().astype(np.float32)


class FbankConfig:
    """Feature configuration plus the digest that identifies it on the wire."""

    def __init__(self, analysis_sample_rate=ANALYSIS_SAMPLE_RATE, n_mels=DEFAULT_N_MELS,
                 frame_length_ms=DEFAULT_FRAME_LENGTH_MS,
                 frame_shift_ms=DEFAULT_FRAME_SHIFT_MS, peak_normalise=True,
                 legacy_int16_downmix=True, options=None):
        """Store the configuration and validate it."""
        self.options = dict(KALDI_V1_OPTIONS)
        if options:
            unknown = set(options) - set(KALDI_V1_OPTIONS)
            if unknown:
                raise ValueError(f'unknown kaldi options: {sorted(unknown)}')
            self.options.update(options)
        # The named arguments are conveniences that write into the option set,
        # so there is exactly one source of truth for what gets hashed.
        self.options['sample_frequency'] = float(analysis_sample_rate)
        self.options['num_mel_bins'] = int(n_mels)
        self.options['frame_length'] = float(frame_length_ms)
        self.options['frame_shift'] = float(frame_shift_ms)
        self.peak_normalise = bool(peak_normalise)
        self.legacy_int16_downmix = bool(legacy_int16_downmix)

        if self.analysis_sample_rate <= 0:
            raise ValueError('analysis_sample_rate must be positive')
        if self.n_mels <= 0:
            raise ValueError('n_mels must be positive')
        if self.frame_length_ms <= 0.0 or self.frame_shift_ms <= 0.0:
            raise ValueError('frame_length_ms and frame_shift_ms must be positive')

    @property
    def analysis_sample_rate(self):
        """Return the rate the filterbank is computed at."""
        return int(self.options['sample_frequency'])

    @property
    def n_mels(self):
        """Return the number of mel bins."""
        return int(self.options['num_mel_bins'])

    @property
    def frame_length_ms(self):
        """Return the filterbank window length in milliseconds."""
        return float(self.options['frame_length'])

    @property
    def frame_shift_ms(self):
        """Return the filterbank step in milliseconds."""
        return float(self.options['frame_shift'])

    @property
    def preprocessing_id(self):
        """Return a short digest of every field that changes the features."""
        parts = [FEATURE_TYPE,
                 f'peak={int(self.peak_normalise)}',
                 f'legacy={int(self.legacy_int16_downmix)}']
        parts += [f'{key}={self.options[key]!r}' for key in sorted(self.options)]
        digest = hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()
        return f'{FEATURE_TYPE}-{digest[:12]}'

    def frames_for(self, duration_sec):
        """Return the frame count a buffer of duration_sec will produce."""
        n_samples = int(round(duration_sec * self.analysis_sample_rate))
        return expected_frame_count(n_samples, self.analysis_sample_rate,
                                    self.frame_length_ms, self.frame_shift_ms)

    def describe(self):
        """Return a one-line human-readable summary for startup logs."""
        return (f'{FEATURE_TYPE} {self.n_mels} mels @ {self.analysis_sample_rate} Hz, '
                f'{self.frame_length_ms:g}/{self.frame_shift_ms:g} ms, '
                f'peak_norm={self.peak_normalise}, id={self.preprocessing_id}')


def buffer_to_fbank(samples, channels, source_sample_rate, config=None):
    """
    Turn one interleaved int16 buffer into (fbank, peak_gain).

    fbank is time-major (n_frames, n_mels), unpadded and unnormalised.
    peak_gain is the divisor peak normalisation applied, so callers can recover
    absolute amplitude; it is 1.0 when peak normalisation is disabled.
    """
    config = config or FbankConfig()
    mono = downmix_to_mono(samples, channels, config.legacy_int16_downmix)
    audio = np.asarray(mono, dtype=np.float32) / 32768.0
    if config.peak_normalise:
        audio, gain = peak_normalise(audio)
    else:
        gain = 1.0
    audio = resample(audio, source_sample_rate, config.analysis_sample_rate)
    fbank = compute_fbank(audio, config.options)
    return fbank, gain
