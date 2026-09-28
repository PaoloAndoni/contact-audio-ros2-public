"""
Handcrafted mel-statistics baseline: no model, no training, no weights.

Summarises each mel bin over time with a handful of order statistics. This is
NOT an encoder in the learned sense -- it is the floor every learned method has
to beat, and it costs microseconds rather than seconds. Report it as a
non-neural baseline, never as a fourth trained encoder.

Its identity still matters: changing the statistic list changes the embedding
space just as surely as retraining would, so the config is fingerprinted.
"""
from contact_audio_perception.encoders.base import Encoder
from contact_audio_perception.identity import derive_encoder_id, identity_document
import numpy as np

DEFAULT_STATS = ('mean', 'std', 'p10', 'p50', 'p90')


class MelStatsEncoder(Encoder):
    """Summarise a filterbank into per-mel-bin order statistics."""

    encoder_type = 'melstats'

    def __init__(self, n_mels=128, stats=DEFAULT_STATS, config=None):
        """Fix the statistic list and the expected number of mel bins."""
        self.n_mels = int(n_mels)
        self.stats = tuple(stats)
        unknown = [s for s in self.stats
                   if s not in ('mean', 'std', 'min', 'max') and not (
                       s.startswith('p') and s[1:].isdigit())]
        if unknown:
            raise ValueError(f'unknown statistics: {unknown}')
        if not self.stats:
            raise ValueError('at least one statistic is required')
        self.embedding_dim = len(self.stats) * self.n_mels
        self._feature_config = config

    @property
    def encoder_config(self):
        """Return the configuration that fixes this embedding space."""
        return {'n_mels': self.n_mels, 'stats': list(self.stats)}

    @property
    def identity(self):
        """Return this encoder's identity document."""
        return identity_document(self.encoder_type, self.encoder_config, 'none')

    @property
    def encoder_id(self):
        """Return the namespaced fingerprint of this encoder."""
        return derive_encoder_id(self.identity)

    @property
    def required_preprocessing_id(self):
        """Return the feature configuration this encoder was built against."""
        return self._feature_config.preprocessing_id if self._feature_config else None

    def prepare_input(self, fbank):
        """Validate and return the filterbank unchanged; there is no model input."""
        return self.validate_fbank(fbank, expected_n_mels=self.n_mels)

    def embed(self, fbank):
        """Return the concatenated per-mel-bin statistics as one vector."""
        data = self.prepare_input(fbank)
        parts = []
        for stat in self.stats:
            if stat == 'mean':
                parts.append(data.mean(axis=0))
            elif stat == 'std':
                parts.append(data.std(axis=0))
            elif stat == 'min':
                parts.append(data.min(axis=0))
            elif stat == 'max':
                parts.append(data.max(axis=0))
            else:
                parts.append(np.percentile(data, float(stat[1:]), axis=0))
        return np.concatenate(parts).astype(np.float32)
