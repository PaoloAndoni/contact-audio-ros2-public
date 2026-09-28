"""Abstract encoder interface shared by every feature extractor."""
import numpy as np

# Names build_encoder() knows about. Three encoder conditions plus one
# non-neural baseline -- not four equivalent trained encoders:
#   ast         pretrained neural encoder    (accuracy baseline)
#   ast_random  randomly initialised control (pretraining control, eval only)
#   cnn         locally trained encoder      (low-latency candidate)
#   melstats    handcrafted statistics       (non-neural baseline)
ENCODER_TYPES = ('ast', 'ast_random', 'cnn', 'melstats')


class Encoder:
    """Turn a (n_frames, n_mels) filterbank into a fixed-length embedding."""

    encoder_type = 'abstract'
    embedding_dim = 0

    @property
    def encoder_id(self):
        """Return a string identifying the exact weights or definition used."""
        raise NotImplementedError

    @property
    def required_preprocessing_id(self):
        """Return the preprocessing_id this encoder needs, or None for any."""
        return None

    def prepare_input(self, fbank):
        """Turn a validated filterbank into whatever the model consumes."""
        raise NotImplementedError

    def embed(self, fbank):
        """Return a 1-D float32 embedding for one filterbank."""
        raise NotImplementedError

    def validate_fbank(self, fbank, expected_n_mels=None):
        """Raise ValueError unless fbank is a finite 2-D (frames, mels) array."""
        fbank = np.asarray(fbank, dtype=np.float32)
        if fbank.ndim != 2:
            raise ValueError(
                f'expected a 2-D (n_frames, n_mels) filterbank, got shape {fbank.shape}')
        n_frames, n_mels = fbank.shape
        if n_frames < 1 or n_mels < 1:
            raise ValueError(f'filterbank must be non-empty, got shape {fbank.shape}')
        if expected_n_mels is not None and n_mels != expected_n_mels:
            raise ValueError(
                f'expected {expected_n_mels} mel bins, got {n_mels}. A transposed '
                f'time-major/mel-major mix-up is the usual cause.')
        if not np.isfinite(fbank).all():
            raise ValueError('filterbank contains NaN or infinite values')
        return fbank
