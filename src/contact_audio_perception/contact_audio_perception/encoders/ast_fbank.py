"""
AST encoder that consumes a precomputed filterbank.

The four-node pipeline handed AST a waveform and let ASTFeatureExtractor compute
the filterbank internally. Here the DAQ node has already computed it, so this
encoder performs only the two steps the extractor did afterwards:

    (n_frames, n_mels) unpadded, unnormalised
      -> right-pad (or truncate) to max_length rows with 0.0
      -> (x - mean) / (2 * std)
      -> ASTModel -> CLS token

Padding before normalisation is deliberate and matches upstream: the padded rows
therefore reach the model as a uniform (0 - mean) / (2 * std) = 0.4670, not as
zeros. Getting that order wrong changes 90.4% of the input tensor.

mean, std and max_length are read from the checkpoint's own feature-extractor
config rather than hardcoded, so they cannot drift from the weights.
"""
from contact_audio_perception.encoders.base import Encoder
from contact_audio_perception.identity import (
    derive_encoder_id,
    identity_document,
    resolve_hf_revision,
)
from contact_audio_perception.spectrogram import FbankConfig
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTModel

DEFAULT_MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'


def pad_and_normalise(fbank, max_length, mean, std):
    """
    Right-pad (or truncate) to max_length rows, then normalise.

    Split out from the encoder so the arithmetic that rewrites 90% of AST's
    input tensor can be tested without loading 87M parameters.
    """
    fbank = np.asarray(fbank, dtype=np.float32)
    n_frames, n_mels = fbank.shape
    if n_frames < max_length:
        pad = np.zeros((max_length - n_frames, n_mels), dtype=np.float32)
        padded = np.concatenate((fbank, pad), axis=0)
    else:
        padded = fbank[:max_length]
    return ((padded - mean) / (std * 2.0)).astype(np.float32)


class AstFbankEncoder(Encoder):
    """Encode a filterbank into AST's 768-D CLS embedding."""

    encoder_type = 'ast'

    def __init__(self, model_id=DEFAULT_MODEL_ID, config=None, revision=None):
        """Load the pinned checkpoint revision and read its padding constants."""
        # There is deliberately no `pretrained=False` option here. An earlier
        # version had one that changed only the reported id while still calling
        # from_pretrained -- a randomly initialised control that quietly returned
        # pretrained embeddings would invalidate the whole Block 3 comparison.
        # ast_random will be its own encoder, building from config with a
        # recorded seed and persisted weights.
        self.model_id = model_id
        self._config = config or FbankConfig()

        # Resolve from cache only when the caller has no pinned revision. A model
        # loaded for an existing bundle must use THAT bundle's revision, or the
        # recorded identity describes weights other than the ones in memory.
        self.checkpoint_revision = revision or resolve_hf_revision(model_id)
        extractor = ASTFeatureExtractor.from_pretrained(
            model_id, revision=self.checkpoint_revision)
        self.max_length = int(extractor.max_length)
        self.mean = float(extractor.mean)
        self.std = float(extractor.std)
        self.n_mels = int(extractor.num_mel_bins)
        if self.n_mels != self._config.n_mels:
            raise ValueError(
                f'{model_id} expects {self.n_mels} mel bins but the feature config '
                f'produces {self._config.n_mels}')

        self.model = ASTModel.from_pretrained(
            model_id, revision=self.checkpoint_revision)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    @property
    def encoder_config(self):
        """Return the configuration that, with the weights, fixes the embedding space."""
        return {
            'hidden_size': self.embedding_dim,
            'max_length': self.max_length,
            'num_mel_bins': self.n_mels,
            'mean': self.mean,
            'std': self.std,
            'pooling': 'cls',
        }

    @property
    def identity(self):
        """Return this encoder's identity document."""
        return identity_document(
            self.encoder_type, self.encoder_config, 'checkpoint',
            checkpoint_id=self.model_id, checkpoint_revision=self.checkpoint_revision)

    @property
    def encoder_id(self):
        """Return the namespaced fingerprint of this encoder."""
        return derive_encoder_id(self.identity)

    @property
    def required_preprocessing_id(self):
        """Return the feature configuration this encoder was built against."""
        return self._config.preprocessing_id

    def prepare_input(self, fbank):
        """Pad to max_length rows and normalise, returning a (1, L, n_mels) tensor."""
        fbank = self.validate_fbank(fbank, expected_n_mels=self.n_mels)
        normalised = pad_and_normalise(fbank, self.max_length, self.mean, self.std)
        return torch.from_numpy(np.ascontiguousarray(normalised)).unsqueeze(0)

    def embed(self, fbank):
        """Return the 768-D CLS embedding for one filterbank."""
        tensor = self.prepare_input(fbank)
        with torch.no_grad():
            outputs = self.model(input_values=tensor)
            cls = outputs.last_hidden_state[:, 0, :]
        return cls.squeeze(0).cpu().numpy().astype(np.float32)
