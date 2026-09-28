"""
Randomly initialised AST: the pretraining control, not a deployment candidate.

Identical architecture, identical padding and normalisation, identical compute
cost -- only the weights differ. It isolates how much of AST's performance comes
from the architecture and how much from AudioSet pretraining. If the two score
alike, the pretrained checkpoint is not contributing what its cost implies.

The weights are PERSISTED rather than regenerated from the seed. PyTorch's
initialisation is not guaranteed stable across versions, so a seed alone is
provenance, not reproduction evidence. The artifact is ~345 MB, which is exactly
why it lives under eval_runs/ and is never installed or committed.
"""
from contact_audio_perception.encoders.ast_fbank import DEFAULT_MODEL_ID, pad_and_normalise
from contact_audio_perception.encoders.base import Encoder
from contact_audio_perception.identity import (
    derive_encoder_id,
    identity_document,
    resolve_hf_revision,
)
from contact_audio_perception.spectrogram import FbankConfig
import numpy as np
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

DEFAULT_SEED = 20260815


class AstRandomEncoder(Encoder):
    """AST architecture with random weights, used as a pretraining control."""

    encoder_type = 'ast_random'

    def __init__(self, state_dict=None, seed=DEFAULT_SEED, model_id=DEFAULT_MODEL_ID,
                 revision=None, weights_sha256='', config=None):
        """Build the architecture from config and either randomise or load weights."""
        self.model_id = model_id
        self.seed = int(seed)
        self.weights_sha256 = str(weights_sha256)
        self._config = config or FbankConfig()
        self.checkpoint_revision = revision or resolve_hf_revision(model_id)

        # Padding and normalisation must match the pretrained encoder exactly, or
        # the control differs from AST in more than just its weights.
        extractor = ASTFeatureExtractor.from_pretrained(
            model_id, revision=self.checkpoint_revision)
        self.max_length = int(extractor.max_length)
        self.mean = float(extractor.mean)
        self.std = float(extractor.std)
        self.n_mels = int(extractor.num_mel_bins)

        arch = ASTConfig.from_pretrained(model_id, revision=self.checkpoint_revision)
        torch.manual_seed(self.seed)
        self.model = ASTModel(arch)
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    @property
    def encoder_config(self):
        """Return the configuration that, with the weights, fixes the space."""
        return {
            'architecture_of': self.model_id,
            'architecture_revision': self.checkpoint_revision,
            'hidden_size': self.embedding_dim,
            'max_length': self.max_length,
            'num_mel_bins': self.n_mels,
            'mean': self.mean,
            'std': self.std,
            'pooling': 'cls',
            'seed': self.seed,
        }

    @property
    def identity(self):
        """Return this encoder's identity document."""
        return identity_document(self.encoder_type, self.encoder_config, 'bundled',
                                 weights_sha256=self.weights_sha256)

    @property
    def encoder_id(self):
        """Return the namespaced fingerprint of this encoder."""
        return derive_encoder_id(self.identity)

    @property
    def required_preprocessing_id(self):
        """Return the feature configuration this encoder was built against."""
        return self._config.preprocessing_id

    def state_bytes(self):
        """Serialise the current weights as a SafeTensors payload."""
        from safetensors.torch import save
        return save({k: v.contiguous() for k, v in self.model.state_dict().items()})

    def prepare_input(self, fbank):
        """Pad to max_length rows and normalise, exactly as the pretrained AST does."""
        fbank = self.validate_fbank(fbank, expected_n_mels=self.n_mels)
        normalised = pad_and_normalise(fbank, self.max_length, self.mean, self.std)
        return torch.from_numpy(np.ascontiguousarray(normalised)).unsqueeze(0)

    def embed(self, fbank):
        """Return the CLS embedding from the randomly initialised model."""
        tensor = self.prepare_input(fbank)
        with torch.no_grad():
            cls = self.model(input_values=tensor).last_hidden_state[:, 0, :]
        return cls.squeeze(0).cpu().numpy().astype(np.float32)
