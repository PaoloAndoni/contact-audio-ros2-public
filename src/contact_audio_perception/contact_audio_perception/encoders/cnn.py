"""
Small CNN trained from scratch on the published filterbank.

The low-latency candidate: milliseconds per window against AST's ~1.5 s, on the
same 98x128 input every other extractor sees. It trains as an n-way classifier
and the 128-D penultimate layer is used as the embedding, so the same
nearest-centroid classifier runs on top of it.

METHODOLOGY: this encoder is LEARNED, so under leave-one-trial-out it must be
retrained inside every fold. Training it once on all windows and then running
cross-validation over the centroids leaks the held-out trial into the encoder
and makes the reported accuracy meaningless. See evaluate_encoders.py.

With ~81 training windows drawn from 9 clips it will overfit. That is worth
reporting, not tuning away on a dataset this size.
"""
from contact_audio_perception.encoders.base import Encoder
from contact_audio_perception.identity import derive_encoder_id, identity_document
from contact_audio_perception.spectrogram import FbankConfig
import numpy as np
import torch
from torch import nn

DEFAULT_SEED = 20260815
DEFAULT_CHANNELS = (16, 32, 64)
DEFAULT_EMBEDDING_DIM = 128
# The same affine AST applies, so every extractor sees identically scaled input.
NORM_MEAN = -4.2677393
NORM_STD = 4.5689974


class _SmallCnn(nn.Module):
    """Three conv blocks, global pooling, an embedding layer and a linear head."""

    def __init__(self, channels, embedding_dim, n_classes):
        """Build the network."""
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(in_ch, embedding_dim)
        self.head = nn.Linear(embedding_dim, n_classes)

    def embed(self, x):
        """Return the penultimate 128-D representation."""
        x = self.features(x.unsqueeze(1))
        x = self.pool(x).flatten(1)
        return self.embedding(x)

    def forward(self, x):
        """Return class logits."""
        return self.head(torch.relu(self.embed(x)))


class SmallCnnEncoder(Encoder):
    """Trainable CNN encoder over the published filterbank."""

    encoder_type = 'cnn'

    def __init__(self, n_classes, state_dict=None, n_mels=128, channels=DEFAULT_CHANNELS,
                 embedding_dim=DEFAULT_EMBEDDING_DIM, seed=DEFAULT_SEED,
                 weights_sha256='', config=None):
        """Build the network and optionally load trained weights."""
        self.n_mels = int(n_mels)
        self.channels = tuple(int(c) for c in channels)
        self.embedding_dim = int(embedding_dim)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.weights_sha256 = str(weights_sha256)
        self._config = config or FbankConfig()

        torch.manual_seed(self.seed)
        self.model = _SmallCnn(self.channels, self.embedding_dim, self.n_classes)
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @property
    def encoder_config(self):
        """Return the configuration that, with the weights, fixes the space."""
        return {
            'n_mels': self.n_mels,
            'channels': list(self.channels),
            'embedding_dim': self.embedding_dim,
            'n_classes': self.n_classes,
            'seed': self.seed,
            'norm_mean': NORM_MEAN,
            'norm_std': NORM_STD,
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
        """Validate and normalise one filterbank into a (1, frames, mels) tensor."""
        fbank = self.validate_fbank(fbank, expected_n_mels=self.n_mels)
        scaled = (fbank - NORM_MEAN) / (NORM_STD * 2.0)
        return torch.from_numpy(np.ascontiguousarray(scaled, dtype=np.float32))\
            .unsqueeze(0)

    def embed(self, fbank):
        """Return the penultimate-layer embedding for one filterbank."""
        with torch.no_grad():
            return self.model.embed(self.prepare_input(fbank))\
                .squeeze(0).numpy().astype(np.float32)

    def fit(self, fbanks, labels, epochs=40, batch_size=16, learning_rate=1e-3):
        """
        Train the classifier head end to end on the given windows.

        Returns the final mean training loss. Callers under cross-validation MUST
        pass only that fold's training windows.
        """
        torch.manual_seed(self.seed)
        data = np.stack([
            (np.asarray(f, dtype=np.float32) - NORM_MEAN) / (NORM_STD * 2.0)
            for f in fbanks])
        x = torch.from_numpy(data)
        y = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        self.model.train()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self.seed)
        loss_value = float('nan')
        for _ in range(epochs):
            order = torch.randperm(len(x), generator=generator)
            losses = []
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                optimiser.zero_grad()
                loss = criterion(self.model(x[batch]), y[batch])
                loss.backward()
                optimiser.step()
                losses.append(loss.detach().item())
            loss_value = float(np.mean(losses))
        self.model.eval()
        return loss_value
