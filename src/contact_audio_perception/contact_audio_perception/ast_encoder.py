"""
AST encoder: 16 kHz mono audio -> 768-D CLS embedding.

Kept ROS-free so it can be unit-tested and reused. Mirrors the encoding path of
the original encode_dataset_ast.py (ASTFeatureExtractor + ASTModel, CLS token)
so a live embedding is comparable to the offline dataset embeddings.
"""
import numpy as np
import torch
import torchaudio
from transformers import ASTFeatureExtractor, ASTModel

DEFAULT_MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
AST_SAMPLE_RATE = 16000


def resample_to_16k(audio, sample_rate):
    """Resample a 1-D float32 array to 16 kHz (no-op if already 16 kHz)."""
    if sample_rate == AST_SAMPLE_RATE:
        return np.ascontiguousarray(audio, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    resampled = torchaudio.functional.resample(tensor, sample_rate, AST_SAMPLE_RATE)
    return resampled.numpy()


def window_to_model_input(samples, channels, sample_rate):
    """Convert an int16 window to mono, peak-normalized 16 kHz float32."""
    audio = np.asarray(samples, dtype=np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio / 32768.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    audio = audio / (peak + 1e-6)
    return resample_to_16k(audio, sample_rate)


class AstEncoder:
    """Load the AST model once and encode mono 16 kHz audio into a CLS vector."""

    def __init__(self, model_id=DEFAULT_MODEL_ID):
        """Load the feature extractor and model in eval mode."""
        self.model_id = model_id
        self.extractor = ASTFeatureExtractor.from_pretrained(model_id)
        self.model = ASTModel.from_pretrained(model_id)
        self.model.eval()

    def encode(self, audio_16k):
        """Encode a 1-D float32 16 kHz array into a (768,) float32 CLS vector."""
        inputs = self.extractor(
            audio_16k, sampling_rate=AST_SAMPLE_RATE, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
        return cls.squeeze(0).cpu().numpy().astype(np.float32)
