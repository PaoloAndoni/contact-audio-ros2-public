"""
Nearest-centroid contact classifier over AST CLS embeddings.

Kept ROS-free so it can be trained and unit-tested offline. Centroids are stored
in an .npz with arrays: class_names (str) and centroids (n_classes, dim).

Classification is cosine nearest-centroid. When the best cosine similarity is
below `unknown_threshold`, the label becomes 'unknown'. Output contract of
predict():
  * class_names / probabilities ALWAYS describe the known classes (a softmax over
    cosine similarities); they never include 'unknown'.
  * confidence is the best KNOWN-class probability, even when label == 'unknown'.
    An 'unknown' label means "closest known class fell below threshold"; the
    probabilities still show which known class it leaned toward.
"""
import numpy as np


class CentroidClassifier:
    """Cosine nearest-centroid classifier with an unknown-reject threshold."""

    def __init__(self, class_names, centroids, unknown_threshold=0.0, temperature=0.1):
        """Validate and store class names and unit-normalized centroids."""
        self.class_names = [str(name) for name in class_names]
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.unknown_threshold = float(unknown_threshold)
        self.temperature = float(temperature)
        self._validate()
        self._unit = self._normalize(self.centroids)

    def _validate(self):
        """Raise ValueError if the centroid model is malformed."""
        if self.centroids.ndim != 2:
            raise ValueError(
                f'centroids must be 2-D (n_classes, dim), got {self.centroids.shape}'
            )
        n_classes, dim = self.centroids.shape
        if n_classes == 0 or dim == 0:
            raise ValueError(f'centroids must be non-empty, got {self.centroids.shape}')
        if len(self.class_names) != n_classes:
            raise ValueError(
                f'class_names ({len(self.class_names)}) does not match centroid '
                f'rows ({n_classes})'
            )
        if not np.isfinite(self.centroids).all():
            raise ValueError('centroids contain non-finite values')
        norms = np.linalg.norm(self.centroids, axis=1)
        if np.any(norms == 0.0):
            zero = [self.class_names[i] for i in np.where(norms == 0.0)[0]]
            raise ValueError(f'centroids contain zero vectors for classes: {zero}')
        if self.temperature <= 0.0:
            raise ValueError(f'temperature must be > 0, got {self.temperature}')

    @staticmethod
    def _normalize(x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-8)

    @staticmethod
    def _softmax(x):
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.sum(e)

    @classmethod
    def from_npz(cls, path, unknown_threshold=0.0, temperature=0.1):
        """Load a classifier from a centroids .npz file."""
        data = np.load(path, allow_pickle=True)
        if 'class_names' not in data or 'centroids' not in data:
            raise ValueError(f"{path}: expected arrays 'class_names' and 'centroids'")
        return cls(data['class_names'], data['centroids'], unknown_threshold, temperature)

    def predict(self, embedding):
        """Classify one embedding into (label, confidence, class_names, probabilities)."""
        vec = self._normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        sims = self._unit @ vec
        probs = self._softmax(sims / self.temperature)
        best = int(np.argmax(sims))
        label = self.class_names[best]
        if float(sims[best]) < self.unknown_threshold:
            label = 'unknown'
        return label, float(probs[best]), list(self.class_names), probs.tolist()
