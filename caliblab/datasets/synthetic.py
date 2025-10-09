from typing import Literal, Optional, Tuple
import numpy as np
import torch

class Synthetic2DClassifier:
    """
    2D synthetic classification with up to ~10k classes.
    Ground truth p(y|x) is an equal-prior, shared-covariance Gaussian mixture:
      p(y=k|x) ∝ N(x; μ_k, σ^2 I)
    """
    def __init__(
        self,
        n_classes: int,
        layout: Literal["grid", "circle"] = "grid",
        spacing: float = 1.0,
        sigma_scale: float = 0.35,
        center_jitter: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        max_classes_per_chunk: int = 5000,
    ):
        """
        Args:
            n_classes: number of classes (1..10_000+)
            layout: 'grid' (best for large K) or 'circle' (pretty for small K)
            spacing: distance between neighboring centers (affects difficulty)
            sigma_scale: σ as a fraction of spacing; higher = more overlap (aleatoric)
            center_jitter: std dev of random jitter added to centers (in same units as spacing)
            rng: numpy Generator for reproducibility
            max_classes_per_chunk: chunk size for proba() to control memory when n*K is huge
        """
        assert n_classes >= 1
        self.K = n_classes
        self.layout = layout
        self.name = f"synthetic_{self.K}_{self.layout}"
        self.spacing = float(spacing)
        self.sigma_scale = float(sigma_scale)
        self.center_jitter = float(center_jitter)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_classes_per_chunk = int(max_classes_per_chunk)

        # Place class centers μ_k in 2D
        self.mu = self._make_centers()
        if self.center_jitter > 0:
            self.mu += self.rng.normal(0.0, self.center_jitter, size=self.mu.shape)

        # Shared isotropic covariance σ^2 I
        # Choose σ relative to spacing to guarantee controllable overlap.
        self.sigma = self.sigma_scale * self.spacing
        self.sigma2 = self.sigma ** 2

    # -------------------
    # Center layouts
    # -------------------
    def _make_centers(self) -> np.ndarray:
        if self.layout == "grid":
            # Arrange on a near-square grid centered at the origin
            g = int(np.ceil(np.sqrt(self.K)))
            xs = (np.arange(g) - (g - 1) / 2.0) * self.spacing
            xv, yv = np.meshgrid(xs, xs)
            grid = np.stack([xv.ravel(), yv.ravel()], axis=1)[: self.K]
            return grid.astype(np.float32)
        elif self.layout == "circle":
            # Equiangular points on a circle
            # Radius grows slightly with K to keep local spacing ~constant
            # Approximate arc spacing ≈ spacing → R ≈ spacing * K / (2π)
            radius = max(1e-6, self.spacing * self.K / (2 * np.pi))
            angles = np.linspace(0, 2 * np.pi, self.K, endpoint=False)
            cx = radius * np.cos(angles)
            cy = radius * np.sin(angles)
            return np.stack([cx, cy], axis=1).astype(np.float32)
        else:
            raise ValueError("layout must be 'grid' or 'circle'")

    # -------------------
    # Sampling
    # -------------------
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample n points:
            y ~ Uniform{0..K-1}
            x ~ N(μ_y, σ^2 I)
        Returns:
            x: (n, 2) float32
            y: (n,) int64
            p_true: (n, K) float64, exact p(y|x)
        """
        y = self.rng.integers(0, self.K, size=n, dtype=np.int64)
        x = self.mu[y] + self.rng.normal(0.0, self.sigma, size=(n, 2)).astype(np.float32)
        p = self.proba(x)  # exact p(y|x)
        return x, y, p
    
    def get_test_loader(self, batch_size=512, size=10000):
        x, y, p = self.sample(n=size)
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=batch_size
        )
        return dataloader

    # -------------------
    # Ground-truth probabilities
    # -------------------
    def proba(self, x: np.ndarray, chunk: Optional[int] = None) -> np.ndarray:
        """
        Compute ground-truth categorical vectors p(y|x) for a batch x.
        Args:
            x: (n, 2)
            chunk: override chunk size (defaults to max_classes_per_chunk)
        Returns:
            p: (n, K) where rows sum to 1
        """
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2 and x.shape[1] == 2, "x must be (n, 2)"
        n = x.shape[0]
        K = self.K

        # For stability: compute logits_k(x) = -||x-μ_k||^2/(2σ^2)
        # and then softmax across k.
        sigma2 = self.sigma2
        p = np.empty((n, K), dtype=np.float64)

        chunk = self.max_classes_per_chunk if chunk is None else int(chunk)
        for start in range(0, K, chunk):
            end = min(K, start + chunk)
            mu_block = self.mu[start:end].astype(np.float64)  # (B,2)
            # Compute squared distances efficiently: ||x-μ||^2 = ||x||^2 + ||μ||^2 - 2 x·μ
            x2 = np.sum(x.astype(np.float64) ** 2, axis=1, keepdims=True)  # (n,1)
            mu2 = np.sum(mu_block ** 2, axis=1, keepdims=True).T           # (1,B)
            cross = x.astype(np.float64) @ mu_block.T                      # (n,B)
            d2 = x2 + mu2 - 2.0 * cross                                    # (n,B)
            logits = -0.5 * d2 / sigma2                                     # (n,B)

            # Stable softmax across full K => we need running max & sumexp
            # We'll compute softmax in two passes: collect per-block max, then normalize.
            if start == 0:
                max_logits = logits.max(axis=1, keepdims=True)  # (n,1)
            else:
                max_logits = np.maximum(max_logits, logits.max(axis=1, keepdims=True))

            # Store logits temporarily; we’ll normalize after finding global max
            p[:, start:end] = logits

        # Second pass: exp(logits - max) and normalize
        sums = np.zeros((n, 1), dtype=np.float64)
        for start in range(0, K, chunk):
            end = min(K, start + chunk)
            block = p[:, start:end]
            block -= max_logits  # broadcast (n,1)
            np.exp(block, out=block)
            sums += block.sum(axis=1, keepdims=True)

        p /= sums  # now p holds normalized probabilities
        return p

    # -------------------
    # Bayes-optimal prediction
    # -------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Return argmax_k p(y=k|x).
        """
        return self.proba(x).argmax(axis=1).astype(np.int64)

    # -------------------
    # Quick difficulty controls
    # -------------------
    def set_difficulty(self, overlap: Literal["easy", "medium", "hard"] = "medium"):
        """
        Convenience: adjust sigma relative to spacing to change aleatoric overlap.
        """
        if overlap == "easy":
            self.sigma = 0.20 * self.spacing
        elif overlap == "medium":
            self.sigma = 0.35 * self.spacing
        elif overlap == "hard":
            self.sigma = 0.50 * self.spacing
        else:
            raise ValueError("overlap must be 'easy'|'medium'|'hard'")
        self.sigma2 = self.sigma ** 2
