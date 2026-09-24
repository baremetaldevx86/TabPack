"""Numerical feature preprocessing (a05). Must match the official ``transform_num``."""

from __future__ import annotations

import numpy as np

from tabpack_repro.types import PartKey


def noisy_quantile_transform(
    x_num: dict[PartKey, np.ndarray], *, seed: int, noise_std: float = 1e-5
) -> dict[PartKey, np.ndarray]:
    """Fit a normal-output QuantileTransformer on noisy train data; apply to all parts.

    Official semantics (lib/data.py::transform_num with 'noisy-quantile'):
    * n_quantiles = max(min(n_train // 30, 1000), 10), output_distribution='normal',
      subsample=1_000_000_000, random_state=seed;
    * fit on ``x_train + np.random.RandomState(seed).normal(0, noise_std, shape)``
      (noise cast to the train dtype), transform the *clean* parts;
    * afterwards: NaN -> 0, drop columns constant on train, cast to float32.
    """
    raise NotImplementedError


def drop_constant_columns(
    x_num: dict[PartKey, np.ndarray],
) -> dict[PartKey, np.ndarray]:
    """Remove columns that have a single unique value on the train part."""
    raise NotImplementedError


def standard_transform(x_num: dict[PartKey, np.ndarray]) -> dict[PartKey, np.ndarray]:
    """StandardScaler fitted on train (policy 'standard'), with the same post-steps."""
    raise NotImplementedError
