"""Numerical feature preprocessing (a05). Must match the official ``transform_num``.

Both public transforms follow the same recipe:

1. fit a scikit-learn normalizer on the train part (optionally perturbed by tiny noise),
2. transform every part (the *clean* arrays, never the noisy train copy),
3. ``np.nan_to_num`` (NaN -> 0, +-inf -> the largest finite values),
4. drop the columns that are constant on the transformed train part,
5. cast to float32.

The inputs are never modified in place: every step returns fresh arrays.
"""

from __future__ import annotations

import numpy as np
import sklearn.preprocessing

from tabpack_repro.types import PartKey

_X_NUM_DTYPE = np.float32
# Chosen larger than any realistic train size so that QuantileTransformer never
# subsamples (the official value); `random_state` is then effectively unused.
_QUANTILE_SUBSAMPLE = 1_000_000_000


def _n_quantiles(n_train: int) -> int:
    """Official rule: roughly one quantile per 30 train rows, clipped to [10, 1000]."""
    return max(min(n_train // 30, 1000), 10)


def _fit_transform_parts(
    normalizer: sklearn.preprocessing.StandardScaler
    | sklearn.preprocessing.QuantileTransformer,
    x_fit: np.ndarray,
    x_num: dict[PartKey, np.ndarray],
) -> dict[PartKey, np.ndarray]:
    """Fit `normalizer` on `x_fit` and transform every (clean) part of `x_num`."""
    normalizer.fit(x_fit)
    return {part: normalizer.transform(x) for part, x in x_num.items()}


def _finalize(x_num: dict[PartKey, np.ndarray]) -> dict[PartKey, np.ndarray]:
    """Official post-steps: NaN -> 0, drop train-constant columns, cast to float32."""
    x_num = {part: np.nan_to_num(x) for part, x in x_num.items()}
    x_num = drop_constant_columns(x_num)
    return {part: x.astype(_X_NUM_DTYPE) for part, x in x_num.items()}


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
    x_train = x_num['train']
    normalizer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=_n_quantiles(x_train.shape[0]),
        output_distribution='normal',
        subsample=_QUANTILE_SUBSAMPLE,
        random_state=seed,
    )
    # The noise breaks ties between repeated values (e.g. integer-valued features),
    # so that the fitted quantiles are strictly increasing.
    noise = np.random.RandomState(seed).normal(0.0, noise_std, x_train.shape)
    x_train_noisy = x_train + noise.astype(x_train.dtype)
    return _finalize(_fit_transform_parts(normalizer, x_train_noisy, x_num))


def drop_constant_columns(
    x_num: dict[PartKey, np.ndarray],
) -> dict[PartKey, np.ndarray]:
    """Remove columns that have a single unique value on the train part.

    The mask is computed on train only and applied to every part, so all parts keep
    the same columns. Dtypes are preserved; the returned arrays are new copies.
    """
    x_train = x_num['train']
    # dtype=bool keeps the mask valid for indexing even when there are no columns.
    keep = np.array([len(np.unique(column)) > 1 for column in x_train.T], dtype=bool)
    return {part: x[:, keep] for part, x in x_num.items()}


def standard_transform(x_num: dict[PartKey, np.ndarray]) -> dict[PartKey, np.ndarray]:
    """StandardScaler fitted on train (policy 'standard'), with the same post-steps."""
    normalizer = sklearn.preprocessing.StandardScaler()
    return _finalize(_fit_transform_parts(normalizer, x_num['train'], x_num))
