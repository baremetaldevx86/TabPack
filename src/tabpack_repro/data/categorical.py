"""Binary/categorical feature preprocessing (a06). Must match the official code.

Official order of operations for Churn (lib/data.py::build_dataset):
1. ``extract_bin_from_num``: numeric columns with exactly 2 unique values (over all
   parts, no NaNs) are ordinal-encoded to bool and moved out of x_num. The extracted
   binary columns go *before* the original x_bin columns.
2. ``bin_to_cat`` ('convert-to-cat'): binary features are cast to the dtype of x_cat
   (strings for Churn, e.g. 0.0 -> '0.0') and appended *after* the x_cat columns.
3. ``ordinal_encode`` ('ordinal'): OrdinalEncoder fitted on train (sorted categories);
   unknown val/test categories are mapped to ``train_max + 1`` of that column.
"""

from __future__ import annotations

import numpy as np

from tabpack_repro.types import PartKey


def extract_bin_from_num(
    x_num: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray] | None, dict[PartKey, np.ndarray] | None]:
    """Return ``(extracted_x_bin, remaining_x_num)``; either may be None."""
    raise NotImplementedError


def merge_bin(
    extracted: dict[PartKey, np.ndarray] | None, x_bin: dict[PartKey, np.ndarray] | None
) -> dict[PartKey, np.ndarray] | None:
    """Concatenate ``[extracted, x_bin]`` along features (None-aware)."""
    raise NotImplementedError


def bin_to_cat(
    x_bin: dict[PartKey, np.ndarray], x_cat: dict[PartKey, np.ndarray] | None
) -> dict[PartKey, np.ndarray]:
    """Official 'convert-to-cat'. Without x_cat: NaN -> 2, cast to int64."""
    raise NotImplementedError


def ordinal_encode(
    x_cat: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray], list[int]]:
    """Return int64 codes and per-column cardinalities.

    Cardinality of a column = number of unique *train* codes (official
    ``compute_cat_cardinalities``). Unknown val/test values get code ``train_max+1``
    (which may equal the cardinality; the one-hot encoder maps it to all-zeros).
    """
    raise NotImplementedError
