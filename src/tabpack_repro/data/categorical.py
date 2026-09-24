"""Binary/categorical feature preprocessing (a06). Must match the official code.

Official order of operations for Churn (lib/data.py::build_dataset):
1. ``extract_bin_from_num``: numeric columns with exactly 2 unique values (over all
   parts, no NaNs) are ordinal-encoded to bool and moved out of x_num. The extracted
   binary columns go *before* the original x_bin columns.
2. ``bin_to_cat`` ('convert-to-cat'): binary features are cast to the dtype of x_cat
   (strings for Churn, e.g. 0.0 -> '0.0') and appended *after* the x_cat columns.
3. ``ordinal_encode`` ('ordinal'): OrdinalEncoder fitted on train (sorted categories);
   unknown val/test categories are mapped to ``train_max + 1`` of that column.

None of the functions mutate their inputs: every returned dict is new, and every
returned array is either freshly computed or a copy.

Churn walk-through (raw: x_num float32 (N, 7), x_bin float32 (N, 3), x_cat '<U7'
(N, 1)): no numeric column is binary, so step 1 extracts nothing; step 2 casts the
three 0.0/1.0 columns to '<U7' ('0.0'/'1.0') and yields a '<U7' (N, 4) x_cat
``[Geography, bin0, bin1, bin2]``; step 3 gives int64 codes with cardinalities
``[3, 2, 2, 2]``.
"""

from __future__ import annotations

import numpy as np
import sklearn.preprocessing

from tabpack_repro.types import PartKey

# Code used for NaN binary values when there is no x_cat to take the dtype from.
_BIN_NAN_CODE = 2.0
# Temporary code for unseen categories; replaced by ``train_max + 1`` afterwards.
_UNKNOWN_SENTINEL = -1


def _check_same_parts(
    a: dict[PartKey, np.ndarray], b: dict[PartKey, np.ndarray], what: str
) -> None:
    if a.keys() != b.keys():
        raise ValueError(
            f'{what}: part keys differ ({sorted(a.keys())} vs {sorted(b.keys())})'
        )


def extract_bin_from_num(
    x_num: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray] | None, dict[PartKey, np.ndarray] | None]:
    """Return ``(extracted_x_bin, remaining_x_num)``; either may be None.

    A numeric column is *binary* when, over the concatenation of all parts, it has
    no NaN and exactly two distinct values ``lo < hi``. Binary columns are encoded
    as ``value == hi`` (the ordinal code of the sorted categories, cast to bool) and
    keep their original relative order. Remaining columns keep their dtype.

    * no binary column: ``(None, copy of x_num)``;
    * all columns binary: ``(extracted, None)``.
    """
    x_all = np.concatenate(list(x_num.values()))
    has_nan = np.isnan(x_all).any(axis=0)
    uniques = [np.unique(column) for column in x_all.T]
    is_bin = np.array([len(u) == 2 for u in uniques], dtype=bool) & ~has_nan
    bin_idx = np.flatnonzero(is_bin)

    if len(bin_idx) == 0:
        return None, {part: x.copy() for part, x in x_num.items()}

    # The larger of the two (sorted) values gets ordinal code 1 -> True.
    hi = np.array([uniques[i][1] for i in bin_idx])
    extracted = {part: x[:, bin_idx] == hi for part, x in x_num.items()}
    remaining = (
        None
        if len(bin_idx) == x_all.shape[1]
        else {part: x[:, ~is_bin] for part, x in x_num.items()}
    )
    return extracted, remaining


def merge_bin(
    extracted: dict[PartKey, np.ndarray] | None, x_bin: dict[PartKey, np.ndarray] | None
) -> dict[PartKey, np.ndarray] | None:
    """Concatenate ``[extracted, x_bin]`` along features (None-aware).

    Uses ``np.concatenate`` dtype promotion, so bool extracted columns merged with
    float32 x_bin become float32 0.0/1.0. With only one side present, a copy of that
    side is returned unchanged (bool stays bool); with neither, None.
    """
    if extracted is None and x_bin is None:
        return None
    if extracted is None or x_bin is None:
        only = extracted if x_bin is None else x_bin
        assert only is not None
        return {part: x.copy() for part, x in only.items()}
    _check_same_parts(extracted, x_bin, 'merge_bin')
    return {
        part: np.concatenate([extracted[part], x_bin[part]], axis=-1)
        for part in extracted
    }


def bin_to_cat(
    x_bin: dict[PartKey, np.ndarray], x_cat: dict[PartKey, np.ndarray] | None
) -> dict[PartKey, np.ndarray]:
    """Official 'convert-to-cat'. Without x_cat: NaN -> 2, cast to int64.

    With x_cat, every binary column is cast with ``astype(x_cat dtype)`` (exact numpy
    casting: float32 1.0 -> '1.0', bool True -> 'True', NaN -> 'nan', and strings are
    truncated to the itemsize of x_cat) and the result is ``[x_cat, bins]``.
    """
    if x_cat is None:
        return {
            part: np.where(np.isnan(x), _BIN_NAN_CODE, x).astype(np.int64)
            for part, x in x_bin.items()
        }
    _check_same_parts(x_cat, x_bin, 'bin_to_cat')
    dtype = next(iter(x_cat.values())).dtype
    return {
        part: np.column_stack([x_cat[part], x_bin[part].astype(dtype)])
        for part in x_cat
    }


def ordinal_encode(
    x_cat: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray], list[int]]:
    """Return int64 codes and per-column cardinalities.

    Cardinality of a column = number of unique *train* codes (official
    ``compute_cat_cardinalities``). Unknown val/test values get code ``train_max+1``
    (which may equal the cardinality; the one-hot encoder maps it to all-zeros).
    """
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=_UNKNOWN_SENTINEL,
        dtype=np.int64,
    ).fit(x_cat['train'])
    codes = {
        part: np.asarray(encoder.transform(x), dtype=np.int64)
        for part, x in x_cat.items()
    }
    train_max = codes['train'].max(axis=0)
    for part, c in codes.items():
        if part == 'train':
            continue
        unknown = c == _UNKNOWN_SENTINEL
        c[unknown] = np.broadcast_to(train_max + 1, c.shape)[unknown]
    cardinalities = [len(np.unique(column)) for column in codes['train'].T]
    return codes, cardinalities
