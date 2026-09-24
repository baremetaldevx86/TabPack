"""Tests for binary/categorical preprocessing (a06).

Each rule of the official pipeline (lib/data.py: ``_extract_bin_from_num``,
``Dataset.convert_bin_features_to_cat_``, ``transform_cat`` with 'ordinal',
``compute_cat_cardinalities``) is checked on small hand-built arrays; the last test
runs the whole chain on the real Churn arrays.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from tabpack_repro.data.categorical import (
    bin_to_cat,
    extract_bin_from_num,
    merge_bin,
    ordinal_encode,
)

NAN = np.nan


def _parts(train, val, test, dtype=None) -> dict[str, np.ndarray]:
    return {
        'train': np.array(train, dtype=dtype),
        'val': np.array(val, dtype=dtype),
        'test': np.array(test, dtype=dtype),
    }


def _assert_parts_equal(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for part in expected:
        assert actual[part].dtype == expected[part].dtype, part
        np.testing.assert_array_equal(actual[part], expected[part], err_msg=part)


# >>> extract_bin_from_num


def test_extract_mixed_columns_keeps_order_and_dtypes():
    # Columns: 0 binary (0/1), 1 continuous, 2 binary (-1/3), 3 constant,
    # 4 two values + NaN (not binary), 5 one value + NaN (not binary).
    x_num = _parts(
        [[0, 0.5, 3, 7, 1, 4], [1, 1.5, -1, 7, 2, 4]],
        [[1, 2.5, 3, 7, NAN, NAN]],
        [[0, 3.5, -1, 7, 2, 4]],
        dtype=np.float32,
    )
    original = copy.deepcopy(x_num)

    extracted, remaining = extract_bin_from_num(x_num)

    assert extracted is not None and remaining is not None
    # Binary columns become bool: the larger of the two sorted values is True.
    _assert_parts_equal(
        extracted,
        _parts([[False, True], [True, False]], [[True, True]], [[False, False]], bool),
    )
    _assert_parts_equal(
        remaining,
        _parts(
            [[0.5, 7, 1, 4], [1.5, 7, 2, 4]],
            [[2.5, 7, NAN, NAN]],
            [[3.5, 7, 2, 4]],
            np.float32,
        ),
    )
    _assert_parts_equal(x_num, original)  # inputs are not mutated


def test_extract_uses_unique_values_over_all_parts():
    # Column 0 is constant on train but binary over all parts -> extracted, and the
    # second value (seen only in test) is the larger one -> True.
    # Column 1 has two values on train but a third one in val -> not binary.
    x_num = _parts([[5, 0], [5, 1]], [[5, 2]], [[7, 1]], dtype=np.float32)
    extracted, remaining = extract_bin_from_num(x_num)
    assert extracted is not None and remaining is not None
    _assert_parts_equal(
        extracted, _parts([[False], [False]], [[False]], [[True]], bool)
    )
    _assert_parts_equal(remaining, _parts([[0], [1]], [[2]], [[1]], np.float32))


def test_extract_all_binary():
    x_num = _parts([[0, 10], [1, 20]], [[1, 10]], [[0, 20]], dtype=np.float32)
    extracted, remaining = extract_bin_from_num(x_num)
    assert remaining is None
    assert extracted is not None
    _assert_parts_equal(
        extracted,
        _parts([[False, False], [True, True]], [[True, False]], [[False, True]], bool),
    )


def test_extract_no_binary_returns_copy():
    x_num = _parts([[0.0, 1], [1, 2]], [[2, NAN]], [[3, 3]], dtype=np.float32)
    extracted, remaining = extract_bin_from_num(x_num)
    assert extracted is None
    assert remaining is not None
    _assert_parts_equal(remaining, x_num)
    for part in x_num:
        assert not np.shares_memory(remaining[part], x_num[part])


def test_extract_nan_columns_are_never_binary():
    # 0/1 + NaN (3 unique incl. NaN) and 1 + NaN (np.unique gives 2 entries).
    x_num = _parts([[0, 1], [1, NAN]], [[NAN, 1]], [[1, 1]], dtype=np.float32)
    extracted, remaining = extract_bin_from_num(x_num)
    assert extracted is None
    assert remaining is not None
    _assert_parts_equal(remaining, x_num)


def test_extract_float64_keeps_dtype():
    x_num = _parts([[0.25, 1], [0.75, 2]], [[0.25, 3]], [[0.75, 4]], dtype=np.float64)
    extracted, remaining = extract_bin_from_num(x_num)
    assert extracted is not None and remaining is not None
    assert remaining['train'].dtype == np.float64
    np.testing.assert_array_equal(extracted['train'][:, 0], [False, True])


# >>> merge_bin


def test_merge_bin_none_aware():
    bools = _parts([[True], [False]], [[True]], [[False]], bool)
    floats = _parts([[0.0], [1.0]], [[1.0]], [[0.0]], np.float32)
    assert merge_bin(None, None) is None
    only_x_bin = merge_bin(None, floats)
    only_extracted = merge_bin(bools, None)
    assert only_x_bin is not None and only_extracted is not None
    _assert_parts_equal(only_x_bin, floats)
    _assert_parts_equal(only_extracted, bools)  # stays bool (no promotion)
    for part in floats:
        assert not np.shares_memory(only_x_bin[part], floats[part])


def test_merge_bin_puts_extracted_first_with_dtype_promotion():
    extracted = _parts([[True], [False]], [[True]], [[False]], bool)
    x_bin = _parts([[0.0, 1], [1, 1]], [[0, 0]], [[1, 0]], np.float32)
    original = copy.deepcopy(x_bin)
    merged = merge_bin(extracted, x_bin)
    assert merged is not None
    _assert_parts_equal(
        merged,
        _parts([[1.0, 0, 1], [0, 1, 1]], [[1, 0, 0]], [[0, 1, 0]], np.float32),
    )
    _assert_parts_equal(x_bin, original)


def test_merge_bin_rejects_mismatched_parts():
    a = {'train': np.zeros((2, 1), bool), 'val': np.zeros((1, 1), bool)}
    b = {'train': np.zeros((2, 1), np.float32), 'test': np.zeros((1, 1), np.float32)}
    with pytest.raises(ValueError, match='part keys'):
        merge_bin(a, b)


# >>> bin_to_cat


def test_bin_to_cat_casts_floats_to_x_cat_strings_and_appends_after_x_cat():
    x_cat = _parts([['Spain'], ['France']], [['Germany']], [['Spain']], '<U7')
    x_bin = _parts([[0.0, 1], [1, NAN]], [[1, 1]], [[0, 0]], np.float32)
    original = copy.deepcopy((x_cat, x_bin))
    result = bin_to_cat(x_bin, x_cat)
    _assert_parts_equal(
        result,
        _parts(
            [['Spain', '0.0', '1.0'], ['France', '1.0', 'nan']],
            [['Germany', '1.0', '1.0']],
            [['Spain', '0.0', '0.0']],
            '<U7',
        ),
    )
    _assert_parts_equal(x_cat, original[0])
    _assert_parts_equal(x_bin, original[1])


def test_bin_to_cat_bool_bins_become_true_false_strings():
    # Extracted-only bins (no original x_bin) are bool -> 'True'/'False'.
    x_cat = _parts([['a'], ['b']], [['a']], [['b']], '<U7')
    x_bin = _parts([[True], [False]], [[False]], [[True]], bool)
    result = bin_to_cat(x_bin, x_cat)
    np.testing.assert_array_equal(result['train'], [['a', 'True'], ['b', 'False']])
    assert result['train'].dtype == np.dtype('<U7')


def test_bin_to_cat_truncates_to_x_cat_itemsize():
    x_cat = _parts([['a'], ['b']], [['a']], [['b']], '<U1')
    x_bin = _parts([[0.0, 1], [1, 0]], [[1, 1]], [[0, 0]], np.float32)
    result = bin_to_cat(x_bin, x_cat)
    np.testing.assert_array_equal(result['train'], [['a', '0', '1'], ['b', '1', '0']])
    assert result['train'].dtype == np.dtype('<U1')


def test_bin_to_cat_int_x_cat():
    x_cat = _parts([[4], [5]], [[6]], [[4]], np.int64)
    x_bin = _parts([[0.0], [1.0]], [[1.0]], [[0.0]], np.float32)
    result = bin_to_cat(x_bin, x_cat)
    _assert_parts_equal(result, _parts([[4, 0], [5, 1]], [[6, 1]], [[4, 0]], np.int64))


@pytest.mark.parametrize('dtype', [np.float32, bool])
def test_bin_to_cat_without_x_cat(dtype):
    values = [[0, 1], [1, 0]] if dtype is bool else [[0, 1], [1, NAN]]
    x_bin = _parts(values, [[1, 1]], [[0, 0]], dtype)
    result = bin_to_cat(x_bin, None)
    expected_train = [[0, 1], [1, 0]] if dtype is bool else [[0, 1], [1, 2]]
    _assert_parts_equal(result, _parts(expected_train, [[1, 1]], [[0, 0]], np.int64))


def test_bin_to_cat_rejects_mismatched_parts():
    x_cat = {'train': np.array([['a']]), 'val': np.array([['b']])}
    x_bin = {'train': np.array([[0.0]]), 'test': np.array([[1.0]])}
    with pytest.raises(ValueError, match='part keys'):
        bin_to_cat(x_bin, x_cat)


# >>> ordinal_encode


def test_ordinal_encode_sorted_categories_fitted_on_train():
    x_cat = _parts(
        [['b', 'z'], ['a', 'y'], ['c', 'z'], ['a', 'z']],
        [['c', 'y']],
        [['b', 'z']],
        '<U1',
    )
    original = copy.deepcopy(x_cat)
    codes, cardinalities = ordinal_encode(x_cat)
    _assert_parts_equal(
        codes,
        _parts([[1, 1], [0, 0], [2, 1], [0, 1]], [[2, 0]], [[1, 1]], np.int64),
    )
    assert cardinalities == [3, 2]
    _assert_parts_equal(x_cat, original)


def test_ordinal_encode_unknown_values_get_train_max_plus_one_per_column():
    x_cat = _parts(
        [['a', 'p'], ['b', 'q'], ['a', 'r']],
        [['UNSEEN', 'p'], ['b', 'UNSEEN']],
        [['x', 'y'], ['a', 'r']],
        '<U6',
    )
    codes, cardinalities = ordinal_encode(x_cat)
    np.testing.assert_array_equal(codes['train'], [[0, 0], [1, 1], [0, 2]])
    # Column 0: train max 1 -> unknown 2; column 1: train max 2 -> unknown 3.
    np.testing.assert_array_equal(codes['val'], [[2, 0], [1, 3]])
    np.testing.assert_array_equal(codes['test'], [[2, 3], [0, 2]])
    assert cardinalities == [2, 3]
    assert all(codes[p].dtype == np.int64 for p in codes)


def test_ordinal_encode_nan_string_is_a_category():
    # NaN binary values cast to strings become 'nan', an ordinary category.
    x_cat = _parts([['0.0'], ['nan'], ['1.0']], [['nan']], [['1.0']], '<U7')
    codes, cardinalities = ordinal_encode(x_cat)
    np.testing.assert_array_equal(codes['train'][:, 0], [0, 2, 1])
    np.testing.assert_array_equal(codes['val'][:, 0], [2])
    assert cardinalities == [3]


def test_ordinal_encode_int_codes_from_bins_without_x_cat():
    x_bin = _parts([[0.0], [NAN], [1.0]], [[1.0]], [[NAN]], np.float32)
    codes, cardinalities = ordinal_encode(bin_to_cat(x_bin, None))
    np.testing.assert_array_equal(codes['train'][:, 0], [0, 2, 1])
    np.testing.assert_array_equal(codes['test'][:, 0], [2])
    assert cardinalities == [3]


def test_ordinal_encode_value_missing_from_train_is_unknown():
    # '1.0' exists only in val -> it is unknown, not a new category.
    x_cat = _parts([['0.0'], ['0.0']], [['1.0']], [['0.0']], '<U7')
    codes, cardinalities = ordinal_encode(x_cat)
    np.testing.assert_array_equal(codes['val'][:, 0], [1])
    assert cardinalities == [1]


# >>> the whole chain


def _run_chain(x_num, x_bin, x_cat):
    extracted, remaining = extract_bin_from_num(x_num)
    merged = merge_bin(extracted, x_bin)
    assert merged is not None
    codes, cardinalities = ordinal_encode(bin_to_cat(merged, x_cat))
    return remaining, codes, cardinalities


def test_chain_column_order_is_cat_then_extracted_then_original_bins():
    x_num = _parts([[9.0, 0.5], [8, 1.5]], [[9, 2.5]], [[8, 3.5]], np.float32)
    x_bin = _parts([[1.0], [0]], [[1]], [[0]], np.float32)
    x_cat = _parts([['q'], ['p']], [['r']], [['p']], '<U1')
    remaining, codes, cardinalities = _run_chain(x_num, x_bin, x_cat)
    assert remaining is not None
    np.testing.assert_array_equal(remaining['train'], [[0.5], [1.5]])
    # [x_cat, extracted (9 -> 1, 8 -> 0), x_bin]; 'r' is unknown in val -> 2.
    np.testing.assert_array_equal(codes['train'], [[1, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(codes['val'], [[2, 1, 1]])
    np.testing.assert_array_equal(codes['test'], [[0, 0, 0]])
    assert cardinalities == [2, 2, 2]


def test_chain_bool_and_float_bins_give_the_same_codes():
    # 'False'/'True' and '0.0'/'1.0' sort the same way.
    x_cat = _parts([['a'], ['b'], ['a']], [['b']], [['a']], '<U7')
    as_bool = _parts([[True], [False], [True]], [[False]], [[True]], bool)
    as_float = {k: v.astype(np.float32) for k, v in as_bool.items()}
    codes_bool, card_bool = ordinal_encode(bin_to_cat(as_bool, x_cat))
    codes_float, card_float = ordinal_encode(bin_to_cat(as_float, x_cat))
    _assert_parts_equal(codes_bool, codes_float)
    assert card_bool == card_float == [2, 2]


# >>> real data


@pytest.mark.data
def test_churn_raw_arrays(churn_dir: Path):
    split_dir = churn_dir / 'splits' / 'default'
    idx = {p: np.load(split_dir / f'{p}.npy') for p in ('train', 'val', 'test')}
    raw = {
        name: np.load(churn_dir / f'{name}.npy', allow_pickle=False)
        for name in ('x_num', 'x_bin', 'x_cat')
    }
    x_num, x_bin, x_cat = ({p: a[i] for p, i in idx.items()} for a in raw.values())

    extracted, remaining = extract_bin_from_num(x_num)
    # Churn has no binary numeric column (Tenure has 11 values, NumOfProducts 4).
    assert extracted is None
    assert remaining is not None and remaining['train'].shape[1] == 7
    merged = merge_bin(extracted, x_bin)
    assert merged is not None and merged['train'].dtype == np.float32

    as_cat = bin_to_cat(merged, x_cat)
    assert as_cat['train'].dtype == x_cat['train'].dtype
    assert set(np.unique(as_cat['train'][:, 1:])) == {'0.0', '1.0'}
    np.testing.assert_array_equal(as_cat['train'][:, 0], x_cat['train'][:, 0])

    codes, cardinalities = ordinal_encode(as_cat)
    n_cat = x_cat['train'].shape[1] + x_bin['train'].shape[1]
    assert n_cat == 4
    assert len(cardinalities) == n_cat
    # Geography has 3 values (France, Germany, Spain); then three 0/1 flags.
    assert cardinalities == [3, 2, 2, 2]
    for part, c in codes.items():
        assert c.dtype == np.int64
        assert c.shape == (len(idx[part]), n_cat)
        # Every val/test category was seen on train: no unknown codes.
        assert (c >= 0).all() and (c < np.array(cardinalities)).all(), part
