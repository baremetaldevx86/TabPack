from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tabpack_repro.utils.seed import seed_everything


def _draw() -> tuple[list[float], np.ndarray, torch.Tensor]:
    return (
        [random.random() for _ in range(5)],
        np.random.rand(5),
        torch.rand(5),
    )


@pytest.mark.parametrize('seed', [0, 1, 12345, 2**32 - 1])
def test_same_seed_reproduces_all_rngs(seed: int) -> None:
    seed_everything(seed)
    py1, np1, pt1 = _draw()
    seed_everything(seed)
    py2, np2, pt2 = _draw()
    assert py1 == py2
    np.testing.assert_array_equal(np1, np2)
    assert torch.equal(pt1, pt2)


def test_different_seeds_differ() -> None:
    seed_everything(0)
    py1, np1, pt1 = _draw()
    seed_everything(1)
    py2, np2, pt2 = _draw()
    assert py1 != py2
    assert not np.array_equal(np1, np2)
    assert not torch.equal(pt1, pt2)


def test_matches_individual_seeding() -> None:
    seed_everything(7)
    got = _draw()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = _draw()
    assert got[0] == expected[0]
    np.testing.assert_array_equal(got[1], expected[1])
    assert torch.equal(got[2], expected[2])


def test_numpy_integer_seed_is_accepted() -> None:
    seed_everything(np.int64(3))
    a = torch.rand(3)
    seed_everything(3)
    assert torch.equal(a, torch.rand(3))


@pytest.mark.parametrize('seed', [-1, 2**32])
def test_out_of_range_seed_raises(seed: int) -> None:
    with pytest.raises(ValueError, match='seed'):
        seed_everything(seed)


def test_non_integer_seed_raises() -> None:
    with pytest.raises(TypeError):
        seed_everything(1.5)  # type: ignore[arg-type]


@pytest.mark.gpu
def test_cuda_rng_is_seeded(cuda_device: torch.device) -> None:
    seed_everything(11)
    a = torch.rand(8, device=cuda_device)
    seed_everything(11)
    b = torch.rand(8, device=cuda_device)
    assert torch.equal(a, b)
