"""Tests for per-member training batches (a18)."""

from __future__ import annotations

import math

import pytest
import torch

from tabpack_repro.training.batches import epoch_size, generate_member_batches


def _seeded(seed: int = 0, device: str | torch.device = 'cpu') -> torch.Generator:
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return generator


def _official_formula(
    train_size: int, batch_size: int, pack_size: int, generator: torch.Generator
) -> list[torch.Tensor]:
    # Independent re-statement of the official ``generate_training_batches``.
    values = torch.rand(
        (pack_size, train_size), generator=generator, device=generator.device
    )
    return list(values.argsort(dim=1).split(batch_size, dim=1))


def _batches(
    train_size: int = 103, batch_size: int = 16, pack_size: int = 4, seed: int = 0
) -> list[torch.Tensor]:
    return generate_member_batches(
        train_size=train_size,
        batch_size=batch_size,
        pack_size=pack_size,
        generator=_seeded(seed),
    )


def test_returns_list_of_int64_pack_batches() -> None:
    batches = _batches(train_size=103, batch_size=16, pack_size=4)
    assert isinstance(batches, list)
    assert len(batches) == epoch_size(103, 16)
    for batch in batches:
        assert batch.dtype == torch.int64
        assert batch.ndim == 2
        assert batch.shape[0] == 4
        assert batch.device.type == 'cpu'


def test_each_member_sees_a_permutation() -> None:
    train_size = 103
    batches = _batches(train_size=train_size, batch_size=16, pack_size=4)
    full = torch.cat(batches, dim=1)
    assert full.shape == (4, train_size)
    expected = torch.arange(train_size)
    for member in full:
        assert torch.equal(member.sort().values, expected)


def test_members_get_different_orders() -> None:
    full = torch.cat(_batches(train_size=200, batch_size=32, pack_size=5), dim=1)
    for i in range(full.shape[0]):
        for j in range(i + 1, full.shape[0]):
            assert not torch.equal(full[i], full[j])


def test_deterministic_for_seeded_generator() -> None:
    first = _batches(seed=7)
    second = _batches(seed=7)
    other = _batches(seed=8)
    assert len(first) == len(second)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_consecutive_epochs_differ() -> None:
    generator = _seeded(3)
    kwargs = {'train_size': 64, 'batch_size': 8, 'pack_size': 2}
    epoch1 = torch.cat(generate_member_batches(generator=generator, **kwargs), dim=1)
    epoch2 = torch.cat(generate_member_batches(generator=generator, **kwargs), dim=1)
    assert not torch.equal(epoch1, epoch2)


@pytest.mark.parametrize(
    ('train_size', 'batch_size', 'last'),
    [(103, 16, 7), (96, 16, 16), (5, 16, 5), (1, 1, 1), (17, 1, 1)],
)
def test_last_batch_size(train_size: int, batch_size: int, last: int) -> None:
    batches = _batches(train_size=train_size, batch_size=batch_size, pack_size=3)
    assert len(batches) == math.ceil(train_size / batch_size)
    assert all(b.shape == (3, batch_size) for b in batches[:-1])
    assert batches[-1].shape == (3, last)


@pytest.mark.parametrize(
    ('train_size', 'batch_size'),
    [(1, 1), (1, 7), (7, 7), (8, 7), (14, 7), (15, 7), (6412, 256), (100, 1000)],
)
def test_epoch_size_is_ceil(train_size: int, batch_size: int) -> None:
    assert epoch_size(train_size, batch_size) == math.ceil(train_size / batch_size)
    assert epoch_size(train_size, batch_size) == len(
        _batches(train_size=train_size, batch_size=batch_size, pack_size=1)
    )


def test_single_member() -> None:
    batches = _batches(train_size=50, batch_size=12, pack_size=1)
    assert all(b.shape[0] == 1 for b in batches)
    full = torch.cat(batches, dim=1)[0]
    assert torch.equal(full.sort().values, torch.arange(50))


@pytest.mark.parametrize(
    ('train_size', 'batch_size', 'pack_size', 'seed'),
    [(103, 16, 4, 0), (6412, 256, 32, 1), (10, 3, 1, 2), (7, 100, 2, 3)],
)
def test_matches_official_formula(
    train_size: int, batch_size: int, pack_size: int, seed: int
) -> None:
    ours_gen = _seeded(seed)
    ref_gen = _seeded(seed)
    # Two epochs in a row: also checks that the RNG stream advances identically.
    for _ in range(2):
        ours = generate_member_batches(
            train_size=train_size,
            batch_size=batch_size,
            pack_size=pack_size,
            generator=ours_gen,
        )
        ref = _official_formula(train_size, batch_size, pack_size, ref_gen)
        assert len(ours) == len(ref)
        assert all(torch.equal(a, b) for a, b in zip(ours, ref, strict=True))
    assert torch.equal(ours_gen.get_state(), ref_gen.get_state())


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'train_size': 0}, ValueError),
        ({'train_size': -3}, ValueError),
        ({'batch_size': 0}, ValueError),
        ({'pack_size': 0}, ValueError),
        ({'train_size': 10.0}, TypeError),
        ({'batch_size': True}, TypeError),
        ({'pack_size': '2'}, TypeError),
        ({'generator': 0}, TypeError),
    ],
)
def test_generate_member_batches_validates(kwargs: dict, error: type) -> None:
    arguments = {
        'train_size': 10,
        'batch_size': 4,
        'pack_size': 2,
        'generator': _seeded(),
    } | kwargs
    with pytest.raises(error):
        generate_member_batches(**arguments)


@pytest.mark.parametrize(
    ('train_size', 'batch_size', 'error'),
    [(0, 4, ValueError), (4, 0, ValueError), (4, -1, ValueError), (4.0, 2, TypeError)],
)
def test_epoch_size_validates(
    train_size: object, batch_size: object, error: type
) -> None:
    with pytest.raises(error):
        epoch_size(train_size, batch_size)  # type: ignore[arg-type]


@pytest.mark.gpu
def test_cuda_generator(cuda_device: torch.device) -> None:
    kwargs = {'train_size': 1000, 'batch_size': 128, 'pack_size': 8}
    ours = generate_member_batches(generator=_seeded(5, cuda_device), **kwargs)
    ref = _official_formula(
        kwargs['train_size'],
        kwargs['batch_size'],
        kwargs['pack_size'],
        _seeded(5, cuda_device),
    )
    assert all(b.device.type == 'cuda' for b in ours)
    assert all(torch.equal(a, b) for a, b in zip(ours, ref, strict=True))
    full = torch.cat(ours, dim=1)
    expected = torch.arange(kwargs['train_size'], device=cuda_device)
    for member in full:
        assert torch.equal(member.sort().values, expected)
