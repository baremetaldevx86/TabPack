"""Tests for make_param_groups and optimizer_select_ (a17)."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from tabpack_repro.nn.linear_pack import LinearPack
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.nn.pack_ops import pack_select_
from tabpack_repro.optim.adamw_pack import AdamWPack
from tabpack_repro.optim.muon_adamw_pack import MuonAdamWPack
from tabpack_repro.optim.pack_utils import make_param_groups, optimizer_select_

N_NUM = 3
CAT_CARDINALITIES = (3, 2)
D_IN = N_NUM + sum(CAT_CARDINALITIES)  # 8


def _make_model(
    *,
    pack_size: int = 4,
    d_block: int = 16,
    n_blocks: int | list[int] = 3,
    seed: int = 0,
) -> ModelPack:
    torch.manual_seed(seed)
    return ModelPack(
        n_num_features=N_NUM,
        cat_cardinalities=CAT_CARDINALITIES,
        n_classes=None,
        pack_size=pack_size,
        d_block=d_block,
        n_blocks=n_blocks,
        dropout=0.0,
    )


def _group_params(groups: list[dict[str, Any]]) -> list[Tensor]:
    return [p for group in groups for p in group['params']]


def _zero_wd_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        group
        for group in groups
        if isinstance(group.get('weight_decay'), float) and group['weight_decay'] == 0.0
    ]


# >>> make_param_groups


@pytest.mark.parametrize('muon', [False, True])
@pytest.mark.parametrize('n_blocks', [1, 3, [1, 3, 2, 3]])
def test_every_parameter_appears_exactly_once(
    muon: bool, n_blocks: int | list[int]
) -> None:
    model = _make_model(n_blocks=n_blocks)
    params = _group_params(make_param_groups(model, muon=muon))
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters()}
    for group in make_param_groups(model, muon=muon):
        assert group['params'], 'empty groups must be omitted'


@pytest.mark.parametrize('muon', [False, True])
def test_biases_go_to_the_zero_weight_decay_group(muon: bool) -> None:
    model = _make_model()
    groups = make_param_groups(model, muon=muon)
    zero_wd = _zero_wd_groups(groups)
    assert len(zero_wd) == 1
    assert type(zero_wd[0]['weight_decay']) is float
    biases = [p for p in model.parameters() if p.ndim <= 2]
    expected = [block.linear.bias for block in model.backbone.blocks]
    expected.append(model.head.bias)
    assert {id(p) for p in biases} == {id(p) for p in expected}
    assert [id(p) for p in zero_wd[0]['params']] == [id(p) for p in biases]
    # No other group overrides the weight decay or contains a bias.
    for group in groups:
        if group is not zero_wd[0]:
            assert 'weight_decay' not in group
            assert all(p.ndim == 3 for p in group['params'])


def test_muon_false_layout() -> None:
    model = _make_model(n_blocks=[1, 3, 2, 3])
    groups = make_param_groups(model, muon=False)
    assert len(groups) == 2
    default, zero_wd = groups
    assert set(default) == {'params'}
    assert set(zero_wd) == {'params', 'weight_decay'}
    assert zero_wd['weight_decay'] == 0.0
    expected_weights = [block.linear.weight for block in model.backbone.blocks]
    expected_weights.append(model.head.weight)
    assert [id(p) for p in default['params']] == [
        id(p) for p in model.parameters() if p.ndim == 3
    ]
    assert {id(p) for p in default['params']} == {id(p) for p in expected_weights}
    assert all('muon' not in group for group in groups)


@pytest.mark.parametrize(
    ('d_block', 'first_scale'),
    [(16, math.sqrt(16 / D_IN)), (32, math.sqrt(32 / D_IN)), (4, 1.0)],
)
def test_muon_groups_one_per_backbone_block(d_block: int, first_scale: float) -> None:
    pack_size = 4
    model = _make_model(pack_size=pack_size, d_block=d_block, n_blocks=[1, 3, 2, 3])
    groups = make_param_groups(model, muon=True)
    blocks = list(model.backbone.blocks)
    assert len(blocks) == 3

    muon_groups = [group for group in groups if group.get('muon') is True]
    assert len(muon_groups) == len(blocks)
    # Order of the official make_parameter_groups: default, zero wd, custom (Muon).
    assert groups[-len(blocks) :] == muon_groups
    for i, (group, block) in enumerate(zip(muon_groups, blocks, strict=True)):
        assert set(group) == {'params', 'muon', 'muon_scale'}
        assert len(group['params']) == 1
        assert group['params'][0] is block.linear.weight
        scale = group['muon_scale']
        assert isinstance(scale, Tensor)
        assert scale.shape == (pack_size,)
        assert scale.dtype == torch.float32
        assert scale.device == block.linear.weight.device
        _, d_in, d_out = block.linear.weight.shape
        expected = first_scale if i == 0 else 1.0
        assert expected == pytest.approx(math.sqrt(max(1.0, d_out / d_in)))
        torch.testing.assert_close(scale, torch.full((pack_size,), expected))

    # The head weight is the only weight left for the default (AdamW) group.
    default = [g for g in groups if 'muon' not in g and 'weight_decay' not in g]
    assert len(default) == 1
    assert [id(p) for p in default[0]['params']] == [id(model.head.weight)]


def test_muon_scale_matches_the_official_float32_formula() -> None:
    # Official: (out_features.float() / in_features.float()).clamp_(min=1).sqrt_().
    model = _make_model(d_block=13)
    (group,) = [g for g in make_param_groups(model, muon=True) if g.get('muon')][:1]
    expected = (
        (torch.full((4,), 13.0) / torch.full((4,), float(D_IN))).clamp_(min=1.0).sqrt_()
    )
    assert torch.equal(group['muon_scale'], expected)


class _Block(nn.Module):
    def __init__(self, d_in: int, d_out: int, pack_size: int) -> None:
        super().__init__()
        self.linear = LinearPack(d_in, d_out, pack_size=pack_size, bias=False)


class _NoBiasModel(nn.Module):
    """Duck-typed ModelPack without biases (to check that empty groups are omitted)."""

    def __init__(self, pack_size: int) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList(
            [_Block(4, 8, pack_size), _Block(8, 8, pack_size)]
        )
        self.head = LinearPack(8, 1, pack_size=pack_size, bias=False)


def test_empty_groups_are_omitted() -> None:
    model: Any = _NoBiasModel(pack_size=3)
    groups = make_param_groups(model, muon=True)
    assert len(groups) == 3
    assert [id(p) for p in groups[0]['params']] == [id(model.head.weight)]
    assert 'weight_decay' not in groups[0]
    assert all(group['muon'] is True for group in groups[1:])
    torch.testing.assert_close(groups[1]['muon_scale'], torch.full((3,), math.sqrt(2)))

    groups = make_param_groups(model, muon=False)
    assert len(groups) == 1
    assert len(groups[0]['params']) == 3


# >>> optimizer_select_: generic state layouts (no dependency on a15/a16)


class _StateLayoutOptimizer(torch.optim.Optimizer):
    """Fills its state with every layout that optimizer_select_ must handle."""

    def __init__(self, params: Any, *, pack_size: int) -> None:
        super().__init__(
            params,
            {
                'lr': torch.arange(pack_size, dtype=torch.float32) + 1.0,
                'weight_decay': 0.5,
                'beta1': 0.9,
                'muon_lr': None,
                'flag': True,
            },
        )
        for i, group in enumerate(self.param_groups):
            group['lr'] = group['lr'] * 10 ** (i + 1)  # its own storage
            for p in group['params']:
                k = p.shape[0]
                self.state[p] = {
                    'exp_avg': torch.randn_like(p),
                    'buffer': torch.randn(k, 2, 3),
                    'step_tensor': torch.arange(k, dtype=torch.int64),
                    'step_int': 7,
                    'step_scalar_tensor': torch.tensor(3.0),
                    'name': 'x',
                }
        # Optimizer-level entries under non-param keys.
        self.state['shared_step'] = 5
        self.state['shared_step_tensor'] = torch.tensor(5)
        self.state['per_member_step'] = torch.arange(pack_size) * 100


def _make_params(pack_size: int) -> list[nn.Parameter]:
    return [
        nn.Parameter(torch.randn(pack_size, 3, 2)),
        nn.Parameter(torch.randn(pack_size, 2)),
        nn.Parameter(torch.randn(pack_size, 5, 1)),
    ]


def _select_params_(params: list[nn.Parameter], keep_idx: Tensor) -> None:
    for p in params:
        p.data = p.data[keep_idx]
        p.grad = None


@pytest.mark.parametrize('keep', [[1, 3], [3, 0, 2], [0, 1, 2, 3], [2]])
def test_optimizer_select_slices_every_state_layout(keep: list[int]) -> None:
    pack_size = 4
    params = _make_params(pack_size)
    optimizer = _StateLayoutOptimizer(
        [{'params': params[:2]}, {'params': params[2:], 'weight_decay': 0.0}],
        pack_size=pack_size,
    )
    groups_before = copy.deepcopy(
        [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups]
    )
    defaults_before = copy.deepcopy(optimizer.defaults)
    state_before = {id(p): copy.deepcopy(optimizer.state[p]) for p in params}
    keep_idx = torch.tensor(keep)

    _select_params_(params, keep_idx)
    optimizer_select_(optimizer, keep_idx)

    # Param groups: the same parameter objects, (K,) tensors sliced, scalars kept.
    assert [id(p) for p in optimizer.param_groups[0]['params']] == [
        id(p) for p in params[:2]
    ]
    for group, before in zip(optimizer.param_groups, groups_before, strict=True):
        assert set(group) - {'params'} == set(before)
        for key, value in before.items():
            if isinstance(value, Tensor):
                assert torch.equal(group[key], value[keep_idx])
            else:
                assert group[key] == value
                assert type(group[key]) is type(value)
    assert torch.equal(optimizer.defaults['lr'], defaults_before['lr'][keep_idx])
    assert optimizer.defaults['weight_decay'] == 0.5
    assert optimizer.defaults['muon_lr'] is None

    # Per-param state.
    for p in params:
        state, before = optimizer.state[p], state_before[id(p)]
        assert set(state) == set(before)
        assert torch.equal(state['exp_avg'], before['exp_avg'][keep_idx])
        assert state['exp_avg'].shape == p.shape
        assert torch.equal(state['buffer'], before['buffer'][keep_idx])
        assert torch.equal(state['step_tensor'], before['step_tensor'][keep_idx])
        assert state['step_tensor'].dtype == torch.int64
        assert state['step_int'] == 7
        assert torch.equal(state['step_scalar_tensor'], torch.tensor(3.0))
        assert state['name'] == 'x'

    # Optimizer-level entries.
    assert optimizer.state['shared_step'] == 5
    assert torch.equal(optimizer.state['shared_step_tensor'], torch.tensor(5))
    assert torch.equal(
        optimizer.state['per_member_step'], (torch.arange(pack_size) * 100)[keep_idx]
    )


def test_optimizer_select_does_not_alias_shared_tensors() -> None:
    params = _make_params(4)
    optimizer = torch.optim.SGD(params, lr=0.1)
    shared = torch.arange(4.0)
    optimizer.param_groups[0]['lr_pack'] = shared
    optimizer.state[params[0]]['buf'] = shared
    keep_idx = torch.tensor([0, 2])
    _select_params_(params, keep_idx)
    optimizer_select_(optimizer, keep_idx)
    lr_pack = optimizer.param_groups[0]['lr_pack']
    buf = optimizer.state[params[0]]['buf']
    assert torch.equal(lr_pack, torch.tensor([0.0, 2.0]))
    assert torch.equal(buf, torch.tensor([0.0, 2.0]))
    lr_pack.add_(1.0)
    assert torch.equal(buf, torch.tensor([0.0, 2.0]))
    assert torch.equal(shared, torch.arange(4.0))


def test_optimizer_select_without_per_member_tensors_is_a_no_op() -> None:
    params = _make_params(4)
    optimizer = torch.optim.SGD(params, lr=0.1, weight_decay=0.01)
    keep_idx = torch.tensor([1, 2])
    _select_params_(params, keep_idx)
    optimizer_select_(optimizer, keep_idx)
    assert optimizer.param_groups[0]['lr'] == 0.1
    assert len(optimizer.state) == 0


def test_optimizer_select_requires_pack_select_first() -> None:
    params = _make_params(4)
    optimizer = _StateLayoutOptimizer(params, pack_size=4)
    with pytest.raises(ValueError, match='pack_select_'):
        optimizer_select_(optimizer, torch.tensor([0, 1]))


def test_optimizer_select_rejects_invalid_indices() -> None:
    params = _make_params(4)
    optimizer = _StateLayoutOptimizer(params, pack_size=4)
    _select_params_(params, torch.tensor([0, 1]))
    with pytest.raises(IndexError):
        optimizer_select_(optimizer, torch.tensor([0, 4]))
    with pytest.raises(IndexError):
        optimizer_select_(optimizer, torch.tensor([-1, 0]))
    with pytest.raises(ValueError, match='integer'):
        optimizer_select_(optimizer, torch.tensor([True, True]))


def test_optimizer_select_rejects_inconsistent_pack_sizes() -> None:
    params = _make_params(4)
    optimizer = _StateLayoutOptimizer(params, pack_size=4)
    optimizer.state[params[0]]['bad'] = torch.zeros(5)
    _select_params_(params, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match='disagree'):
        optimizer_select_(optimizer, torch.tensor([0, 1]))


# >>> optimizer_select_ with the real optimizers: training continues exactly as if
# the removed members had never been in the pack.

PACK_SIZE = 4
N_BLOCKS = [2, 1, 3, 2]
LR = [1e-3, 3e-3, 2e-3, 5e-4]
WEIGHT_DECAY = [0.0, 1e-2, 3e-2, 1e-3]
MUON_LR = [2e-2, 1e-2, 5e-3, 3e-2]
BATCH_SIZE = 16
N_STEPS_BEFORE = 4
N_STEPS_AFTER = 4


def _make_adamw(model: ModelPack, member_idx: Tensor, shared_step: bool) -> Any:
    return AdamWPack(
        make_param_groups(model, muon=False),
        lr=[LR[i] for i in member_idx.tolist()],
        weight_decay=[WEIGHT_DECAY[i] for i in member_idx.tolist()],
        pack_size=model.pack_size,
        shared_step=shared_step,
    )


def _make_muon_adamw(model: ModelPack, member_idx: Tensor, shared_step: bool) -> Any:
    return MuonAdamWPack(
        make_param_groups(model, muon=True),
        lr=[LR[i] for i in member_idx.tolist()],
        weight_decay=[WEIGHT_DECAY[i] for i in member_idx.tolist()],
        muon_lr=[MUON_LR[i] for i in member_idx.tolist()],
        pack_size=model.pack_size,
        shared_step=shared_step,
    )


OptimizerFactory = Callable[[ModelPack, Tensor, bool], torch.optim.Optimizer]


def _make_data(n_steps: int) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Per-member batches (K, B, ...) for every step (members see different rows)."""
    generator = torch.Generator().manual_seed(1)
    data = []
    for _ in range(n_steps):
        x_num = torch.randn(PACK_SIZE, BATCH_SIZE, N_NUM, generator=generator)
        x_cat = torch.stack(
            [
                torch.randint(0, c, (PACK_SIZE, BATCH_SIZE), generator=generator)
                for c in CAT_CARDINALITIES
            ],
            dim=-1,
        )
        y = torch.randint(0, 2, (PACK_SIZE, BATCH_SIZE), generator=generator).float()
        data.append((x_num, x_cat, y))
    return data


def _train_step(
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    batch: tuple[Tensor, Tensor, Tensor],
    member_idx: Tensor,
) -> None:
    x_num, x_cat, y = (x[member_idx] for x in batch)
    optimizer.zero_grad()
    logits = model(x_num, x_cat)
    # Sum over members of per-member mean losses: members are independent.
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction='none'
    ).mean(1)
    loss.sum().backward()
    optimizer.step()


def _assert_nested_equal(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_nested_equal(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list | tuple):
        assert isinstance(actual, list | tuple), path
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_nested_equal(a, e, f'{path}[{i}]')
    elif isinstance(expected, Tensor):
        assert isinstance(actual, Tensor), path
        torch.testing.assert_close(actual, expected, msg=path)
    else:
        assert type(actual) is type(expected), path
        assert actual == expected, path


def _assert_optimizers_equal(
    actual: torch.optim.Optimizer, expected: torch.optim.Optimizer
) -> None:
    """Compare the full state_dicts (per-param state + hyperparameters)."""
    _assert_nested_equal(actual.state_dict(), expected.state_dict(), 'state_dict')


@pytest.mark.parametrize('shared_step', [True, False])
@pytest.mark.parametrize(
    'make_optimizer',
    [_make_adamw, _make_muon_adamw],
    ids=['AdamWPack', 'MuonAdamWPack'],
)
@pytest.mark.parametrize('keep', [[0, 2], [2, 0, 3], [1, 2]])
def test_training_after_removal_matches_a_pack_without_the_removed_members(
    make_optimizer: OptimizerFactory, shared_step: bool, keep: list[int]
) -> None:
    all_idx = torch.arange(PACK_SIZE)
    keep_idx = torch.tensor(keep)
    data = _make_data(N_STEPS_BEFORE + N_STEPS_AFTER)
    initial = _make_model(pack_size=PACK_SIZE, n_blocks=N_BLOCKS)

    # (a) Full pack, members removed after N_STEPS_BEFORE steps.
    model = copy.deepcopy(initial)
    optimizer = make_optimizer(model, all_idx, shared_step)
    for batch in data[:N_STEPS_BEFORE]:
        _train_step(model, optimizer, batch, all_idx)
    pack_select_(model, keep_idx)
    optimizer_select_(optimizer, keep_idx)
    assert model.pack_size == len(keep)
    for batch in data[N_STEPS_BEFORE:]:
        _train_step(model, optimizer, batch, keep_idx)

    # (b) A pack that never contained the removed members.
    model_kept = copy.deepcopy(initial)
    pack_select_(model_kept, keep_idx)
    optimizer_kept = make_optimizer(model_kept, keep_idx, shared_step)
    for batch in data:
        _train_step(model_kept, optimizer_kept, batch, keep_idx)

    # (c) The full pack without removal, sliced at the end.
    model_full = copy.deepcopy(initial)
    optimizer_full = make_optimizer(model_full, all_idx, shared_step)
    for batch in data:
        _train_step(model_full, optimizer_full, batch, all_idx)

    params = dict(model.named_parameters())
    params_kept = dict(model_kept.named_parameters())
    params_full = dict(model_full.named_parameters())
    assert params.keys() == params_kept.keys() == params_full.keys()
    for name, p in params.items():
        torch.testing.assert_close(p, params_kept[name], msg=name)
        torch.testing.assert_close(p, params_full[name][keep_idx], msg=name)
        # Training moved the parameters (the check is not vacuous).
        assert not torch.equal(p, dict(initial.named_parameters())[name][keep_idx])
    _assert_optimizers_equal(optimizer, optimizer_kept)


@pytest.mark.parametrize(
    'make_optimizer',
    [_make_adamw, _make_muon_adamw],
    ids=['AdamWPack', 'MuonAdamWPack'],
)
def test_repeated_removal(make_optimizer: OptimizerFactory) -> None:
    data = _make_data(9)
    initial = _make_model(pack_size=PACK_SIZE, n_blocks=[3, 3, 3, 3])
    model = copy.deepcopy(initial)
    member_idx = torch.arange(PACK_SIZE)
    optimizer = make_optimizer(model, member_idx, True)
    for i, batch in enumerate(data):
        if i in (3, 6):
            local_keep = torch.tensor([0, 2]) if i == 3 else torch.tensor([1])
            pack_select_(model, local_keep)
            optimizer_select_(optimizer, local_keep)
            member_idx = member_idx[local_keep]
        _train_step(model, optimizer, batch, member_idx)
    assert member_idx.tolist() == [2]

    model_full = copy.deepcopy(initial)
    optimizer_full = make_optimizer(model_full, torch.arange(PACK_SIZE), True)
    for batch in data:
        _train_step(model_full, optimizer_full, batch, torch.arange(PACK_SIZE))
    for (name, p), p_full in zip(
        model.named_parameters(), model_full.parameters(), strict=True
    ):
        torch.testing.assert_close(p, p_full[member_idx], msg=name)
