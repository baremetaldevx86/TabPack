"""Numerical parity of our optimizers with the official ones (a40).

Compared against the official clone (imported via the ``official`` fixture):

* ``optim.zeropower_via_newtonschulz5`` vs ``vendor.muon.zeropower_via_newtonschulz5``;
* ``optim.AdamWPack`` vs ``project.optim.AdamWPack``;
* ``optim.MuonAdamWPack`` vs ``project.optim.MuonAdamWPack``;
* ``optim.make_param_groups`` vs the grouping of ``project.tabpack`` (one Muon group
  per backbone block + ``lib.optim.utils.make_parameter_groups`` with
  ``_default_zero_weight_decay_condition``), and ``optimizer_select_`` vs
  ``optimizer_pack_remove`` in an end-to-end run.

Layout: our LinearPack weight is (K, in, out), the official ParameterPack weight is
(K, out, in). Every pair of parameters starts from the same values and receives the
same gradients, transposed consistently. The gradients are fixed (not computed from
the parameters), so the optimizer states (Adam moments, Muon momentum buffers) do not
depend on the parameters: they must be bit-identical, and any rounding difference in
the parameters adds up over the steps without feedback (weight decay only shrinks
it). This makes the "N steps x per-step bound" tolerances below valid.

Where the arithmetic is literally the same the tests demand ``torch.equal``:
Newton-Schulz on the same matrix, and AdamW with Python-float lr / weight decay and
``shared_step=True``. Elsewhere the two implementations round differently, on purpose:

* AdamW with per-member (K,) lr / wd: the official code computes ``1 - lr * wd`` and
  ``lr / bias_correction1`` in float32 and applies ``(m / d) * step``; ours does the
  (K,) scalar math in float64 and applies ``(m * step) / d``. That is a few float32
  roundings per step on quantities bounded by max|p|.
* Muon: Newton-Schulz returns bf16. The official code multiplies that bf16 tensor in
  place by the spectral scale and (for per-member lr) by ``muon_lr``, so it rounds
  the update to bf16 up to twice. Ours casts to float32 first (a16's documented
  choice). NS entries are bounded by the spectral norm of the NS output, which is
  < 1.25 for the quintic used (``_NS_MAX_ENTRY = 1.5`` leaves a margin).
* Square Muon weights: NS(G^T)^T equals NS(G) only in exact arithmetic. Iterating on
  X^T X instead of X X^T changes the bf16 rounding, and the NS polynomial amplifies
  it. Non-square weights do not have the problem: NS moves both layouts to the same
  wide matrix, so the results are bit-identical.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from typing import Any, NamedTuple

import pytest
import torch
from torch import Tensor, nn

from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.nn.pack_ops import make_keep_idx, pack_select_
from tabpack_repro.optim import (
    AdamWPack,
    MuonAdamWPack,
    make_param_groups,
    optimizer_select_,
    zeropower_via_newtonschulz5,
)

K = 4
N_STEPS = 10
EPS32 = torch.finfo(torch.float32).eps  # 2**-23
BF16_UNIT_ROUNDOFF = torch.finfo(torch.bfloat16).eps / 2  # 2**-8
# Bound on |entry| of a Newton-Schulz output (spectral norm < 1.25, see docstring).
_NS_MAX_ENTRY = 1.5

# Per-member hyperparameters in the ranges of the official Churn search space
# (lr in [1e-4, 5e-3], weight_decay in [1e-3, 1], muon_lr in [1e-3, 0.1]); one
# member has wd=0 as a value inside a (K,) tensor (not the "skip" float 0.0).
LR = [1e-3, 3e-3, 5e-4, 2e-3]
WD = [0.0, 1e-3, 0.3, 1.0]
MUON_LR = [1e-2, 3e-2, 1e-3, 0.1]
# The same hyperparameters as Python floats (same value for every member).
LR_F, WD_F, MUON_LR_F = 3e-3, 0.1, 0.02


# >>> Helpers.


@pytest.fixture(scope='module')
def ns_official(official):
    return official('vendor.muon').zeropower_via_newtonschulz5


@pytest.fixture(scope='module')
def optim_off(official):
    return official('project.optim')


@pytest.fixture(scope='module')
def nn_off(official):
    return official('project.nn')


def _stub_module(name: str) -> types.ModuleType:
    """A stand-in module whose every attribute is a fresh (never matching) class."""
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: type(attr, (), {})  # type: ignore[attr-defined]
    return module


@pytest.fixture
def tabpack_official(official, monkeypatch):
    """(project.tabpack, lib.optim.utils); stubs the rtdl packages if missing.

    lib.optim.utils uses the rtdl embedding classes only in isinstance checks (our
    models have no embeddings), so empty stand-ins are enough when they are not
    installed. The installed packages are never replaced.
    """
    for name in ('rtdl_num_embeddings', 'rtdl_revisiting_models'):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            monkeypatch.setitem(sys.modules, name, _stub_module(name))
    return official('project.tabpack'), official('lib.optim.utils')


class _Pair(NamedTuple):
    """One parameter in both implementations; ``transpose``: layouts differ."""

    ours: nn.Parameter
    off: nn.Parameter  # an official ParameterPack
    transpose: bool

    def ours_as_off(self) -> Tensor:
        """Our parameter in the official layout."""
        p = self.ours.detach()
        return p.mT if self.transpose else p


def _make_pair(nn_off, off_value: Tensor, *, transpose: bool) -> _Pair:
    """``off_value`` is in the official layout; ours gets the (transposed) copy."""
    ours = (off_value.mT if transpose else off_value).contiguous().clone()
    return _Pair(nn.Parameter(ours), nn_off.ParameterPack(off_value.clone()), transpose)


def _step_both(
    ours_opt: torch.optim.Optimizer,
    off_opt: torch.optim.Optimizer,
    pairs: list[_Pair],
    n_steps: int,
    generator: torch.Generator,
) -> None:
    """``n_steps`` optimizer steps with the same fresh random gradients in both."""
    for _ in range(n_steps):
        for pair in pairs:
            g = torch.randn(pair.off.shape, generator=generator)
            # A fresh tensor each step: the official Nesterov step clobbers p.grad.
            pair.off.grad = g.clone()
            pair.ours.grad = (g.mT if pair.transpose else g).contiguous()
        ours_opt.step()
        off_opt.step()


def _state_in_off_layout(state: Tensor, pair: _Pair) -> Tensor:
    return state.mT if pair.transpose else state


def _max_abs(pairs: list[_Pair]) -> float:
    return max(pair.off.detach().abs().max().item() for pair in pairs)


def _fp32_atol(n_steps: int, max_abs_p: float) -> float:
    """Tolerance for float32-only rounding differences after ``n_steps`` steps.

    Each step does O(1) float32 roundings, on quantities bounded by max|p| (the
    parameter and, here, the Adam / Muon update), and the two codes may round each of
    them differently. 8 roundings per step is a generous bound. The differences add
    up without feedback (see the module docstring).
    """
    return 8 * n_steps * EPS32 * max(1.0, max_abs_p)


def _muon_atol(
    n_steps: int, lr: Tensor, scale: Tensor, *, bf16_rounded: bool, max_abs_p: float
) -> Tensor:
    """Per-member (K,) tolerance for a Muon weight after ``n_steps`` steps.

    The Muon update of member k is ``lr_k * scale_k * O`` with |O_ij| <=
    _NS_MAX_ENTRY. When the official code rounds it to bf16 (``bf16_rounded``: a
    tensor scale, a Python scale != 1 or a per-member lr), that is at most two bf16
    roundings (relative error 2 * 2**-8) per step, which ours skips. Otherwise only
    float32 differences remain: ``p.sub_(u, alpha=lr)`` (fused) vs ``p.sub_(u * lr)``.
    """
    rel = 2 * BF16_UNIT_ROUNDOFF if bf16_rounded else 2 * EPS32
    per_step = lr * scale * _NS_MAX_ENTRY * rel
    return n_steps * per_step + _fp32_atol(n_steps, max_abs_p)


def _assert_close_per_member(actual: Tensor, expected: Tensor, atol: Tensor) -> None:
    diff = (actual - expected).abs().flatten(1).amax(1)
    assert bool((diff <= atol).all()), (
        f'max |diff| per member {diff.tolist()} exceeds tolerance {atol.tolist()}'
    )


# >>> Newton-Schulz.

NS_SHAPES = [
    (K, 6, 16),  # wide
    (K, 16, 6),  # tall (transposed internally)
    (K, 16, 16),  # square
    (16, 6),  # a single matrix
    (2, 3, 8, 5),  # two batch dims
    (K, 1, 7),  # rank one
]


@pytest.mark.parametrize('steps', [0, 1, 5])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shape', NS_SHAPES)
def test_ns_bit_identical_to_official(ns_official, shape, dtype, steps):
    """Same op order as vendor/muon.py (a14 #47), so results are bit-identical."""
    G = torch.randn(shape, generator=torch.Generator().manual_seed(0)).to(dtype)
    ours = zeropower_via_newtonschulz5(G, steps)
    expected = ns_official(G, steps)
    assert ours.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(ours, expected)


def test_ns_zero_matrix_matches_official(ns_official):
    G = torch.zeros(K, 6, 16)
    assert torch.equal(zeropower_via_newtonschulz5(G, 5), ns_official(G, 5))


@pytest.mark.parametrize('shape', [s for s in NS_SHAPES if len(s) >= 3])
def test_ns_batched_equals_official_per_matrix(ns_official, shape):
    """Our batched call == the official one applied to every matrix separately."""
    G = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    ours = zeropower_via_newtonschulz5(G, 5)
    per_matrix = torch.stack(
        [ns_official(g, 5) for g in G.reshape(-1, *G.shape[-2:])]
    ).reshape(ours.shape)
    assert torch.equal(ours, per_matrix)


@pytest.mark.parametrize(
    'off_shape', [(K, 16, 6), (K, 6, 16), (K, 12, 3), (K, 384, 20)]
)
def test_ns_our_layout_bit_identical_for_non_square(ns_official, off_shape):
    """NS on our (K, in, out) layout == the official (K, out, in) result, transposed.

    NS iterates on the wide orientation of every matrix, so for non-square matrices
    both layouts run on the very same matrix.
    """
    G = torch.randn(off_shape, generator=torch.Generator().manual_seed(2))
    ours = zeropower_via_newtonschulz5(G.mT.contiguous(), 5).mT
    assert torch.equal(ours, ns_official(G, 5))


@pytest.mark.parametrize('n', [16, 64])
def test_ns_our_layout_close_for_square(ns_official, n):
    """Square matrices: NS(G^T)^T vs NS(G) differ by amplified bf16 rounding only.

    Measured about 4% relative Frobenius difference per matrix for Gaussian inputs
    (after 1 step it is ~2e-3, the NS polynomial amplifies it). 10% is the bound we
    accept. This is the only source of the loose tolerance for square Muon weights.
    """
    G = torch.randn(8, n, n, generator=torch.Generator().manual_seed(3))
    ours = zeropower_via_newtonschulz5(G.mT.contiguous(), 5).mT.float()
    expected = ns_official(G, 5).float()
    rel = torch.linalg.matrix_norm(ours - expected) / torch.linalg.matrix_norm(expected)
    assert bool((rel < 0.1).all()), rel.tolist()
    assert not torch.equal(ours, expected)  # really a different rounding path


# >>> AdamWPack.


def _adamw_pairs(nn_off, seed: int) -> list[_Pair]:
    gen = torch.Generator().manual_seed(seed)
    return [
        _make_pair(nn_off, torch.randn(K, 16, 6, generator=gen), transpose=True),
        _make_pair(nn_off, torch.randn(K, 1, 16, generator=gen), transpose=True),
        _make_pair(nn_off, torch.randn(K, 16, generator=gen), transpose=False),
    ]


def _run_adamw(optim_off, nn_off, *, per_member: bool, shared_step: bool):
    pairs = _adamw_pairs(nn_off, seed=10)
    hparams: dict[str, Any] = (
        {'lr': LR, 'weight_decay': WD}
        if per_member
        else {'lr': LR_F, 'weight_decay': WD_F}
    )

    def groups(which: str) -> list[dict[str, Any]]:
        weights = [getattr(p, which) for p in pairs[:2]]
        return [
            {'params': weights},
            {'params': [getattr(pairs[2], which)], 'weight_decay': 0.0},
        ]

    kwargs = {**hparams, 'pack_size': K, 'shared_step': shared_step}
    ours_opt = AdamWPack(groups('ours'), **kwargs)
    off_opt = optim_off.AdamWPack(groups('off'), **kwargs)
    initial = [pair.off.detach().clone() for pair in pairs]
    _step_both(ours_opt, off_opt, pairs, N_STEPS, torch.Generator().manual_seed(11))
    return pairs, ours_opt, off_opt, initial


def _assert_adam_moments_equal(ours_opt, off_opt, pairs: list[_Pair]) -> None:
    for pair in pairs:
        for key in ('exp_avg', 'exp_avg_sq'):
            ours = _state_in_off_layout(ours_opt.state[pair.ours][key], pair)
            assert torch.equal(ours, off_opt.state[pair.off][key]), key


def test_adamw_float_hparams_bit_identical(optim_off, nn_off):
    """Python-float lr / wd and a shared int step: literally the official arithmetic."""
    pairs, ours_opt, off_opt, _ = _run_adamw(
        optim_off, nn_off, per_member=False, shared_step=True
    )
    _assert_adam_moments_equal(ours_opt, off_opt, pairs)
    for pair in pairs:
        assert torch.equal(pair.ours_as_off(), pair.off.detach())


@pytest.mark.parametrize(
    ('per_member', 'shared_step'), [(True, True), (True, False), (False, False)]
)
def test_adamw_matches_official(optim_off, nn_off, per_member, shared_step):
    """Per-member (K,) hparams and/or per-param (K,) steps: float32 rounding only.

    With per-member values or a (K,) step tensor the official code does the (K,)
    scalar math (1 - lr * wd, lr / bias_correction1, bias_correction2 ** 0.5) in
    float32, ours in float64 (see the module docstring). The moments do not depend
    on lr / wd / step and stay bit-identical.
    """
    pairs, ours_opt, off_opt, initial = _run_adamw(
        optim_off, nn_off, per_member=per_member, shared_step=shared_step
    )
    _assert_adam_moments_equal(ours_opt, off_opt, pairs)
    atol = _fp32_atol(N_STEPS, _max_abs(pairs))
    for pair, start in zip(pairs, initial, strict=True):
        torch.testing.assert_close(
            pair.ours_as_off(), pair.off.detach(), rtol=0.0, atol=atol
        )
        # The tolerance is far below how far the parameters moved.
        assert atol < 0.01 * (pair.off.detach() - start).abs().max()


# >>> MuonAdamWPack.


def _run_muon(
    optim_off,
    nn_off,
    muon_off_value: Tensor,
    *,
    transpose: bool,
    per_member: bool,
    scale_mode: str,
    nesterov: bool = True,
    n_steps: int = N_STEPS,
):
    """A Muon weight + an AdamW head weight + an AdamW bias (wd=0 group).

    ``muon_off_value`` is the Muon weight as the official optimizer sees it;
    ``transpose=True`` gives ours its transpose (our real layout).
    scale_mode: 'default' (no scale in the groups: both use sqrt(max(1, out/in))),
    or 'tensor' (ours 'muon_scale' == official 'muon_update_scale', a (K,) tensor).
    Returns (pairs, ours_opt, off_opt, lr (K,), scale (K,)).
    """
    gen = torch.Generator().manual_seed(20)
    out_f, in_f = muon_off_value.shape[-2:]  # official orientation
    pairs = [
        _make_pair(nn_off, muon_off_value, transpose=transpose),
        _make_pair(nn_off, torch.randn(K, 1, out_f, generator=gen), transpose=True),
        _make_pair(nn_off, torch.randn(K, out_f, generator=gen), transpose=False),
    ]
    scale = torch.full((K,), math.sqrt(max(1.0, out_f / in_f)))
    muon_ours: dict[str, Any] = {'muon': True}
    muon_off: dict[str, Any] = {'muon': True}
    if scale_mode == 'tensor':
        muon_ours['muon_scale'] = scale.clone()
        muon_off['muon_update_scale'] = scale.clone()
    else:
        assert scale_mode == 'default'

    def groups(which: str, muon_group: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {'params': [pairs[1].ours if which == 'ours' else pairs[1].off]},
            {
                'params': [pairs[2].ours if which == 'ours' else pairs[2].off],
                'weight_decay': 0.0,
            },
            {
                'params': [pairs[0].ours if which == 'ours' else pairs[0].off],
                **muon_group,
            },
        ]

    hparams: dict[str, Any] = (
        {'lr': LR, 'weight_decay': WD, 'muon_lr': MUON_LR}
        if per_member
        else {'lr': LR_F, 'weight_decay': WD_F, 'muon_lr': MUON_LR_F}
    )
    kwargs = {**hparams, 'muon_nesterov': nesterov, 'pack_size': K, 'shared_step': True}
    ours_opt = MuonAdamWPack(groups('ours', muon_ours), **kwargs)
    off_opt = optim_off.MuonAdamWPack(groups('off', muon_off), **kwargs)
    _step_both(ours_opt, off_opt, pairs, n_steps, torch.Generator().manual_seed(21))
    lr = torch.tensor(MUON_LR if per_member else [MUON_LR_F] * K, dtype=torch.float64)
    return pairs, ours_opt, off_opt, lr, scale.double()


def _assert_muon_states_equal(ours_opt, off_opt, pairs: list[_Pair]) -> None:
    """Momentum buffers and Adam moments: elementwise ops only, bit-identical."""
    muon, *adamw = pairs
    key = 'muon_momentum_buffer'
    ours_buf = _state_in_off_layout(ours_opt.state[muon.ours][key], muon)
    assert torch.equal(ours_buf, off_opt.state[muon.off][key])
    _assert_adam_moments_equal(ours_opt, off_opt, adamw)


def _assert_adamw_pairs_close(pairs: list[_Pair], n_steps: int, exact: bool) -> None:
    for pair in pairs:
        if exact:
            assert torch.equal(pair.ours_as_off(), pair.off.detach())
        else:
            atol = _fp32_atol(n_steps, _max_abs(pairs))
            torch.testing.assert_close(
                pair.ours_as_off(), pair.off.detach(), rtol=0.0, atol=atol
            )


@pytest.mark.parametrize('nesterov', [True, False])
@pytest.mark.parametrize('scale_mode', ['default', 'tensor'])
@pytest.mark.parametrize('per_member', [False, True])
@pytest.mark.parametrize(
    'off_shape',
    [(K, 16, 6), (K, 6, 16), (K, 12, 3)],
    ids=['scale_sqrt8_3', 'scale_1', 'scale_2'],
)
def test_muon_matches_official_non_square(
    optim_off, nn_off, off_shape, per_member, scale_mode, nesterov
):
    """Official (K, out, in) vs our (K, in, out): per-member lr/wd/muon_lr, shared step.

    NS is bit-identical here (non-square); the remaining difference is the bf16
    rounding of the scaled update in the official code (see `_muon_atol`). With a
    Python-float muon_lr and scale 1 the official code does not round to bf16 and the
    bound drops to float32 level.
    """
    value = 0.3 * torch.randn(off_shape, generator=torch.Generator().manual_seed(22))
    pairs, ours_opt, off_opt, lr, scale = _run_muon(
        optim_off,
        nn_off,
        value,
        transpose=True,
        per_member=per_member,
        scale_mode=scale_mode,
        nesterov=nesterov,
    )
    _assert_muon_states_equal(ours_opt, off_opt, pairs)
    _assert_adamw_pairs_close(pairs[1:], N_STEPS, exact=not per_member)

    # Multiplying by exactly 1.0 does not round, even in bf16.
    bf16_rounded = per_member or bool((scale != 1).any())
    atol = _muon_atol(
        N_STEPS, lr, scale, bf16_rounded=bf16_rounded, max_abs_p=_max_abs(pairs)
    )
    muon = pairs[0]
    _assert_close_per_member(
        muon.ours_as_off().double(), muon.off.detach().double(), atol
    )
    # The tolerance is well below how far the weights moved.
    moved = (muon.off.detach() - value).abs().flatten(1).amax(1).double()
    assert bool((atol < 0.1 * moved).all())


@pytest.mark.parametrize('per_member', [False, True])
def test_muon_square_same_orientation_matches_official(optim_off, nn_off, per_member):
    """Square weights, official optimizer fed our (K, in, out) tensor as is.

    For a square weight the default scale is 1 in both layouts, so running the
    official optimizer on our orientation is legitimate and NS then sees the same
    matrix: the tight (non-square) bound holds. Together with the loose test below,
    this shows that the only square-specific difference is the NS orientation.
    """
    value = 0.3 * torch.randn(K, 16, 16, generator=torch.Generator().manual_seed(23))
    pairs, ours_opt, off_opt, lr, scale = _run_muon(
        optim_off,
        nn_off,
        value,
        transpose=False,
        per_member=per_member,
        scale_mode='default',
    )
    _assert_muon_states_equal(ours_opt, off_opt, pairs)
    atol = _muon_atol(
        N_STEPS, lr, scale, bf16_rounded=per_member, max_abs_p=_max_abs(pairs)
    )
    muon = pairs[0]
    _assert_close_per_member(
        muon.ours_as_off().double(), muon.off.detach().double(), atol
    )


def _assert_displacement_close(
    ours: Tensor, off: Tensor, initial: Tensor, rel: float
) -> None:
    """Per member: ||d_ours - d_off||_F <= rel * ||d_off||_F (d = p - p_initial)."""
    d_ours = (ours - initial).flatten(1)
    d_off = (off - initial).flatten(1)
    err = (d_ours - d_off).norm(dim=1) / d_off.norm(dim=1)
    assert bool((err <= rel).all()), f'relative displacement error {err.tolist()}'


@pytest.mark.parametrize('per_member', [False, True])
def test_muon_square_official_layout_close(optim_off, nn_off, per_member):
    """Square weights in the real layouts: NS orientation differs (bf16 rounding).

    Measured ~2-3% relative error of the parameter displacement; we accept 10%, the
    per-call bound of `test_ns_our_layout_close_for_square`. States stay bit-exact.
    """
    value = 0.3 * torch.randn(K, 16, 16, generator=torch.Generator().manual_seed(24))
    pairs, ours_opt, off_opt, *_ = _run_muon(
        optim_off,
        nn_off,
        value,
        transpose=True,
        per_member=per_member,
        scale_mode='default',
    )
    _assert_muon_states_equal(ours_opt, off_opt, pairs)
    _assert_adamw_pairs_close(pairs[1:], N_STEPS, exact=not per_member)
    muon = pairs[0]
    _assert_displacement_close(muon.ours_as_off(), muon.off.detach(), value, rel=0.1)


# >>> Param groups and an end-to-end run on equivalent models.

N_NUM = 3
CARDS = [3, 2]  # d_in = 3 + 5 = 8: the first block (8 -> 16) is non-square
D_BLOCK = 16
N_BLOCKS = [1, 3, 2, 3]
DROPOUT = [0.0, 0.1, 0.2, 0.3]


def _off_name(name: str) -> str:
    """Our parameter name -> the official one (the head is called 'output')."""
    return 'output.' + name.removeprefix('head.') if name.startswith('head.') else name


def _make_models(tabpack_off) -> tuple[ModelPack, nn.Module]:
    """Our ModelPack and the official one with the same (transposed) weights."""
    with torch.random.fork_rng():
        torch.manual_seed(30)
        return _make_models_impl(tabpack_off)


def _make_models_impl(tabpack_off) -> tuple[ModelPack, nn.Module]:
    ours = ModelPack(
        n_num_features=N_NUM,
        cat_cardinalities=CARDS,
        n_classes=None,
        pack_size=K,
        d_block=D_BLOCK,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    )
    off = tabpack_off.ModelPack(
        n_num_features=N_NUM,
        cat_cardinalities=CARDS,
        n_classes=None,
        pack_size=K,
        n_blocks=N_BLOCKS,
        max_n_blocks=max(N_BLOCKS),
        d_block=D_BLOCK,
        dropout=DROPOUT,
        activation='ReLU',
    )
    off_params = dict(off.named_parameters())
    assert sorted(map(_off_name, dict(ours.named_parameters()))) == sorted(off_params)
    with torch.no_grad():
        for name, p in ours.named_parameters():
            q = off_params[_off_name(name)]
            q.copy_(p.mT if p.ndim == 3 else p)
    return ours, off


def _official_groups(tabpack_off, optim_utils_off, model, *, muon: bool):
    """The official grouping, as in project/tabpack.py."""
    custom = (
        [
            {
                'params': [block.linear.weight],
                'muon': True,
                'muon_scale': tabpack_off._make_muon_scale(block.linear),
            }
            for block in model.backbone._iter_blocks()
        ]
        if muon
        else []
    )
    return optim_utils_off.make_parameter_groups(
        model, tabpack_off._default_zero_weight_decay_condition, custom_groups=custom
    )


def _describe(groups: list[dict[str, Any]], names: dict[int, str]) -> list[dict]:
    """Groups as comparable data: parameter names + every non-param key."""
    return [
        {
            'params': sorted(names[id(p)] for p in group['params']),
            **{k: v for k, v in group.items() if k != 'params'},
        }
        for group in groups
        if group['params']
    ]


@pytest.mark.parametrize('muon', [True, False])
def test_param_groups_match_official(tabpack_official, muon):
    """Same groups in the same order: default, weight_decay=0.0, one Muon per block."""
    tabpack_off, optim_utils_off = tabpack_official
    ours, off = _make_models(tabpack_off)
    ours_names = {id(p): _off_name(n) for n, p in ours.named_parameters()}
    off_names = {id(p): n for n, p in off.named_parameters()}

    ours_groups = _describe(make_param_groups(ours, muon=muon), ours_names)
    off_groups = _describe(
        _official_groups(tabpack_off, optim_utils_off, off, muon=muon), off_names
    )
    assert [g['params'] for g in ours_groups] == [g['params'] for g in off_groups]
    assert len(ours_groups) == 2 + (max(N_BLOCKS) if muon else 0)
    for ours_g, off_g in zip(ours_groups, off_groups, strict=True):
        assert ours_g.keys() == off_g.keys()
        for key in ours_g.keys() - {'params'}:
            if isinstance(off_g[key], Tensor):
                # muon_scale: float32 (K,), computed the same way -> bit-identical.
                assert ours_g[key].dtype == off_g[key].dtype == torch.float32
                assert torch.equal(ours_g[key], off_g[key]), key
            else:
                assert type(ours_g[key]) is type(off_g[key]), key
                assert ours_g[key] == off_g[key], key


def test_official_ignores_muon_scale_but_default_is_equal(tabpack_official):
    """The official optimizer reads 'muon_update_scale', tabpack.py sets 'muon_scale'.

    So the official run uses the default ``max(1, out/in) ** 0.5`` of the (padded)
    weight shape. Our MuonAdamWPack honors 'muon_scale'. For the fixed-width MLPs used
    here both are the same number (up to float32 vs float64 sqrt rounding).
    """
    tabpack_off, optim_utils_off = tabpack_official
    optim_off = importlib.import_module('project.optim')
    ours, off = _make_models(tabpack_off)
    off_opt = optim_off.MuonAdamWPack(
        _official_groups(tabpack_off, optim_utils_off, off, muon=True),
        lr=LR,
        weight_decay=WD,
        muon_lr=MUON_LR,
        pack_size=K,
        shared_step=True,
    )
    ours_groups = [g for g in make_param_groups(ours, muon=True) if g.get('muon')]
    off_groups = [g for g in off_opt.param_groups if g['muon']]
    for ours_g, off_g in zip(ours_groups, off_groups, strict=True):
        assert off_g['muon_update_scale'] is None  # the key the optimizer reads
        out_f, in_f = off_g['params'][0].shape[-2:]
        default = torch.full((K,), math.sqrt(max(1, out_f / in_f)))
        torch.testing.assert_close(ours_g['muon_scale'], default, rtol=EPS32, atol=0)


def test_model_optimizer_parity_with_member_removal(tabpack_official, optim_off):
    """Equivalent models, official grouping vs ours, Churn-like MuonAdamWPack setup.

    Per-member lr / wd / muon_lr lists, shared_step=True, synthetic gradients.
    4 steps, remove members 1 and 3 (official module_pack_remove +
    optimizer_pack_remove; ours pack_select_ + optimizer_select_), 4 more steps.
    Hyperparameter groups and states must match exactly, parameters within the
    bounds of the focused tests above: non-square Muon (block 0) and AdamW params
    tightly, square Muon (hidden blocks) by the NS-orientation bound.
    """
    tabpack_off, optim_utils_off = tabpack_official
    nn_off = importlib.import_module('project.nn')
    ours, off = _make_models(tabpack_off)
    initial = {n: p.detach().clone() for n, p in ours.named_parameters()}
    kwargs: dict[str, Any] = {
        'lr': LR,
        'weight_decay': WD,
        'muon_lr': MUON_LR,
        'pack_size': K,
        'shared_step': True,
    }
    ours_opt = MuonAdamWPack(make_param_groups(ours, muon=True), **kwargs)
    off_opt = optim_off.MuonAdamWPack(
        _official_groups(tabpack_off, optim_utils_off, off, muon=True), **kwargs
    )

    def pairs() -> list[tuple[str, _Pair]]:
        off_params = dict(off.named_parameters())
        return [
            (n, _Pair(p, off_params[_off_name(n)], p.ndim == 3))
            for n, p in ours.named_parameters()
        ]

    def check_groups() -> None:
        for ours_g, off_g in zip(
            ours_opt.param_groups, off_opt.param_groups, strict=True
        ):
            for key in ('lr', 'weight_decay', 'muon_lr'):
                if isinstance(off_g[key], Tensor):
                    assert torch.equal(ours_g[key], off_g[key]), key
                else:
                    assert ours_g[key] == off_g[key], key

    def check_states() -> None:
        for _, pair in pairs():
            ours_state = ours_opt.state[pair.ours]
            off_state = off_opt.state[pair.off]
            keys = [k for k in off_state if k != 'step']
            assert sorted(keys) == sorted(ours_state)
            for key in keys:
                expected = off_state[key]
                assert torch.equal(
                    _state_in_off_layout(ours_state[key], pair), expected
                )

    check_groups()
    gen = torch.Generator().manual_seed(31)
    _step_both(ours_opt, off_opt, [p for _, p in pairs()], 4, gen)
    check_states()

    remove_idx = torch.tensor([1, 3])
    keep_idx = make_keep_idx(K, remove_idx)
    pack_select_(ours, keep_idx)
    optimizer_select_(ours_opt, keep_idx)
    old_to_new = nn_off.module_pack_remove(off, remove_idx)
    optim_off.optimizer_pack_remove(off_opt, remove_idx, old_to_new)
    check_groups()
    check_states()

    _step_both(ours_opt, off_opt, [p for _, p in pairs()], 4, gen)
    check_states()
    n_steps = 8

    all_pairs = [p for _, p in pairs()]
    max_abs_p = _max_abs(all_pairs)
    lr = torch.tensor(MUON_LR, dtype=torch.float64)[keep_idx]
    for name, pair in pairs():
        ours_p, off_p = pair.ours_as_off(), pair.off.detach()
        start = initial[name][keep_idx]
        start = start.mT if pair.transpose else start
        if name.endswith('linear.weight') and ours_p.shape[-1] != ours_p.shape[-2]:
            out_f, in_f = off_p.shape[-2:]
            scale = torch.full((len(keep_idx),), math.sqrt(max(1, out_f / in_f)))
            atol = _muon_atol(
                n_steps, lr, scale.double(), bf16_rounded=True, max_abs_p=max_abs_p
            )
            _assert_close_per_member(ours_p.double(), off_p.double(), atol)
        elif name.endswith('linear.weight'):
            _assert_displacement_close(ours_p, off_p, start, rel=0.1)
        else:
            torch.testing.assert_close(
                ours_p, off_p, rtol=0.0, atol=_fp32_atol(n_steps, max_abs_p)
            )
        assert not torch.equal(off_p, start), name  # every parameter was trained
