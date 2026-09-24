"""Tests for MuonAdamWPack (a16).

The Muon part is checked against a naive per-member reference written here: for
every member k it keeps an nn.Linear-layout (out, in) matrix, i.e. the official /
Keller Jordan layout, and applies the Muon update with 2-D matrices only. Our
LinearPack weights are (K, in, out), so the packed weight is compared with the
transposed reference matrices.

Newton-Schulz runs in bfloat16. For tall/wide matrices both layouts orthogonalize
the very same (wide) matrix, so packed and reference agree to fp32 rounding. For
square matrices NS(G^T) = NS(G)^T only in exact arithmetic: the packed code
iterates with X^T X where the reference uses X X^T, which differs by bf16 rounding
(amplified by the NS polynomial), so square shapes are compared exactly against a
reference that runs NS on the transpose, and loosely against the official one.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import Tensor, nn

from tabpack_repro.optim.adamw_pack import AdamWPack
from tabpack_repro.optim.muon_adamw_pack import MuonAdamWPack

K = 3
N_STEPS = 5


# >>> Naive per-member reference (2-D matrices, nn.Linear (out, in) layout).


def _ns_reference(G: Tensor, steps: int) -> Tensor:
    """Keller Jordan's Newton-Schulz orthogonalization of ONE 2-D matrix."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = G.size(0) > G.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if tall:
        X = X.T
    return X


class _NaiveMuonMember:
    """Muon for a single (out, in) matrix, written with plain 2-D arithmetic."""

    def __init__(
        self,
        weight: Tensor,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        scale: float | None = None,
        ns_on_transpose: bool = False,
    ) -> None:
        self.weight = weight.clone()
        self.buf = torch.zeros_like(weight)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.ns_on_transpose = ns_on_transpose
        n_out, n_in = weight.shape
        # The spectral scale of Keller Jordan's Muon: rows / cols of (out, in).
        self.scale = math.sqrt(max(1.0, n_out / n_in)) if scale is None else scale

    def step(self, grad: Tensor) -> None:
        self.weight = self.weight * (1.0 - self.lr * self.weight_decay)
        self.buf = self.momentum * self.buf + (1.0 - self.momentum) * grad
        update = (
            (1.0 - self.momentum) * grad + self.momentum * self.buf
            if self.nesterov
            else self.buf
        )
        if self.ns_on_transpose:
            update = _ns_reference(update.T, self.ns_steps).T.float()
        else:
            update = _ns_reference(update, self.ns_steps).float()
        self.weight = self.weight - self.lr * self.scale * update


def _make_muon_param(n_in: int, n_out: int, *, seed: int = 0) -> nn.Parameter:
    g = torch.Generator().manual_seed(seed)
    bound = n_in**-0.5
    w = (torch.rand(K, n_in, n_out, generator=g) * 2 - 1) * bound
    return nn.Parameter(w)


def _grads(shape: tuple[int, ...], n_steps: int, *, seed: int = 1) -> list[Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g) for _ in range(n_steps)]


LR = [1e-3, 5e-3, 2e-3]
WEIGHT_DECAY = [0.0, 0.1, 0.5]
MUON_LR = [0.02, 0.05, 0.01]

# (in, out) shapes that are not square (see the module docstring).
NON_SQUARE_SHAPES = [(4, 12), (12, 4), (5, 7)]
# Tolerance for "same computation up to fp32 rounding" (in practice bit-identical).
EXACT_ATOL = 1e-5


def _run_packed(
    p: nn.Parameter,
    grads: list[Tensor],
    *,
    group_extra: dict | None = None,
    **optimizer_kwargs,
) -> MuonAdamWPack:
    kwargs = {
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'muon_lr': MUON_LR,
        'pack_size': K,
    }
    kwargs.update(optimizer_kwargs)
    optimizer = MuonAdamWPack(
        [{'params': [p], 'muon': True, **(group_extra or {})}], **kwargs
    )
    for grad in grads:
        p.grad = grad.clone()
        optimizer.step()
    return optimizer


def _run_reference(
    p0: Tensor,
    grads: list[Tensor],
    *,
    nesterov: bool = True,
    momentum: float = 0.95,
    ns_steps: int = 5,
    muon_lr: list[float] = MUON_LR,
    weight_decay: list[float] = WEIGHT_DECAY,
    scales: list[float] | None = None,
    ns_on_transpose: bool = False,
) -> Tensor:
    members = [
        _NaiveMuonMember(
            p0[k].T,  # (in, out) -> nn.Linear layout (out, in)
            lr=muon_lr[k],
            weight_decay=weight_decay[k],
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            scale=None if scales is None else scales[k],
            ns_on_transpose=ns_on_transpose,
        )
        for k in range(K)
    ]
    for grad in grads:
        for k, member in enumerate(members):
            member.step(grad[k].T)
    return torch.stack([m.weight.T for m in members])


# >>> Muon groups vs the naive reference.


@pytest.mark.parametrize('nesterov', [True, False])
@pytest.mark.parametrize('shape', NON_SQUARE_SHAPES, ids=lambda s: f'{s[0]}x{s[1]}')
def test_muon_matches_naive_per_member_reference(shape, nesterov):
    n_in, n_out = shape
    p = _make_muon_param(n_in, n_out)
    p0 = p.detach().clone()
    grads = _grads(p.shape, N_STEPS)

    _run_packed(p, grads, muon_nesterov=nesterov)
    expected = _run_reference(p0, grads, nesterov=nesterov)

    # The members moved by ~0.1-0.3 in total (muon_lr * scale * O(1) per step).
    assert (p.detach() - p0).abs().max() > 0.05
    torch.testing.assert_close(p.detach(), expected, rtol=0.0, atol=EXACT_ATOL)


@pytest.mark.parametrize('nesterov', [True, False])
def test_muon_square_matches_naive_reference(nesterov):
    p = _make_muon_param(8, 8)
    p0 = p.detach().clone()
    grads = _grads(p.shape, N_STEPS)
    _run_packed(p, grads, muon_nesterov=nesterov)
    movement = (p.detach() - p0).abs().max()
    assert movement > 0.05

    # Same NS orientation as the packed code: identical up to fp32 rounding.
    expected = _run_reference(p0, grads, nesterov=nesterov, ns_on_transpose=True)
    torch.testing.assert_close(p.detach(), expected, rtol=0.0, atol=EXACT_ATOL)

    # Official orientation: bf16 rounding of X X^T vs X^T X, measured <= ~5% of the
    # total movement (8e-3 of ~0.15) for 8x8, 16x16 and 32x32 over 5 steps.
    official = _run_reference(p0, grads, nesterov=nesterov)
    torch.testing.assert_close(p.detach(), official, rtol=0.0, atol=0.1 * movement)


@pytest.mark.parametrize('ns_steps', [1, 3])
def test_muon_ns_steps_and_momentum_are_used(ns_steps):
    p = _make_muon_param(5, 7)
    p0 = p.detach().clone()
    grads = _grads(p.shape, N_STEPS)

    _run_packed(p, grads, muon_ns_steps=ns_steps, muon_momentum=0.8)
    expected = _run_reference(p0, grads, ns_steps=ns_steps, momentum=0.8)
    torch.testing.assert_close(p.detach(), expected, rtol=0.0, atol=EXACT_ATOL)


# >>> muon_scale.


@pytest.mark.parametrize('shape', [(3, 12), (12, 3), (6, 6)])
def test_muon_scale_default_equals_explicit(shape):
    n_in, n_out = shape
    default_scale = math.sqrt(max(1.0, n_out / n_in))
    grads = _grads((K, n_in, n_out), N_STEPS)

    p_default = _make_muon_param(n_in, n_out)
    _run_packed(p_default, grads)

    p_tensor = _make_muon_param(n_in, n_out)
    _run_packed(
        p_tensor, grads, group_extra={'muon_scale': torch.full((K,), default_scale)}
    )

    p_none = _make_muon_param(n_in, n_out)
    _run_packed(p_none, grads, group_extra={'muon_scale': None})

    p_float = _make_muon_param(n_in, n_out)
    _run_packed(p_float, grads, group_extra={'muon_scale': default_scale})

    for p in (p_tensor, p_none, p_float):
        torch.testing.assert_close(p.detach(), p_default.detach(), rtol=0, atol=1e-7)


def test_muon_scale_default_uses_out_over_in_of_our_layout():
    # (K, in=3, out=12): out/in = 4 -> scale 2 (the official formula applied to our
    # transposed layout would give sqrt(max(1, 3/12)) = 1).
    p = _make_muon_param(3, 12)
    p0 = p.detach().clone()
    grads = _grads(p.shape, 1)
    _run_packed(p, grads, weight_decay=0.0)

    p_unit = _make_muon_param(3, 12)
    _run_packed(p_unit, grads, weight_decay=0.0, group_extra={'muon_scale': 1.0})

    delta = p.detach() - p0
    delta_unit = p_unit.detach() - p0
    torch.testing.assert_close(delta, 2.0 * delta_unit, rtol=1e-5, atol=1e-7)


def test_explicit_per_member_muon_scale():
    scales = [0.5, 1.0, 3.0]
    p = _make_muon_param(6, 4)
    p0 = p.detach().clone()
    grads = _grads(p.shape, N_STEPS)

    _run_packed(p, grads, group_extra={'muon_scale': torch.tensor(scales)})
    expected = _run_reference(p0, grads, scales=scales)
    torch.testing.assert_close(p.detach(), expected, rtol=0.0, atol=EXACT_ATOL)


# >>> Per-member hyperparameters.


def test_per_member_weight_decay_uses_muon_lr():
    # With a zero gradient the NS output is 0, so only the decoupled weight decay
    # acts: p_k <- p_k * (1 - muon_lr_k * wd_k). The AdamW lr must not be used.
    p = _make_muon_param(4, 6)
    p0 = p.detach().clone()
    _run_packed(p, [torch.zeros(p.shape)])

    factor = 1.0 - torch.tensor(MUON_LR) * torch.tensor(WEIGHT_DECAY)
    torch.testing.assert_close(p.detach(), p0 * factor[:, None, None])


def test_member_with_zero_muon_lr_is_frozen():
    p = _make_muon_param(5, 5)
    p0 = p.detach().clone()
    _run_packed(p, _grads(p.shape, N_STEPS), muon_lr=[0.0, 0.02, 0.03])

    assert torch.equal(p.detach()[0], p0[0])
    assert not torch.equal(p.detach()[1], p0[1])
    assert not torch.equal(p.detach()[2], p0[2])


def test_members_are_independent():
    grads_a = _grads((K, 6, 4), N_STEPS, seed=1)
    grads_b = [g.clone() for g in grads_a]
    for g in grads_b:
        g[1:] = torch.randn_like(g[1:]) * 10.0

    p_a = _make_muon_param(6, 4)
    p_b = _make_muon_param(6, 4)
    _run_packed(p_a, grads_a)
    _run_packed(p_b, grads_b)

    torch.testing.assert_close(p_a.detach()[0], p_b.detach()[0], rtol=0, atol=1e-7)
    assert not torch.allclose(p_a.detach()[1], p_b.detach()[1])


def test_muon_lr_none_falls_back_to_lr():
    grads = _grads((K, 4, 6), N_STEPS)

    p_none = _make_muon_param(4, 6)
    _run_packed(p_none, grads, lr=MUON_LR, muon_lr=None)

    p_explicit = _make_muon_param(4, 6)
    _run_packed(p_explicit, grads, lr=[9.0, 9.0, 9.0], muon_lr=MUON_LR)

    torch.testing.assert_close(p_none.detach(), p_explicit.detach(), rtol=0, atol=0)


def test_float_hyperparameters_match_per_member_tensors():
    grads = _grads((K, 4, 6), N_STEPS)

    p_float = _make_muon_param(4, 6)
    _run_packed(p_float, grads, weight_decay=0.1, muon_lr=0.02)

    p_list = _make_muon_param(4, 6)
    _run_packed(p_list, grads, weight_decay=[0.1] * K, muon_lr=[0.02] * K)

    torch.testing.assert_close(p_float.detach(), p_list.detach(), rtol=0, atol=1e-6)


def test_hyperparameters_are_stored_per_group_as_float32_tensors():
    w = _make_muon_param(4, 6)
    b = nn.Parameter(torch.zeros(K, 6))
    head = nn.Parameter(torch.zeros(K, 6, 1))
    optimizer = MuonAdamWPack(
        [
            {'params': [w], 'muon': True},
            {'params': [b], 'weight_decay': 0.0},
            {'params': [head]},
        ],
        lr=LR,
        weight_decay=torch.tensor(WEIGHT_DECAY, dtype=torch.float64),
        muon_lr=0.02,
        pack_size=K,
    )
    g_muon, g_bias, g_head = optimizer.param_groups
    assert g_muon['muon'] is True
    assert g_bias['muon'] is False and g_head['muon'] is False
    assert g_bias['weight_decay'] == 0.0 and isinstance(g_bias['weight_decay'], float)
    assert isinstance(g_muon['muon_lr'], float)
    for group in optimizer.param_groups:
        for key in ('lr',):
            assert group[key].dtype == torch.float32
            assert group[key].shape == (K,)
    assert g_head['weight_decay'].dtype == torch.float32
    # Separate storage per group (and not the defaults' storage).
    assert g_muon['lr'].data_ptr() != g_head['lr'].data_ptr()
    assert g_muon['lr'].data_ptr() != optimizer.defaults['lr'].data_ptr()
    assert isinstance(optimizer.defaults['weight_decay'], Tensor)


def test_hyperparameter_validation():
    w = _make_muon_param(4, 6)
    with pytest.raises(ValueError):
        MuonAdamWPack([w], lr=[1e-3, 1e-3], weight_decay=0.0, muon_lr=0.02, pack_size=K)
    with pytest.raises(ValueError):
        MuonAdamWPack([w], lr=-1.0, weight_decay=0.0, muon_lr=0.02, pack_size=K)
    with pytest.raises(ValueError):
        MuonAdamWPack(
            [w], lr=1e-3, weight_decay=[0.0, -0.1, 0.0], muon_lr=0.02, pack_size=K
        )
    with pytest.raises(ValueError):
        MuonAdamWPack(
            [w],
            lr=1e-3,
            weight_decay=0.0,
            muon_lr=0.02,
            muon_momentum=1.0,
            pack_size=K,
        )
    with pytest.raises(ValueError):
        MuonAdamWPack([w], lr=1e-3, weight_decay=0.0, muon_lr=0.02, pack_size=K + 1)
    with pytest.raises(ValueError):
        MuonAdamWPack(
            [{'params': [nn.Parameter(torch.zeros(K, 4))], 'muon': True}],
            lr=1e-3,
            weight_decay=0.0,
            muon_lr=0.02,
            pack_size=K,
        )
    with pytest.raises(ValueError):
        MuonAdamWPack(
            [{'params': [w], 'muon': True, 'muon_scale': torch.ones(K + 1)}],
            lr=1e-3,
            weight_decay=0.0,
            muon_lr=0.02,
            pack_size=K,
        )


# >>> Non-Muon groups vs AdamWPack.


def _make_mixed_params(seed: int = 0) -> dict[str, nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    return {
        'hidden_w': nn.Parameter(torch.randn(K, 5, 8, generator=g) * 0.3),
        'hidden_b': nn.Parameter(torch.randn(K, 8, generator=g) * 0.1),
        'head_w': nn.Parameter(torch.randn(K, 8, 1, generator=g) * 0.3),
        'head_b': nn.Parameter(torch.randn(K, 1, generator=g) * 0.1),
    }


@pytest.mark.parametrize('shared_step', [True, False])
def test_non_muon_groups_equal_adamwpack(shared_step):
    params = _make_mixed_params()
    ref = {name: nn.Parameter(p.detach().clone()) for name, p in params.items()}
    hyper = {
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'beta1': 0.8,
        'beta2': 0.99,
        'eps': 1e-6,
        'pack_size': K,
        'shared_step': shared_step,
    }
    muon_opt = MuonAdamWPack(
        [
            {'params': [params['hidden_w']], 'muon': True},
            {'params': [params['hidden_b'], params['head_b']], 'weight_decay': 0.0},
            {'params': [params['head_w']]},
        ],
        muon_lr=MUON_LR,
        **hyper,
    )
    adamw_opt = AdamWPack(
        [
            {'params': [ref['hidden_b'], ref['head_b']], 'weight_decay': 0.0},
            {'params': [ref['head_w']]},
        ],
        **hyper,
    )
    g = torch.Generator().manual_seed(3)
    for step in range(N_STEPS):
        for name, p in params.items():
            # head_b skips one step so that per-param step counters diverge.
            if name == 'head_b' and step == 1:
                p.grad, ref[name].grad = None, None
                continue
            grad = torch.randn(p.shape, generator=g)
            p.grad = grad.clone()
            ref[name].grad = grad.clone()
        muon_opt.step()
        adamw_opt.step()

    for name in ('hidden_b', 'head_w', 'head_b'):
        assert torch.equal(params[name].detach(), ref[name].detach()), name
        state, ref_state = muon_opt.state[params[name]], adamw_opt.state[ref[name]]
        for key in ('exp_avg', 'exp_avg_sq'):
            assert torch.equal(state[key], ref_state[key]), (name, key)
    # Same step layout as AdamWPack: one optimizer-level int, or per-param (K,).
    if shared_step:
        assert muon_opt.state['__shared__'] == adamw_opt.state['__shared__']
        assert muon_opt.state['__shared__'] == {'step': N_STEPS}
        assert 'step' not in muon_opt.state[params['head_b']]
    else:
        head_b_step = muon_opt.state[params['head_b']]['step']
        assert torch.equal(head_b_step, torch.full((K,), N_STEPS - 1))
    # The Muon group has only the momentum buffer.
    assert set(muon_opt.state[params['hidden_w']]) == {'muon_momentum_buffer'}


def test_shared_step_counts_only_adamw_updates():
    params = _make_mixed_params()
    optimizer = MuonAdamWPack(
        [
            {'params': [params['hidden_w']], 'muon': True},
            {'params': [params['head_w']]},
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        muon_lr=MUON_LR,
        pack_size=K,
    )
    params['hidden_w'].grad = torch.randn(params['hidden_w'].shape)
    optimizer.step()  # Muon only: Muon has no step count.
    assert '__shared__' not in optimizer.state
    params['head_w'].grad = torch.randn(params['head_w'].shape)
    optimizer.step()
    assert optimizer.state['__shared__'] == {'step': 1}


# >>> Misc behaviour.


def test_grads_none_are_skipped():
    params = _make_mixed_params()
    before = {name: p.detach().clone() for name, p in params.items()}
    optimizer = MuonAdamWPack(
        [
            {'params': [params['hidden_w']], 'muon': True},
            {'params': [params['hidden_b'], params['head_w'], params['head_b']]},
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        muon_lr=MUON_LR,
        pack_size=K,
    )
    params['head_w'].grad = torch.randn(params['head_w'].shape)
    optimizer.step()

    for name in ('hidden_w', 'hidden_b', 'head_b'):
        assert torch.equal(params[name].detach(), before[name]), name
        assert params[name] not in optimizer.state
    assert not torch.equal(params['head_w'].detach(), before['head_w'])

    # A Muon weight without a gradient keeps its momentum buffer untouched.
    params['hidden_w'].grad = torch.randn(params['hidden_w'].shape)
    optimizer.step()
    buf = optimizer.state[params['hidden_w']]['muon_momentum_buffer'].clone()
    w = params['hidden_w'].detach().clone()
    params['hidden_w'].grad = None
    optimizer.step()
    assert torch.equal(optimizer.state[params['hidden_w']]['muon_momentum_buffer'], buf)
    assert torch.equal(params['hidden_w'].detach(), w)


def test_grad_is_not_modified():
    p = _make_muon_param(4, 6)
    optimizer = MuonAdamWPack(
        [{'params': [p], 'muon': True}],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        muon_lr=MUON_LR,
        pack_size=K,
    )
    for grad in _grads(p.shape, 2):
        p.grad = grad.clone()
        optimizer.step()
        assert torch.equal(p.grad, grad)


def test_closure_and_real_gradients():
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(K, 4, 6) * 0.5)
    b = nn.Parameter(torch.zeros(K, 6))
    x = torch.randn(K, 32, 4)
    y = torch.randn(K, 32, 6)
    optimizer = MuonAdamWPack(
        [{'params': [w], 'muon': True}, {'params': [b], 'weight_decay': 0.0}],
        lr=1e-2,
        weight_decay=0.0,
        muon_lr=0.05,
        pack_size=K,
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = ((torch.baddbmm(b[:, None], x, w) - y) ** 2).mean()
        loss.backward()
        return loss

    first = optimizer.step(closure)
    assert isinstance(first, Tensor)
    for _ in range(30):
        last = optimizer.step(closure)
    assert last < first


def test_state_dict_round_trip():
    params = _make_mixed_params()
    groups = [
        {'params': [params['hidden_w']], 'muon': True},
        {'params': [params['hidden_b'], params['head_b']], 'weight_decay': 0.0},
        {'params': [params['head_w']]},
    ]
    kwargs = {
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'muon_lr': MUON_LR,
        'pack_size': K,
    }
    optimizer = MuonAdamWPack(groups, **kwargs)
    grads = [
        {name: torch.randn(p.shape) for name, p in params.items()} for _ in range(4)
    ]
    for step_grads in grads[:2]:
        for name, p in params.items():
            p.grad = step_grads[name].clone()
        optimizer.step()

    params_copy = {n: nn.Parameter(p.detach().clone()) for n, p in params.items()}
    optimizer_copy = MuonAdamWPack(
        [
            {'params': [params_copy['hidden_w']], 'muon': True},
            {
                'params': [params_copy['hidden_b'], params_copy['head_b']],
                'weight_decay': 0.0,
            },
            {'params': [params_copy['head_w']]},
        ],
        **kwargs,
    )
    optimizer_copy.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    # deepcopy keeps the private attributes (pack size, shared_step) too.
    params_deep, optimizer_deep = copy.deepcopy((params, optimizer))

    for step_grads in grads[2:]:
        for name in params:
            params[name].grad = step_grads[name].clone()
            params_copy[name].grad = step_grads[name].clone()
            params_deep[name].grad = step_grads[name].clone()
        optimizer.step()
        optimizer_copy.step()
        optimizer_deep.step()
    for name in params:
        assert torch.equal(params[name].detach(), params_copy[name].detach()), name
        assert torch.equal(params[name].detach(), params_deep[name].detach()), name
