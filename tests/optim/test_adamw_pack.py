"""Tests for AdamWPack / adamw_update_ (a15).

The reference is K independent ``torch.optim.AdamW(foreach=False)`` optimizers, one
per pack member, fed with the member slices of the same gradients.
"""

from __future__ import annotations

import copy
import io
import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from tabpack_repro.optim.adamw_pack import (
    _SHARED_STATE_KEY,
    AdamWPack,
    adamw_update_,
)

K = 3
LR = [1e-3, 3e-3, 1e-2]
WD = [0.0, 0.01, 0.5]
N_STEPS = 10
ATOL = 1e-6

# Mixed 3D weights (K, in, out) and 2D biases (K, out).
SHAPES = {'w1': (K, 4, 5), 'b1': (K, 5), 'w2': (K, 5, 2), 'b2': (K, 2)}
WEIGHTS = ('w1', 'w2')
BIASES = ('b1', 'b2')


def _make_params(seed: int = 0) -> dict[str, nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: nn.Parameter(torch.randn(shape, generator=g))
        for name, shape in SHAPES.items()
    }


def _groups(params: dict[str, nn.Parameter]) -> list[dict]:
    """Weights in the default group; biases in a zero-weight-decay group."""
    return [
        {'params': [params[n] for n in WEIGHTS]},
        {'params': [params[n] for n in BIASES], 'weight_decay': 0.0},
    ]


def _make_reference(
    params: dict[str, nn.Parameter], member: int, lr: float, wd: float, **kwargs
) -> tuple[dict[str, nn.Parameter], torch.optim.AdamW]:
    ref = {n: nn.Parameter(p.detach()[member].clone()) for n, p in params.items()}
    opt = torch.optim.AdamW(
        [
            {'params': [ref[n] for n in WEIGHTS], 'weight_decay': wd},
            {'params': [ref[n] for n in BIASES], 'weight_decay': 0.0},
        ],
        lr=lr,
        foreach=False,
        **kwargs,
    )
    return ref, opt


def _random_grads(step: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(1000 + step)
    # Different gradient scales per member and per step.
    scale = torch.tensor([1.0, 0.1, 3.0])
    return {
        name: torch.randn(shape, generator=g)
        * scale.view(-1, *(1,) * (len(shape) - 1))
        * (1 + step % 3)
        for name, shape in SHAPES.items()
    }


def _set_grads(
    params: dict[str, nn.Parameter],
    grads: dict[str, Tensor],
    member: int | None = None,
    skip: tuple[str, ...] = (),
) -> None:
    for name, p in params.items():
        if name in skip:
            p.grad = None
        else:
            p.grad = (grads[name] if member is None else grads[name][member]).clone()


def _assert_member_equal(
    pack: dict[str, nn.Parameter],
    refs: dict[int, dict[str, nn.Parameter]],
    atol: float = ATOL,
    rtol: float = 0.0,
) -> None:
    for i, (member, ref) in enumerate(refs.items()):
        for name, p in pack.items():
            torch.testing.assert_close(
                p.detach()[i],
                ref[name].detach(),
                atol=atol,
                rtol=rtol,
                msg=lambda m, n=name, k=member: f'member {k}, param {n}: {m}',
            )


PER_MEMBER_INPUTS: dict[str, Callable[[list[float]], object]] = {
    'list': list,
    'tuple': tuple,
    'tensor_f64': lambda v: torch.tensor(v, dtype=torch.float64),
}


# ----------------------------------------------------------------------------------
# Member-by-member equivalence with torch.optim.AdamW
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize('shared_step', [True, False])
@pytest.mark.parametrize('kind', list(PER_MEMBER_INPUTS))
@pytest.mark.parametrize(
    'hparams',
    [{}, {'beta1': 0.8, 'beta2': 0.99, 'eps': 1e-6}],
    ids=['default', 'custom'],
)
def test_matches_independent_torch_adamw(
    shared_step: bool, kind: str, hparams: dict
) -> None:
    params = _make_params()
    to_input = PER_MEMBER_INPUTS[kind]
    opt = AdamWPack(
        _groups(params),
        lr=to_input(LR),
        weight_decay=to_input(WD),
        pack_size=K,
        shared_step=shared_step,
        **hparams,
    )
    torch_kwargs = {}
    if hparams:
        torch_kwargs = {
            'betas': (hparams['beta1'], hparams['beta2']),
            'eps': hparams['eps'],
        }
    refs = {
        k: _make_reference(params, k, LR[k], WD[k], **torch_kwargs) for k in range(K)
    }

    for step in range(N_STEPS):
        grads = _random_grads(step)
        _set_grads(params, grads)
        opt.step()
        for k, (ref, ref_opt) in refs.items():
            _set_grads(ref, grads, member=k)
            ref_opt.step()
        _assert_member_equal(params, {k: ref for k, (ref, _) in refs.items()})

    # The optimizer states also match.
    for k, (ref, ref_opt) in refs.items():
        for name, p in params.items():
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(
                    opt.state[p][key][k],
                    ref_opt.state[ref[name]][key],
                    atol=1e-7,
                    rtol=1e-5,
                )


def test_float_hyperparameters_are_bit_exact_with_torch_adamw() -> None:
    # Float lr / wd and a shared int step take literally PyTorch's code path.
    params = _make_params()
    opt = AdamWPack(_groups(params), lr=3e-3, weight_decay=0.1, pack_size=K)
    refs = {k: _make_reference(params, k, 3e-3, 0.1) for k in range(K)}
    for step in range(N_STEPS):
        grads = _random_grads(step)
        _set_grads(params, grads)
        opt.step()
        for k, (ref, ref_opt) in refs.items():
            _set_grads(ref, grads, member=k)
            ref_opt.step()
    _assert_member_equal(params, {k: r for k, (r, _) in refs.items()}, atol=0.0)


# ----------------------------------------------------------------------------------
# Hyperparameter storage
# ----------------------------------------------------------------------------------


def test_group_hyperparameters_are_normalized() -> None:
    params = _make_params()
    lr_input = torch.tensor(LR, dtype=torch.float64)
    opt = AdamWPack(
        [
            {'params': [params['w1']]},
            {'params': [params['w2']], 'lr': [0.1, 0.2, 0.3]},
            {'params': [params[n] for n in BIASES], 'weight_decay': 0},
        ],
        lr=lr_input,
        weight_decay=0.25,
        pack_size=K,
    )
    g0, g1, g2 = opt.param_groups
    for group in (g0, g2):
        assert isinstance(group['lr'], Tensor)
        assert group['lr'].dtype == torch.float32
        assert group['lr'].shape == (K,)
        assert group['lr'].device == params['w1'].device
        torch.testing.assert_close(group['lr'], torch.tensor(LR))
    torch.testing.assert_close(g1['lr'], torch.tensor([0.1, 0.2, 0.3]))
    # A Python number stays a float ("same for all members").
    assert g0['weight_decay'] == 0.25 and isinstance(g0['weight_decay'], float)
    assert g2['weight_decay'] == 0.0 and isinstance(g2['weight_decay'], float)
    # Every group owns its tensors: no aliasing with the input or between groups.
    lr_input.mul_(10)
    assert g0['lr'].data_ptr() != g2['lr'].data_ptr()
    g0['lr'].mul_(2)
    torch.testing.assert_close(g2['lr'], torch.tensor(LR))


def test_zero_weight_decay_group_override() -> None:
    params = _make_params()
    before = {n: p.detach().clone() for n, p in params.items()}
    wd = torch.tensor([0.1, 0.2, 0.3])
    lr = torch.tensor(LR)
    opt = AdamWPack(_groups(params), lr=lr, weight_decay=wd, pack_size=K)
    # With zero gradients the Adam step is exactly 0, so only weight decay acts.
    for p in params.values():
        p.grad = torch.zeros_like(p)
    opt.step()
    for name in WEIGHTS:
        factor = (1 - lr * wd).view(-1, 1, 1)
        torch.testing.assert_close(params[name].detach(), before[name] * factor)
    for name in BIASES:
        assert torch.equal(params[name].detach(), before[name])


def test_invalid_hyperparameters_are_rejected() -> None:
    params = _make_params()
    with pytest.raises(ValueError, match='shape'):
        AdamWPack(_groups(params), lr=[1e-3, 1e-3], pack_size=K)
    with pytest.raises(ValueError, match='non-negative'):
        AdamWPack(_groups(params), lr=[1e-3, -1e-3, 1e-3], pack_size=K)
    with pytest.raises(ValueError, match='non-negative'):
        AdamWPack(_groups(params), lr=1e-3, weight_decay=-0.1, pack_size=K)
    with pytest.raises(ValueError, match='non-negative'):
        AdamWPack(_groups(params), lr=[1e-3, math.nan, 1e-3], pack_size=K)
    with pytest.raises(TypeError):
        AdamWPack(_groups(params), lr=True, pack_size=K)
    with pytest.raises(ValueError, match='beta2'):
        AdamWPack(_groups(params), lr=1e-3, beta2=1.0, pack_size=K)
    with pytest.raises(ValueError, match='pack_size'):
        AdamWPack(_groups(params), lr=1e-3, pack_size=K + 1)
    with pytest.raises(ValueError, match='shape'):
        AdamWPack([{'params': [params['w1']], 'lr': [0.1, 0.2]}], lr=1e-3, pack_size=K)


# ----------------------------------------------------------------------------------
# Step counters
# ----------------------------------------------------------------------------------


def test_shared_step_is_one_optimizer_level_int() -> None:
    params = _make_params()
    opt = AdamWPack(_groups(params), lr=LR, weight_decay=WD, pack_size=K)
    for step in range(4):
        _set_grads(params, _random_grads(step))
        opt.step()
    assert opt.state[_SHARED_STATE_KEY] == {'step': 4}
    for p in params.values():
        assert set(opt.state[p]) == {'exp_avg', 'exp_avg_sq'}
    # A step() without any gradient does not count.
    opt.zero_grad()
    opt.step()
    assert opt.state[_SHARED_STATE_KEY] == {'step': 4}


def test_per_param_step_is_an_int64_tensor() -> None:
    params = _make_params()
    opt = AdamWPack(
        _groups(params), lr=LR, weight_decay=WD, pack_size=K, shared_step=False
    )
    for step in range(4):
        # b2 has no gradient at steps 1 and 3.
        _set_grads(params, _random_grads(step), skip=('b2',) if step % 2 else ())
        opt.step()
    assert _SHARED_STATE_KEY not in opt.state
    for name, p in params.items():
        step_t = opt.state[p]['step']
        assert step_t.dtype == torch.int64 and step_t.shape == (K,)
        assert step_t.tolist() == [2 if name == 'b2' else 4] * K


def test_shared_and_per_param_steps_agree_when_all_grads_exist() -> None:
    results = {}
    for shared_step in (True, False):
        params = _make_params()
        opt = AdamWPack(
            _groups(params),
            lr=LR,
            weight_decay=WD,
            pack_size=K,
            shared_step=shared_step,
        )
        for step in range(N_STEPS):
            _set_grads(params, _random_grads(step))
            opt.step()
        results[shared_step] = params
    for name in SHAPES:
        torch.testing.assert_close(
            results[True][name], results[False][name], atol=1e-7, rtol=0
        )


def _remove_member_(
    opt: torch.optim.Optimizer, params: dict[str, nn.Parameter], keep: Tensor
) -> None:
    """A minimal stand-in for nn.pack_ops.pack_select_ + optim.optimizer_select_."""
    old_k = next(iter(params.values())).shape[0]
    for p in params.values():
        p.data = p.data[keep]
        p.grad = None
        for key, value in opt.state[p].items():
            if isinstance(value, Tensor) and value.ndim > 0 and len(value) == old_k:
                opt.state[p][key] = value[keep].clone()
    for group in opt.param_groups:
        for key, value in group.items():
            if isinstance(value, Tensor) and value.ndim > 0:
                group[key] = value[keep].clone()


@pytest.mark.parametrize('shared_step', [True, False])
def test_member_removal_keeps_the_remaining_members_exact(shared_step: bool) -> None:
    params = _make_params()
    opt = AdamWPack(
        _groups(params), lr=LR, weight_decay=WD, pack_size=K, shared_step=shared_step
    )
    refs = {k: _make_reference(params, k, LR[k], WD[k]) for k in range(K)}
    keep = torch.tensor([0, 2])
    for step in range(N_STEPS):
        if step == 4:
            _remove_member_(opt, params, keep)
            refs = {k: refs[k] for k in keep.tolist()}
        grads = _random_grads(step)
        for name in grads:
            grads[name] = grads[name][: len(refs)]
        _set_grads(params, grads)
        opt.step()
        for i, (ref, ref_opt) in enumerate(refs.values()):
            _set_grads(ref, grads, member=i)
            ref_opt.step()
    _assert_member_equal(params, {k: r for k, (r, _) in refs.items()})
    if shared_step:
        assert opt.state[_SHARED_STATE_KEY] == {'step': N_STEPS}


# ----------------------------------------------------------------------------------
# Missing gradients
# ----------------------------------------------------------------------------------


def test_grad_none_is_skipped_like_torch_adamw() -> None:
    # With per-param steps, skipping follows PyTorch exactly (no decay, no count).
    params = _make_params()
    opt = AdamWPack(
        _groups(params), lr=LR, weight_decay=WD, pack_size=K, shared_step=False
    )
    refs = {k: _make_reference(params, k, LR[k], WD[k]) for k in range(K)}
    for step in range(N_STEPS):
        skip = ('w2', 'b1') if step in (2, 5, 6) else ()
        grads = _random_grads(step)
        _set_grads(params, grads, skip=skip)
        opt.step()
        for k, (ref, ref_opt) in refs.items():
            _set_grads(ref, grads, member=k, skip=skip)
            ref_opt.step()
    _assert_member_equal(params, {k: r for k, (r, _) in refs.items()})


@pytest.mark.parametrize('shared_step', [True, False])
def test_param_without_grad_is_untouched_and_has_no_state(shared_step: bool) -> None:
    params = _make_params()
    frozen = params['w2'].detach().clone()
    opt = AdamWPack(
        _groups(params), lr=LR, weight_decay=WD, pack_size=K, shared_step=shared_step
    )
    for step in range(3):
        _set_grads(params, _random_grads(step), skip=('w2',))
        opt.step()
    assert torch.equal(params['w2'].detach(), frozen)
    assert params['w2'] not in opt.state


# ----------------------------------------------------------------------------------
# Devices, closure, state_dict, pickling
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize('shared_step', [True, False])
def test_per_member_tensors_follow_the_params_device(shared_step: bool) -> None:
    # 'meta' plays the role of a second device on CPU-only machines.
    params = {n: nn.Parameter(p.detach().to('meta')) for n, p in _make_params().items()}
    opt = AdamWPack(
        _groups(params), lr=LR, weight_decay=WD, pack_size=K, shared_step=shared_step
    )
    # Eager: moved to the params' device at construction.
    for group in opt.param_groups:
        assert group['lr'].device.type == 'meta'
    assert opt.param_groups[0]['weight_decay'].device.type == 'meta'
    # Lazy: a hyperparameter left on another device (e.g. a checkpoint loaded with
    # map_location='cpu') is moved by step() and stored back in the group.
    opt.param_groups[0]['lr'] = torch.tensor(LR)
    opt.param_groups[0]['weight_decay'] = torch.tensor(WD)
    for p in params.values():
        p.grad = torch.zeros_like(p)
    opt.step()
    assert opt.param_groups[0]['lr'].device.type == 'meta'
    assert opt.param_groups[0]['weight_decay'].device.type == 'meta'
    if not shared_step:
        assert opt.state[params['w1']]['step'].device.type == 'meta'


@pytest.mark.gpu
@pytest.mark.parametrize('shared_step', [True, False])
def test_cuda_per_member_tensors_and_equivalence(
    cuda_device: torch.device, shared_step: bool
) -> None:
    cpu_params = _make_params()
    params = {
        n: nn.Parameter(p.detach().to(cuda_device)) for n, p in cpu_params.items()
    }
    opt = AdamWPack(
        _groups(params), lr=LR, weight_decay=WD, pack_size=K, shared_step=shared_step
    )
    for group in opt.param_groups:
        assert group['lr'].device == params['w1'].device
    cpu_opt = AdamWPack(
        _groups(cpu_params),
        lr=LR,
        weight_decay=WD,
        pack_size=K,
        shared_step=shared_step,
    )
    for step in range(N_STEPS):
        grads = _random_grads(step)
        _set_grads(cpu_params, grads)
        _set_grads(params, {n: g.to(cuda_device) for n, g in grads.items()})
        opt.step()
        cpu_opt.step()
    for name, p in params.items():
        torch.testing.assert_close(p.detach().cpu(), cpu_params[name].detach())
        if not shared_step:
            assert opt.state[p]['step'].device == p.device


def test_step_with_closure_returns_the_loss() -> None:
    torch.manual_seed(0)
    x = torch.randn(K, 16, 4)
    y = torch.randn(K, 16, 2)
    params = {
        'w': nn.Parameter(torch.zeros(K, 4, 2)),
        'b': nn.Parameter(torch.zeros(K, 2)),
    }
    opt = AdamWPack(params.values(), lr=[1e-2, 3e-2, 1e-1], pack_size=K)

    def closure() -> Tensor:
        opt.zero_grad()
        pred = torch.bmm(x, params['w']) + params['b'][:, None]
        loss = (pred - y).square().mean()
        loss.backward()
        return loss

    losses = [opt.step(closure).item() for _ in range(20)]
    assert losses[-1] < losses[0]


@pytest.mark.parametrize('shared_step', [True, False])
def test_state_dict_round_trip(shared_step: bool) -> None:
    def make(params: dict[str, nn.Parameter]) -> AdamWPack:
        return AdamWPack(
            _groups(params),
            lr=LR,
            weight_decay=WD,
            pack_size=K,
            shared_step=shared_step,
        )

    params = _make_params()
    opt = make(params)
    for step in range(4):
        _set_grads(params, _random_grads(step))
        opt.step()

    # Checkpoint through torch.save / torch.load(weights_only=True).
    buffer = io.BytesIO()
    torch.save(opt.state_dict(), buffer)
    buffer.seek(0)
    state_dict = torch.load(buffer, weights_only=True)
    params2 = {n: nn.Parameter(p.detach().clone()) for n, p in params.items()}
    opt2 = make(params2)
    opt2.load_state_dict(state_dict)
    assert isinstance(opt2.param_groups[0]['lr'], Tensor)
    assert opt2.param_groups[1]['weight_decay'] == 0.0
    if shared_step:
        assert opt2.state[_SHARED_STATE_KEY] == {'step': 4}

    for step in range(4, N_STEPS):
        grads = _random_grads(step)
        _set_grads(params, grads)
        _set_grads(params2, grads)
        opt.step()
        opt2.step()
    for name in SHAPES:
        assert torch.equal(params[name], params2[name])


def test_state_dict_snapshot_of_shared_step_is_not_mutated() -> None:
    params = _make_params()
    opt = AdamWPack(_groups(params), lr=LR, pack_size=K)
    _set_grads(params, _random_grads(0))
    opt.step()
    state_dict = opt.state_dict()
    opt.step()
    assert state_dict['state'][_SHARED_STATE_KEY] == {'step': 1}
    assert opt.state[_SHARED_STATE_KEY] == {'step': 2}


def test_shared_step_mismatch_on_load_is_an_error() -> None:
    params = _make_params()
    opt = AdamWPack(_groups(params), lr=LR, pack_size=K, shared_step=True)
    _set_grads(params, _random_grads(0))
    opt.step()
    opt2 = AdamWPack(_groups(params), lr=LR, pack_size=K, shared_step=False)
    opt2.load_state_dict(opt.state_dict())
    with pytest.raises(RuntimeError, match='shared_step'):
        opt2.step()


def test_deepcopy_keeps_the_configuration() -> None:
    params = _make_params()
    opt = AdamWPack(_groups(params), lr=LR, pack_size=K, shared_step=False)
    _set_grads(params, _random_grads(0))
    opt.step()
    opt2 = copy.deepcopy(opt)
    assert opt2._shared_step is False
    assert opt2._pack_size == K
    opt2.step()  # must not fail


# ----------------------------------------------------------------------------------
# adamw_update_ on its own
# ----------------------------------------------------------------------------------


def _formula(
    p: Tensor, g: Tensor, lr: Tensor, wd: Tensor, step: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """The module docstring formula in float64, from zero moments (one update)."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    col = (-1, *(1,) * (p.ndim - 1))
    lr, wd, t = (x.double().view(col) for x in (lr, wd, step))
    p, g = p.double(), g.double()
    p = p * (1 - lr * wd)
    m = (1 - beta1) * g
    v = (1 - beta2) * g**2
    p = p - (lr / (1 - beta1**t)) * m / (v.sqrt() / (1 - beta2**t).sqrt() + eps)
    return p, m, v


@pytest.mark.parametrize('shape', [(K, 7), (K, 3, 4)])
def test_adamw_update_with_per_member_step_tensor(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    p, g = torch.randn(shape), torch.randn(shape)
    lr, wd = torch.tensor(LR), torch.tensor(WD)
    step = torch.tensor([1, 4, 9])
    m, v = torch.zeros(shape), torch.zeros(shape)
    expected = _formula(p, g, lr, wd, step)
    adamw_update_(
        p,
        g,
        m,
        v,
        lr=lr,
        weight_decay=wd,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        step=step,
    )
    for actual, exp in zip((p, m, v), expected, strict=True):
        torch.testing.assert_close(actual, exp.float(), atol=ATOL, rtol=1e-6)


def test_adamw_update_int_step_equals_constant_step_tensor() -> None:
    torch.manual_seed(0)
    p, g = torch.randn(K, 3, 4), torch.randn(K, 3, 4)
    outputs = []
    for step in (5, torch.full((K,), 5), torch.tensor(5)):
        p2, m, v = p.clone(), torch.zeros_like(p), torch.zeros_like(p)
        adamw_update_(
            p2,
            g,
            m,
            v,
            lr=torch.tensor(LR),
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            step=step,
        )
        outputs.append(p2)
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-7, rtol=0)
    torch.testing.assert_close(outputs[0], outputs[2], atol=0, rtol=0)


def test_adamw_update_rejects_bad_inputs() -> None:
    p = torch.randn(K, 4)
    kwargs = {'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'weight_decay': 0.0}
    with pytest.raises(ValueError, match='step'):
        adamw_update_(p, p, p.clone(), p.clone(), lr=1e-3, step=0, **kwargs)
    with pytest.raises(ValueError, match='lr'):
        adamw_update_(
            p, p, p.clone(), p.clone(), lr=torch.ones(K + 1), step=1, **kwargs
        )
