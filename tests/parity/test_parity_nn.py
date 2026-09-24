"""Numerical parity of ``tabpack_repro.nn`` with the official pack modules.

The official modules are imported from the reference clone (``project.nn`` and, for
``ModelPack``, ``project.tabpack``) through the ``official`` fixture; nothing is
copied from them. Everything runs in float32 on CPU.

Layout mapping (official -> ours)
---------------------------------
* ``LinearPack``: official ``weight`` is ``(K, out, in)`` (nn.Linear per member),
  ours is ``(K, in, out)``, so ``ours.weight == official.weight.transpose(-2, -1)``;
  ``bias`` is ``(K, out)`` in both. Official ``pack_idx`` == our ``member_idx``.
  Under the same ``torch.manual_seed`` both constructors draw identical values.
* ``DropoutPack``: our buffer ``p`` (float32 ``(K,)``) == official ``BufferPack`` ``p``
  when the official module gets a *list* of rates. A float ``p`` makes the official
  module call ``F.dropout`` (another RNG pattern), so train-mode RNG parity is only
  compared against the list form (the form TabPack uses for heterogeneous packs).
* ``MLPBackbonePack``: ``blocks[i].linear`` <-> ``blocks[i].linear`` (LinearPack
  mapping above), ``blocks[i].dropout.p`` <-> ``blocks[i].dropout.p``. Our
  ``n_blocks`` is always an int64 ``(K,)`` buffer; the official one is a
  ``BufferPack`` for a list (then ``max_n_blocks=max(n_blocks)`` is required) and
  ``None`` for an int.
* ``ModelPack`` (official: ``project.tabpack.ModelPack``, no num embeddings):
  ``backbone`` <-> ``backbone``, ``head`` <-> ``output``, ``cat_encoding`` <->
  ``cat_module`` (no tensors); the official ``pack_view`` holds no tensors either.
  Official logits are ``(K, B, 1)`` for binclass/regression (squeezed by its
  ``apply_model_impl``), ours ``(K, B)``; multiclass is ``(K, B, C)`` in both.
  The official one-hot is int64 (cast to float32 by ModelPack), ours float32.
* Member removal: official ``module_pack_remove(m, remove_idx)`` <-> ours
  ``pack_select_(m, make_keep_idx(K, remove_idx))``; official
  ``module_pack_select(m, idx)`` (temporary, eval only) <-> ours ``pack_select_`` on
  a copy; official ``module_pack_load_state_dict(m, sd, pack_idx=idx)`` <-> ours
  ``pack_load_members_(m, pack_state_dict(src), idx)``.

Tolerances: comparisons are bit-exact (``torch.equal``) where the peer agents
reported bit-identity (a09 init/forward, a10 dropout masks, a11 backbone
forward/grads); the same computations (bmm on transposed vs. contiguous weights,
identical RNG draw order) make the whole ModelPack bit-identical too. If a BLAS
backend ever breaks that, ``EXACT`` below is the single switch to a 1e-6 tolerance.
"""

from __future__ import annotations

import copy
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from tabpack_repro.nn import (
    DropoutPack,
    LinearPack,
    MLPBackbonePack,
    ModelPack,
    OneHotEncoding,
    make_keep_idx,
    pack_load_members_,
    pack_select_,
    pack_state_dict,
)

EXACT = True  # False -> compare with rtol = atol = 1e-6 instead of bit equality.
TOL = 1e-6

K = 6
BATCH = 5
D_IN = 7
D_BLOCK = 8
N_BLOCKS = [1, 3, 2, 3, 1, 2]
DROPOUT = [0.0, 0.1, 0.25, 0.5, 0.0, 0.3]
CARDS = [3, 2, 4]


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Fixtures
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@pytest.fixture(scope='module')
def onn(official) -> ModuleType:
    """The official ``project.nn`` module."""
    return official('project.nn')


@pytest.fixture(scope='module')
def otp(official) -> ModuleType:
    """The official ``project.tabpack`` module (needs the locked parity deps)."""
    return official('project.tabpack')


@pytest.fixture(autouse=True)
def _fp32_default_dtype():
    assert torch.get_default_dtype() == torch.float32


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Helpers: weight copy and comparison with the explicit layout mapping
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def assert_same(actual: Tensor | None, expected: Tensor | None, what: str = '') -> None:
    """Bit-exact (EXACT) or 1e-6-close comparison; None must match None."""
    if actual is None or expected is None:
        assert actual is None and expected is None, f'{what}: {actual} vs {expected}'
        return
    assert actual.shape == expected.shape, f'{what}: {actual.shape} vs {expected.shape}'
    assert actual.dtype == expected.dtype, f'{what}: {actual.dtype} vs {expected.dtype}'
    if EXACT:
        if not torch.equal(actual, expected):
            diff = (actual.double() - expected.double()).abs().max().item()
            pytest.fail(f'{what}: not bit-identical, max abs diff {diff:.3g}')
    else:
        torch.testing.assert_close(actual, expected, rtol=TOL, atol=TOL, msg=what)


def linear_pairs(ours: nn.Module, off: nn.Module) -> list[tuple[LinearPack, Any]]:
    """(our LinearPack, official LinearPack) pairs covering every parameter."""
    if isinstance(ours, LinearPack):
        pairs = [(ours, off)]
    elif isinstance(ours, MLPBackbonePack):
        assert len(ours.blocks) == len(off.blocks)
        pairs = [
            (a.linear, b.linear) for a, b in zip(ours.blocks, off.blocks, strict=True)
        ]
    elif isinstance(ours, ModelPack):
        assert off.num_module is None, 'num embeddings are not part of our ModelPack'
        pairs = [*linear_pairs(ours.backbone, off.backbone), (ours.head, off.output)]
    else:
        raise TypeError(type(ours))
    # The mapping must be complete on both sides: no parameter left uncompared.
    for module, side in ((ours, 0), (off, 1)):
        mapped = {id(p) for pair in pairs for p in pair[side].parameters()}
        assert mapped == {id(p) for p in module.parameters()}
    return pairs


def copy_official_to_ours_(ours: nn.Module, off: nn.Module) -> None:
    """ours.weight = official.weight.mT and ours.bias = official.bias, everywhere."""
    with torch.no_grad():
        for a, b in linear_pairs(ours, off):
            a.weight.copy_(b.weight.transpose(-2, -1))
            assert (a.bias is None) == (b.bias is None)
            if a.bias is not None:
                a.bias.copy_(b.bias)


def assert_params_same(ours: nn.Module, off: nn.Module, *, grad: bool = False) -> None:
    """Compare every parameter (or its .grad) through the layout mapping."""
    suffix = '.grad' if grad else ''
    for i, (a, b) in enumerate(linear_pairs(ours, off)):
        assert (a.bias is None) == (b.bias is None)
        names = ['weight'] if a.bias is None else ['weight', 'bias']
        for name in names:
            x, y = getattr(a, name), getattr(b, name)
            if grad:
                x, y = x.grad, y.grad
            if name == 'weight' and y is not None:
                y = y.transpose(-2, -1)  # official (K, out, in) -> ours (K, in, out)
            assert_same(x, y, f'linear[{i}].{name}{suffix}')


def randomize_official_(off: nn.Module, seed: int) -> None:
    """Fill the official parameters with non-init values (so the copy matters)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in off.parameters():
            p.copy_(torch.randn(p.shape, generator=gen))


def rand(*shape: int, seed: int) -> Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def rand_codes(*shape_prefix: int, cards: list[int], seed: int) -> Tensor:
    """int64 codes in [0, card] per column: `card` is the official unknown code."""
    gen = torch.Generator().manual_seed(seed)
    cols = [torch.randint(0, c + 1, (*shape_prefix, 1), generator=gen) for c in cards]
    x = torch.cat(cols, dim=-1)
    # Make sure the unknown code actually occurs in every column.
    x.view(-1, len(cards))[0] = torch.tensor(cards)
    return x


def backward_with(out: Tensor, seed: int) -> None:
    (out * rand(*out.shape, seed=seed)).sum().backward()


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# LinearPack
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize(('k', 'd_in', 'd_out'), [(1, 3, 4), (4, 7, 1), (6, 5, 9)])
def test_linear_same_seed_init(onn, bias, k, d_in, d_out):
    torch.manual_seed(123)
    ours = LinearPack(d_in, d_out, pack_size=k, bias=bias)
    rng_after_ours = torch.get_rng_state()
    torch.manual_seed(123)
    off = onn.LinearPack(d_in, d_out, bias=bias, pack_size=k)
    rng_after_off = torch.get_rng_state()
    assert ours.weight.shape == (k, d_in, d_out)
    assert off.weight.shape == (k, d_out, d_in)
    assert_params_same(ours, off)
    # Same number of draws: later initializations stay in sync too.
    assert torch.equal(rng_after_ours, rng_after_off)


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('member_idx', [None, [3, 0, 2], [5], [1, 1, 4]])
def test_linear_forward_backward(onn, bias, member_idx):
    off = onn.LinearPack(D_IN, D_BLOCK, bias=bias, pack_size=K)
    randomize_official_(off, seed=1)
    ours = LinearPack(D_IN, D_BLOCK, pack_size=K, bias=bias)
    copy_official_to_ours_(ours, off)
    assert_params_same(ours, off)

    idx = None if member_idx is None else torch.tensor(member_idx)
    k_eff = K if idx is None else len(idx)
    x = rand(k_eff, BATCH, D_IN, seed=2)
    x_ours = x.clone().requires_grad_()
    x_off = x.clone().requires_grad_()

    out_ours = ours(x_ours, idx)
    out_off = off(x_off, idx)
    assert_same(out_ours, out_off, 'output')

    backward_with(out_ours, seed=3)
    backward_with(out_off, seed=3)
    assert_same(x_ours.grad, x_off.grad, 'x.grad')
    assert_params_same(ours, off, grad=True)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# DropoutPack
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@pytest.mark.parametrize('official_p', [DROPOUT, 0.3])
def test_dropout_eval_is_identity(onn, official_p):
    ours = DropoutPack(DROPOUT, pack_size=K).eval()
    off = onn.DropoutPack(official_p, pack_size=K).eval()
    x = rand(K, BATCH, D_BLOCK, seed=4)
    assert_same(ours(x), off(x), 'eval output')
    assert_same(ours(x), x, 'eval output vs input')
    idx = torch.tensor([4, 1])
    assert_same(ours(x[idx], idx), off(x[idx], idx), 'eval output (member_idx)')


def test_dropout_p0_train_is_identity(onn):
    ours = DropoutPack([0.0] * K, pack_size=K).train()
    off = onn.DropoutPack([0.0] * K, pack_size=K).train()
    x = rand(K, BATCH, D_BLOCK, seed=5)
    torch.manual_seed(0)
    out_ours = ours(x)
    torch.manual_seed(0)
    out_off = off(x)
    assert_same(out_ours, out_off, 'p=0 train output')
    assert torch.equal(out_ours, x)


@pytest.mark.parametrize('member_idx', [None, [5, 2, 3], [1]])
def test_dropout_train_same_seed(onn, member_idx):
    ours = DropoutPack(DROPOUT, pack_size=K).train()
    off = onn.DropoutPack(DROPOUT, pack_size=K).train()
    assert_same(ours.p, off.p, 'p buffer')

    idx = None if member_idx is None else torch.tensor(member_idx)
    k_eff = K if idx is None else len(idx)
    x = rand(k_eff, 64, D_BLOCK, seed=6)
    x_ours = x.clone().requires_grad_()
    x_off = x.clone().requires_grad_()
    torch.manual_seed(7)
    out_ours = ours(x_ours, idx)
    torch.manual_seed(7)
    out_off = off(x_off, idx)
    assert_same(out_ours, out_off, 'train output')
    # Not trivially equal: the masks actually drop something.
    assert not torch.equal(out_ours, x)

    backward_with(out_ours, seed=8)
    backward_with(out_off, seed=8)
    assert_same(x_ours.grad, x_off.grad, 'x.grad')
    # Same number of RNG draws: the next draws agree too.
    torch.manual_seed(7)
    ours(x, idx)
    after_ours = torch.rand(3)
    torch.manual_seed(7)
    off(x, idx)
    after_off = torch.rand(3)
    assert torch.equal(after_ours, after_off)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# MLPBackbonePack
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def make_official_backbone(onn, n_blocks, dropout, activation='ReLU'):
    return onn.MLPBackbonePack(
        d_in=D_IN,
        n_blocks=n_blocks,
        d_block=D_BLOCK,
        dropout=dropout,
        activation=activation,
        max_n_blocks=max(n_blocks) if isinstance(n_blocks, list) else None,
        pack_size=K,
    )


def make_our_backbone(n_blocks, dropout, activation='ReLU'):
    return MLPBackbonePack(
        d_in=D_IN,
        d_block=D_BLOCK,
        n_blocks=n_blocks,
        dropout=dropout,
        activation=activation,
        pack_size=K,
    )


@pytest.mark.parametrize('n_blocks', [N_BLOCKS, 2])
def test_backbone_same_seed_init(onn, n_blocks):
    torch.manual_seed(11)
    ours = make_our_backbone(n_blocks, DROPOUT)
    torch.manual_seed(11)
    off = make_official_backbone(onn, n_blocks, DROPOUT)
    assert_params_same(ours, off)
    if isinstance(n_blocks, list):
        assert_same(ours.n_blocks, off.n_blocks, 'n_blocks')
    else:
        assert off.n_blocks is None
        assert ours.n_blocks.tolist() == [n_blocks] * K
    for i, (a, b) in enumerate(zip(ours.blocks, off.blocks, strict=True)):
        assert_same(a.dropout.p, b.dropout.p, f'blocks[{i}].dropout.p')


@pytest.mark.parametrize('activation', ['ReLU', 'GELU', 'SiLU'])
@pytest.mark.parametrize('n_blocks', [N_BLOCKS, [3, 1, 1, 1, 2, 2], [2] * K, 3])
def test_backbone_forward_backward_eval(onn, n_blocks, activation):
    off = make_official_backbone(onn, n_blocks, DROPOUT, activation).eval()
    randomize_official_(off, seed=12)
    ours = make_our_backbone(n_blocks, DROPOUT, activation).eval()
    copy_official_to_ours_(ours, off)

    x = rand(K, BATCH, D_IN, seed=13)
    x_ours = x.clone().requires_grad_()
    x_off = x.clone().requires_grad_()
    out_ours = ours(x_ours)
    out_off = off(x_off)
    assert out_ours.shape == (K, BATCH, D_BLOCK)
    assert_same(out_ours, out_off, 'output')

    backward_with(out_ours, seed=14)
    backward_with(out_off, seed=14)
    assert_same(x_ours.grad, x_off.grad, 'x.grad')
    assert_params_same(ours, off, grad=True)


def test_backbone_train_same_seed(onn):
    """Dropout > 0 on partial blocks: same RNG draws in the same order."""
    off = make_official_backbone(onn, N_BLOCKS, DROPOUT).train()
    randomize_official_(off, seed=15)
    ours = make_our_backbone(N_BLOCKS, DROPOUT).train()
    copy_official_to_ours_(ours, off)

    x = rand(K, 32, D_IN, seed=16)
    x_ours = x.clone().requires_grad_()
    x_off = x.clone().requires_grad_()
    torch.manual_seed(17)
    out_ours = ours(x_ours)
    torch.manual_seed(17)
    out_off = off(x_off)
    assert_same(out_ours, out_off, 'train output')

    backward_with(out_ours, seed=18)
    backward_with(out_off, seed=18)
    assert_same(x_ours.grad, x_off.grad, 'x.grad')
    assert_params_same(ours, off, grad=True)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# OneHotEncoding + ModelPack
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@pytest.mark.parametrize('shape_prefix', [(BATCH,), (K, BATCH)])
def test_one_hot_encoding(onn, shape_prefix):
    x = rand_codes(*shape_prefix, cards=CARDS, seed=20)
    assert (x == torch.tensor(CARDS)).any(-1).any()  # unknown codes present
    out_ours = OneHotEncoding(CARDS)(x)
    out_off = onn.OneHotEncoding(CARDS)(x)
    assert out_off.dtype == torch.int64  # the official ModelPack casts it to f32
    assert_same(out_ours, out_off.to(torch.float32), 'one-hot')
    # Unknown code == cardinality -> the all-zeros block for that column.
    assert out_ours.view(-1, sum(CARDS))[0].sum() == 0


def test_one_hot_code_above_cardinality_is_our_extension(onn):
    """Documented divergence: official one_hot raises, ours encodes all zeros."""
    x = torch.tensor([[CARDS[0] + 1, 0, 0]])
    with pytest.raises(RuntimeError):
        onn.OneHotEncoding(CARDS)(x)
    assert OneHotEncoding(CARDS)(x)[0, : CARDS[0]].sum() == 0


def model_kwargs(features: str, n_classes: int | None, n_blocks, dropout):
    return {
        'n_num_features': 0 if features == 'cat' else D_IN,
        'cat_cardinalities': [] if features == 'num' else CARDS,
        'n_classes': n_classes,
        'pack_size': K,
        'n_blocks': n_blocks,
        'd_block': D_BLOCK,
        'dropout': dropout,
        'activation': 'ReLU',
    }


def make_model_pair(
    otp,
    *,
    features: str = 'both',
    n_classes: int | None = None,
    seed: int = 30,
    n_blocks: int | list[int] = N_BLOCKS,
):
    """Our and the official ModelPack, both built right after torch.manual_seed(seed).

    The parameters are then equal already (same-seed init parity); tests that must
    not rely on that call randomize_official_ + copy_official_to_ours_.
    """
    kwargs = model_kwargs(features, n_classes, n_blocks, DROPOUT)
    torch.manual_seed(seed)
    ours = ModelPack(**kwargs)
    torch.manual_seed(seed)
    off = otp.ModelPack(
        **kwargs, max_n_blocks=max(n_blocks) if isinstance(n_blocks, list) else None
    )
    return ours, off


def model_inputs(features: str, *shape_prefix: int, seed: int):
    x_num = None if features == 'cat' else rand(*shape_prefix, D_IN, seed=seed)
    x_cat = (
        None
        if features == 'num'
        else rand_codes(*shape_prefix, cards=CARDS, seed=seed + 1)
    )
    return x_num, x_cat


def official_logits(off, x_num, x_cat, n_classes):
    out = off(x_num, x_cat)
    return out.squeeze(-1) if n_classes is None or n_classes == 2 else out


@pytest.mark.parametrize('n_blocks', [N_BLOCKS, 2])
@pytest.mark.parametrize('n_classes', [None, 2, 3])
def test_model_same_seed_init(otp, n_blocks, n_classes):
    ours, off = make_model_pair(otp, n_classes=n_classes, n_blocks=n_blocks)
    assert_params_same(ours, off)
    assert_same(ours.backbone.n_blocks, torch.tensor(n_blocks).expand(K), 'n_blocks')


@pytest.mark.parametrize('features', ['both', 'num', 'cat'])
@pytest.mark.parametrize('n_classes', [None, 2, 3])
def test_model_eval_shared_inputs(otp, features, n_classes):
    ours, off = make_model_pair(otp, features=features, n_classes=n_classes)
    randomize_official_(off, seed=31)
    copy_official_to_ours_(ours, off)
    ours.eval()
    off.eval()

    x_num, x_cat = model_inputs(features, BATCH, seed=32)
    out_ours = ours(x_num, x_cat)
    out_off = official_logits(off, x_num, x_cat, n_classes)
    expected_shape = (K, BATCH) if n_classes in (None, 2) else (K, BATCH, n_classes)
    assert out_ours.shape == expected_shape
    assert_same(out_ours, out_off, 'eval logits')


@pytest.mark.parametrize('features', ['both', 'num', 'cat'])
@pytest.mark.parametrize('n_classes', [None, 3])
@pytest.mark.parametrize('per_member', [True, False])
def test_model_train_same_seed(otp, features, n_classes, per_member):
    """Training forward (dropout on) + backward, per-member or shared batches."""
    ours, off = make_model_pair(otp, features=features, n_classes=n_classes)
    ours.train()
    off.train()
    # Same-seed init already makes the parameters equal; no copy needed here.
    assert_params_same(ours, off)

    prefix = (K, BATCH) if per_member else (BATCH,)
    x_num, x_cat = model_inputs(features, *prefix, seed=33)
    x_num_ours = None if x_num is None else x_num.clone().requires_grad_()
    x_num_off = None if x_num is None else x_num.clone().requires_grad_()
    torch.manual_seed(34)
    out_ours = ours(x_num_ours, x_cat)
    torch.manual_seed(34)
    out_off = official_logits(off, x_num_off, x_cat, n_classes)
    assert_same(out_ours, out_off, 'train logits')

    backward_with(out_ours, seed=35)
    backward_with(out_off, seed=35)
    assert_params_same(ours, off, grad=True)
    if x_num is not None:
        assert_same(x_num_ours.grad, x_num_off.grad, 'x_num.grad')


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Member removal / selection / partial loading
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def assert_buffers_same(ours: ModelPack, off) -> None:
    assert_same(ours.backbone.n_blocks, off.backbone.n_blocks, 'n_blocks')
    for i, (a, b) in enumerate(
        zip(ours.backbone.blocks, off.backbone.blocks, strict=True)
    ):
        assert_same(a.dropout.p, b.dropout.p, f'blocks[{i}].dropout.p')


# [1, 3] removes both 3-block members: the last block is then unused by everyone.
@pytest.mark.parametrize('remove', [[1, 4], [1, 3], [0], [5, 2, 0], [0, 1, 2, 3, 4]])
def test_model_member_removal(otp, onn, remove):
    ours, off = make_model_pair(otp)
    randomize_official_(off, seed=40)
    copy_official_to_ours_(ours, off)

    remove_idx = torch.tensor(remove)
    old_to_new = onn.module_pack_remove(off, remove_idx)
    assert len(old_to_new) == sum(1 for _ in off.parameters())
    params_before = [id(p) for p in ours.parameters()]
    pack_select_(ours, make_keep_idx(K, remove_idx))
    assert [id(p) for p in ours.parameters()] == params_before  # identity kept

    k_new = K - len(remove)
    assert ours.pack_size == off.pack_size == k_new
    assert off.pack_view.pack_size == k_new
    assert_params_same(ours, off)
    assert_buffers_same(ours, off)

    # Eval on shared rows.
    ours.eval()
    off.eval()
    x_num, x_cat = model_inputs('both', BATCH, seed=41)
    with torch.no_grad():
        assert_same(ours(x_num, x_cat), off(x_num, x_cat).squeeze(-1), 'eval logits')

    # Training step on per-member rows with dropout: outputs and grads.
    ours.train()
    off.train()
    x_num, x_cat = model_inputs('both', k_new, BATCH, seed=42)
    torch.manual_seed(43)
    out_ours = ours(x_num, x_cat)
    torch.manual_seed(43)
    out_off = off(x_num, x_cat).squeeze(-1)
    assert_same(out_ours, out_off, 'train logits')
    backward_with(out_ours, seed=44)
    backward_with(out_off, seed=44)
    assert_params_same(ours, off, grad=True)


@pytest.mark.parametrize('select', [[0, 2, 5], [4, 1], [3]])
def test_model_select_matches_official_module_pack_select(otp, onn, select):
    """Official temporary eval-time selection == ours applied to a copy."""
    ours, off = make_model_pair(otp)
    randomize_official_(off, seed=50)
    copy_official_to_ours_(ours, off)
    ours.eval()
    off.eval()
    x_num, x_cat = model_inputs('both', BATCH, seed=51)
    idx = torch.tensor(select)

    selected = copy.deepcopy(ours)
    pack_select_(selected, idx)
    with torch.no_grad(), onn.module_pack_select(off, idx):
        out_off = off(x_num, x_cat).squeeze(-1)
    with torch.no_grad():
        out_ours = selected(x_num, x_cat)
        # Selecting == slicing the full pack's output (members are independent).
        assert_same(out_ours, ours(x_num, x_cat)[idx], 'select vs slice')
    assert_same(out_ours, out_off, 'selected logits')
    # The official selection is temporary; the full pack is unchanged afterwards.
    assert off.pack_size == K
    assert_params_same(ours, off)


def test_model_load_members_matches_official(otp, onn):
    """Restoring best checkpoints of some members (official 'stop' path)."""
    ours, off = make_model_pair(otp, seed=60)
    randomize_official_(off, seed=61)
    copy_official_to_ours_(ours, off)
    ours_best, off_best = make_model_pair(otp, seed=62)
    randomize_official_(off_best, seed=63)
    copy_official_to_ours_(ours_best, off_best)

    stop = torch.tensor([1, 4, 5])
    onn.module_pack_load_state_dict(off, off_best.state_dict(), pack_idx=stop)
    pack_load_members_(ours, pack_state_dict(ours_best), stop)
    assert_params_same(ours, off)
    assert_buffers_same(ours, off)

    ours.eval()
    off.eval()
    ours_best.eval()
    x_num, x_cat = model_inputs('both', BATCH, seed=64)
    with torch.no_grad():
        out_ours = ours(x_num, x_cat)
        out_off = off(x_num, x_cat).squeeze(-1)
        assert_same(out_ours, out_off, 'logits after loading')
        # Loaded members now behave like the checkpoint, the others are untouched.
        assert_same(out_ours[stop], ours_best(x_num, x_cat)[stop], 'loaded members')
