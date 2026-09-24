"""Tests for MLPBlockPack / MLPBackbonePack (a11).

The reference for every pack member is an ordinary ``nn.Sequential`` of
``nn.Linear -> activation -> nn.Dropout`` built from copies of that member's weights,
so the pack is checked against K independent plain-PyTorch MLPs of different depths.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from tabpack_repro.nn.linear_pack import LinearPack
from tabpack_repro.nn.mlp_pack import MLPBackbonePack, MLPBlockPack

ATOL = 1e-5
RTOL = 1e-5


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def _make_backbone(
    *,
    n_blocks: int | list[int],
    pack_size: int,
    d_in: int = 7,
    d_block: int = 16,
    dropout: float | list[float] = 0.0,
    activation: str = 'ReLU',
    seed: int = 0,
) -> MLPBackbonePack:
    torch.manual_seed(seed)
    return MLPBackbonePack(
        d_in=d_in,
        d_block=d_block,
        n_blocks=n_blocks,
        dropout=dropout,
        activation=activation,
        pack_size=pack_size,
    )


def _plain_linear(pack: LinearPack, k: int) -> nn.Linear:
    """An nn.Linear holding a copy of member k of a LinearPack."""
    weight = pack.weight.detach()[k]  # (in, out)
    linear = nn.Linear(weight.shape[0], weight.shape[1], bias=pack.bias is not None)
    with torch.no_grad():
        linear.weight.copy_(weight.T)
        if pack.bias is not None:
            linear.bias.copy_(pack.bias.detach()[k])
    return linear


def _reference_mlps(
    backbone: MLPBackbonePack, *, activation: str = 'ReLU', dropout: float = 0.0
) -> list[nn.Sequential]:
    """K independent plain MLPs; member k has n_blocks[k] blocks."""
    mlps = []
    for k, depth in enumerate(backbone.n_blocks.tolist()):
        layers: list[nn.Module] = []
        for block in list(backbone.blocks)[:depth]:
            layers += [
                _plain_linear(block.linear, k),
                getattr(nn, activation)(),
                nn.Dropout(dropout),
            ]
        mlps.append(nn.Sequential(*layers))
    return mlps


def _reference_forward(mlps: list[nn.Sequential], x: Tensor) -> Tensor:
    return torch.stack([mlp(x[k]) for k, mlp in enumerate(mlps)])


def _select_members_(module: nn.Module, keep_idx: Tensor) -> None:
    """Remove members in place the way pack_ops.pack_select_ is specified to."""
    for param in module.parameters():
        param.data = param.data[keep_idx]
        param.grad = None
    for submodule in module.modules():
        for name, buffer in list(submodule.named_buffers(recurse=False)):
            setattr(submodule, name, buffer[keep_idx])


def _record_member_idx(backbone: MLPBackbonePack) -> list[list[Tensor | None]]:
    """Record, for each block, the member_idx it was called with on every call."""
    calls: list[list[Tensor | None]] = [[] for _ in backbone.blocks]

    def make_hook(i: int):
        def hook(module, args, kwargs):
            member_idx = args[1] if len(args) > 1 else kwargs.get('member_idx')
            calls[i].append(None if member_idx is None else member_idx.clone())

        return hook

    for i, block in enumerate(backbone.blocks):
        block.register_forward_pre_hook(make_hook(i), with_kwargs=True)
    return calls


# ----------------------------------------------------------------------------------
# MLPBlockPack
# ----------------------------------------------------------------------------------
def test_block_pack_structure() -> None:
    block = MLPBlockPack(5, 8, dropout=[0.0, 0.1, 0.2], activation='ReLU', pack_size=3)
    assert isinstance(block.linear, LinearPack)
    assert block.linear.weight.shape == (3, 5, 8)
    assert isinstance(block.activation, nn.ReLU)
    assert block.dropout.p.shape == (3,)
    assert torch.allclose(block.dropout.p, torch.tensor([0.0, 0.1, 0.2]))


@pytest.mark.parametrize('activation', ['ReLU', 'GELU', 'SiLU'])
def test_block_pack_matches_plain_blocks(activation: str) -> None:
    torch.manual_seed(0)
    k, b, d_in, d_out = 4, 6, 5, 8
    block = MLPBlockPack(d_in, d_out, dropout=0.0, activation=activation, pack_size=k)
    x = torch.randn(k, b, d_in)
    act = getattr(nn, activation)()
    expected = torch.stack(
        [act(_plain_linear(block.linear, i)(x[i])) for i in range(k)]
    )
    torch.testing.assert_close(block(x), expected, atol=ATOL, rtol=RTOL)

    # With member_idx: the i-th input row is processed by member member_idx[i].
    member_idx = torch.tensor([3, 1])
    out = block(x[member_idx], member_idx)
    torch.testing.assert_close(out, expected[member_idx], atol=ATOL, rtol=RTOL)


# ----------------------------------------------------------------------------------
# MLPBackbonePack: structure
# ----------------------------------------------------------------------------------
def test_backbone_structure_mixed_depths() -> None:
    backbone = _make_backbone(n_blocks=[1, 3, 2], pack_size=3, d_in=7, d_block=16)
    assert isinstance(backbone.blocks, nn.ModuleList)
    assert len(backbone.blocks) == 3
    assert all(isinstance(block, MLPBlockPack) for block in backbone.blocks)
    assert backbone.blocks[0].linear.weight.shape == (3, 7, 16)
    for block in list(backbone.blocks)[1:]:
        assert block.linear.weight.shape == (3, 16, 16)
    assert backbone.pack_size == 3
    assert backbone.n_blocks.dtype == torch.int64
    assert backbone.n_blocks.tolist() == [1, 3, 2]
    buffers = dict(backbone.named_buffers())
    assert 'n_blocks' in buffers
    assert 'n_blocks' in backbone.state_dict()
    # The pack invariant: every param and buffer has the pack dimension first.
    for name, tensor in [*backbone.named_parameters(), *buffers.items()]:
        assert tensor.shape[0] == 3, name


def test_backbone_int_n_blocks_is_broadcast() -> None:
    backbone = _make_backbone(n_blocks=2, pack_size=4)
    assert len(backbone.blocks) == 2
    assert backbone.n_blocks.tolist() == [2, 2, 2, 2]
    assert backbone.n_blocks.dtype == torch.int64


def test_backbone_per_member_dropout_is_shared_by_all_blocks() -> None:
    p = [0.0, 0.25, 0.5]
    backbone = _make_backbone(n_blocks=[2, 1, 3], pack_size=3, dropout=p)
    for block in backbone.blocks:
        assert torch.allclose(block.dropout.p, torch.tensor(p))


@pytest.mark.parametrize(
    'kwargs',
    [
        {'n_blocks': [1, 2], 'pack_size': 3},  # wrong length
        {'n_blocks': [1, 0, 2], 'pack_size': 3},  # zero depth
        {'n_blocks': 0, 'pack_size': 3},
        {'n_blocks': -1, 'pack_size': 2},
    ],
)
def test_backbone_rejects_invalid_n_blocks(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MLPBackbonePack(d_in=3, d_block=4, dropout=0.0, **kwargs)


# ----------------------------------------------------------------------------------
# MLPBackbonePack: equivalence with K independent plain MLPs
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('activation', ['ReLU', 'GELU', 'SiLU'])
@pytest.mark.parametrize(
    'n_blocks',
    [[1, 3, 2, 3, 1], [4, 1, 2, 3, 2], [2, 2, 2, 2, 2], 3, [1, 1, 1, 1, 4]],
)
def test_backbone_matches_independent_mlps(
    n_blocks: int | list[int], activation: str
) -> None:
    k, b = 5, 9
    backbone = _make_backbone(n_blocks=n_blocks, pack_size=k, activation=activation)
    mlps = _reference_mlps(backbone, activation=activation)
    x = torch.randn(k, b, 7)
    out = backbone(x)
    assert out.shape == (k, b, 16)
    torch.testing.assert_close(out, _reference_forward(mlps, x), atol=ATOL, rtol=RTOL)


def test_backbone_accepts_expanded_shared_input() -> None:
    # ModelPack expands a shared (B, d) batch to (K, B, d) without copying.
    k = 4
    backbone = _make_backbone(n_blocks=[3, 1, 2, 3], pack_size=k)
    x_shared = torch.randn(10, 7)
    out_expanded = backbone(x_shared.expand(k, -1, -1))
    out_copy = backbone(x_shared.expand(k, -1, -1).contiguous())
    torch.testing.assert_close(out_expanded, out_copy, atol=0.0, rtol=0.0)
    mlps = _reference_mlps(backbone)
    expected = torch.stack([mlp(x_shared) for mlp in mlps])
    torch.testing.assert_close(out_expanded, expected, atol=ATOL, rtol=RTOL)


def test_backbone_gradients_match_independent_mlps() -> None:
    k, b = 5, 8
    n_blocks = [1, 3, 2, 3, 1]
    backbone = _make_backbone(n_blocks=n_blocks, pack_size=k)
    mlps = _reference_mlps(backbone)
    x = torch.randn(k, b, 7)
    g = torch.randn(k, b, 16)

    x_pack = x.clone().requires_grad_()
    (backbone(x_pack) * g).sum().backward()

    x_refs = [x[i].clone().requires_grad_() for i in range(k)]
    for i, (mlp, x_ref) in enumerate(zip(mlps, x_refs, strict=True)):
        (mlp(x_ref) * g[i]).sum().backward()

    for i in range(k):
        torch.testing.assert_close(x_pack.grad[i], x_refs[i].grad, atol=ATOL, rtol=RTOL)

    for j, block in enumerate(backbone.blocks):
        w_grad, b_grad = block.linear.weight.grad, block.linear.bias.grad
        assert w_grad is not None and b_grad is not None
        for i in range(k):
            if n_blocks[i] > j:
                ref_linear = mlps[i][3 * j]
                torch.testing.assert_close(
                    w_grad[i], ref_linear.weight.grad.T, atol=ATOL, rtol=RTOL
                )
                torch.testing.assert_close(
                    b_grad[i], ref_linear.bias.grad, atol=ATOL, rtol=RTOL
                )
            else:
                # Member i skips block j: exactly zero gradient, not just small.
                assert torch.equal(w_grad[i], torch.zeros_like(w_grad[i])), (j, i)
                assert torch.equal(b_grad[i], torch.zeros_like(b_grad[i])), (j, i)


def test_skipping_members_get_exactly_zero_grad_and_unchanged_activations() -> None:
    k = 3
    backbone = _make_backbone(n_blocks=[1, 2, 1], pack_size=k)
    x = torch.randn(k, 4, 7)
    out = backbone(x)
    # Members 0 and 2 only apply block 0, so their output is exactly block 0's output.
    first = backbone.blocks[0](x)
    assert torch.equal(out[[0, 2]], first[[0, 2]])

    # A loss that depends only on the members that skip block 1.
    out[[0, 2]].sum().backward()
    block1 = backbone.blocks[1].linear
    assert torch.equal(block1.weight.grad, torch.zeros_like(block1.weight))
    assert torch.equal(block1.bias.grad, torch.zeros_like(block1.bias))
    block0 = backbone.blocks[0].linear
    assert block0.weight.grad[[0, 2]].abs().sum() > 0
    assert torch.equal(block0.weight.grad[1], torch.zeros_like(block0.weight[1]))


# ----------------------------------------------------------------------------------
# MLPBackbonePack: which members each block is applied to
# ----------------------------------------------------------------------------------
def test_all_same_depth_takes_the_fast_path() -> None:
    backbone = _make_backbone(n_blocks=[3, 3, 3, 3], pack_size=4)
    calls = _record_member_idx(backbone)
    backbone(torch.randn(4, 5, 7))
    assert [len(c) for c in calls] == [1, 1, 1]
    assert all(c[0] is None for c in calls)  # full pack, no index_select/index_copy


def test_mixed_depths_apply_each_block_to_the_deep_members_only() -> None:
    backbone = _make_backbone(n_blocks=[1, 3, 2, 3, 1, 2], pack_size=6)
    calls = _record_member_idx(backbone)
    backbone(torch.randn(6, 5, 7))
    assert calls[0] == [None]  # every member applies the first block
    assert calls[1][0].tolist() == [1, 2, 3, 5]
    assert calls[2][0].tolist() == [1, 3]
    assert calls[1][0].dtype == torch.int64


def test_k1() -> None:
    backbone = _make_backbone(n_blocks=[3], pack_size=1)
    assert backbone.pack_size == 1
    calls = _record_member_idx(backbone)
    x = torch.randn(1, 6, 7)
    out = backbone(x)
    assert all(c == [None] for c in calls)
    (mlp,) = _reference_mlps(backbone)
    torch.testing.assert_close(out[0], mlp(x[0]), atol=ATOL, rtol=RTOL)

    backbone_int = _make_backbone(n_blocks=2, pack_size=1, seed=1)
    (mlp_int,) = _reference_mlps(backbone_int)
    torch.testing.assert_close(backbone_int(x)[0], mlp_int(x[0]), atol=ATOL, rtol=RTOL)


# ----------------------------------------------------------------------------------
# MLPBackbonePack: in-place member removal (nothing may be cached)
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    'keep',
    [
        [0, 2, 4],  # removes both depth-3 members: block 2 becomes unused
        [1, 3],  # removes all shallow members: only same-depth members remain
        [4, 1, 0],  # reordering
        [2],  # single member
        [0, 1, 2, 3, 4, 5],  # no-op
    ],
)
def test_member_removal_gives_consistent_outputs(keep: list[int]) -> None:
    k = 6
    backbone = _make_backbone(n_blocks=[1, 3, 2, 3, 1, 2], pack_size=k)
    x = torch.randn(k, 5, 7)
    # A forward before removal must not leave a stale cache. It runs without autograd
    # because `param.data = ...` with a new shape breaks backward through any graph
    # that is still alive (AccumulateGrad remembers the old shape).
    with torch.no_grad():
        before = backbone(x)

    keep_idx = torch.tensor(keep)
    _select_members_(backbone, keep_idx)
    assert backbone.pack_size == len(keep)
    assert backbone.n_blocks.tolist() == [[1, 3, 2, 3, 1, 2][i] for i in keep]

    calls = _record_member_idx(backbone)
    after = backbone(x[keep_idx])
    torch.testing.assert_close(after, before[keep_idx], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        after,
        _reference_forward(_reference_mlps(backbone), x[keep_idx]),
        atol=ATOL,
        rtol=RTOL,
    )
    # Blocks that no remaining member applies are not called at all.
    max_depth = max(backbone.n_blocks.tolist())
    assert [len(c) for c in calls] == [
        1 if i < max_depth else 0 for i in range(len(backbone.blocks))
    ]

    # Gradients after removal still flow only into the remaining members' rows.
    backbone(x[keep_idx]).sum().backward()
    for j, block in enumerate(backbone.blocks):
        grad = block.linear.weight.grad
        if j >= max_depth:
            assert grad is None
            continue
        assert grad.shape[0] == len(keep)
        for i, depth in enumerate(backbone.n_blocks.tolist()):
            if depth <= j:
                assert torch.equal(grad[i], torch.zeros_like(grad[i]))


def test_repeated_removal_and_in_place_depth_change() -> None:
    k = 5
    backbone = _make_backbone(n_blocks=[2, 3, 1, 3, 2], pack_size=k)
    x = torch.randn(k, 4, 7)
    with torch.no_grad():
        full = backbone(x)

    _select_members_(backbone, torch.tensor([0, 1, 3, 4]))
    _select_members_(backbone, torch.tensor([1, 2]))  # originally members 1 and 3
    torch.testing.assert_close(backbone(x[[1, 3]]), full[[1, 3]], atol=ATOL, rtol=RTOL)

    # Editing the buffer in place changes the depth used by the very next forward.
    backbone.n_blocks[0] = 1
    expected = _reference_forward(_reference_mlps(backbone), x[[1, 3]])
    torch.testing.assert_close(backbone(x[[1, 3]]), expected, atol=ATOL, rtol=RTOL)


def test_dtype_conversion_keeps_int_n_blocks() -> None:
    backbone = _make_backbone(n_blocks=[1, 2], pack_size=2).double()
    assert backbone.n_blocks.dtype == torch.int64
    x = torch.randn(2, 3, 7, dtype=torch.float64)
    assert backbone(x).dtype == torch.float64


# ----------------------------------------------------------------------------------
# Dropout: per member, active only in train mode
# ----------------------------------------------------------------------------------
def test_dropout_is_identity_in_eval_mode() -> None:
    k = 4
    n_blocks = [1, 3, 2, 3]
    backbone = _make_backbone(n_blocks=n_blocks, pack_size=k, dropout=0.5).eval()
    mlps = _reference_mlps(backbone)  # dropout 0
    x = torch.randn(k, 6, 7)
    out1, out2 = backbone(x), backbone(x)
    assert torch.equal(out1, out2)
    torch.testing.assert_close(out1, _reference_forward(mlps, x), atol=ATOL, rtol=RTOL)


def test_dropout_is_active_in_train_mode_per_member() -> None:
    k = 4
    p = [0.0, 0.5, 0.0, 0.3]
    backbone = _make_backbone(n_blocks=[1, 3, 3, 2], pack_size=k, dropout=p)
    x = torch.randn(k, 32, 7)
    backbone.eval()
    out_eval = backbone(x)
    backbone.train()
    torch.manual_seed(1)
    out_train1 = backbone(x)
    torch.manual_seed(2)
    out_train2 = backbone(x)

    for i, p_i in enumerate(p):
        if p_i == 0.0:
            # Members without dropout are bit-for-bit unaffected by train mode.
            assert torch.equal(out_train1[i], out_eval[i])
            assert torch.equal(out_train2[i], out_eval[i])
        else:
            assert not torch.allclose(out_train1[i], out_eval[i])
            assert not torch.allclose(out_train1[i], out_train2[i])


def test_dropout_rate_and_scaling_in_a_single_block() -> None:
    torch.manual_seed(0)
    p = [0.0, 0.3]
    backbone = _make_backbone(n_blocks=1, pack_size=2, d_block=64, dropout=p)
    x = torch.randn(2, 512, 7)
    backbone.eval()
    out_eval = backbone(x)
    backbone.train()
    out_train = backbone(x)
    alive = out_eval[1] != 0  # ReLU zeros are zero in both modes
    dropped = (out_train[1] == 0) & alive
    rate = dropped.sum().item() / alive.sum().item()
    assert abs(rate - 0.3) < 0.03
    kept = alive & ~dropped
    torch.testing.assert_close(
        out_train[1][kept], out_eval[1][kept] / 0.7, atol=ATOL, rtol=RTOL
    )
