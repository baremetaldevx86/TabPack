"""Tests for OneHotEncoding and ModelPack (a12)."""

from __future__ import annotations

import pytest
import torch
from _helpers import make_synthetic_dataset
from torch import Tensor, nn

from tabpack_repro.nn.model_pack import ModelPack, OneHotEncoding

N_NUM = 4
CARDS = [3, 2, 4]
D_BLOCK = 8


def make_model(
    *,
    n_num_features: int = N_NUM,
    cat_cardinalities: list[int] | None = None,
    n_classes: int | None = 2,
    pack_size: int = 3,
    n_blocks: int | list[int] = 2,
    dropout: float | list[float] = 0.0,
    seed: int = 0,
) -> ModelPack:
    torch.manual_seed(seed)
    return ModelPack(
        n_num_features=n_num_features,
        cat_cardinalities=CARDS if cat_cardinalities is None else cat_cardinalities,
        n_classes=n_classes,
        pack_size=pack_size,
        d_block=D_BLOCK,
        n_blocks=n_blocks,
        dropout=dropout,
    )


def make_inputs(
    *batch: int, n_num: int = N_NUM, cards: list[int] | None = None, seed: int = 1
) -> tuple[Tensor | None, Tensor | None]:
    """Random x_num (*batch, n_num) and in-vocabulary x_cat (*batch, n_cat)."""
    cards = CARDS if cards is None else cards
    g = torch.Generator().manual_seed(seed)
    x_num = torch.randn(*batch, n_num, generator=g) if n_num else None
    x_cat = (
        torch.stack([torch.randint(0, c, batch, generator=g) for c in cards], -1)
        if cards
        else None
    )
    return x_num, x_cat


def expand_k(x: Tensor | None, k: int) -> Tensor | None:
    """A materialized per-member copy (K, B, f) of shared rows (B, f)."""
    return None if x is None else x.unsqueeze(0).expand(k, -1, -1).clone()


# >>> OneHotEncoding


def test_one_hot_known_codes() -> None:
    x = torch.tensor([[0, 1, 3], [2, 0, 0], [1, 1, 2]])
    out = OneHotEncoding(CARDS)(x)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert out.dtype == torch.float32
    assert torch.equal(out, expected)


def test_one_hot_unknown_codes_encode_to_zeros() -> None:
    # Column 0 (card 3): code 3 == card and code 7 > card are unknown.
    # Column 1 (card 2): code 2 unknown. Column 2 (card 4): code 100 unknown.
    x = torch.tensor([[3, 1, 0], [7, 2, 100], [0, 0, 3]])
    out = OneHotEncoding(CARDS)(x)
    expected = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(out, expected)
    # Each column contributes exactly one 1 for known codes and none for unknown.
    known = (x < torch.tensor(CARDS)).sum(-1).float()
    assert torch.equal(out.sum(-1), known)


def test_one_hot_arbitrary_leading_dims() -> None:
    _, x = make_inputs(3, 5)
    assert x is not None
    enc = OneHotEncoding(CARDS)
    out = enc(x)
    assert out.shape == (3, 5, sum(CARDS))
    for k in range(3):
        assert torch.equal(out[k], enc(x[k]))
    assert enc(x[0, 0]).shape == (sum(CARDS),)


def test_one_hot_int32_codes() -> None:
    _, x = make_inputs(6)
    assert x is not None
    enc = OneHotEncoding(CARDS)
    assert torch.equal(enc(x.int()), enc(x))


def test_one_hot_has_no_parameters_or_buffers() -> None:
    enc = OneHotEncoding(CARDS)
    assert list(enc.parameters()) == []
    assert list(enc.buffers()) == []
    assert enc.state_dict() == {}


def test_one_hot_rejects_bad_input() -> None:
    enc = OneHotEncoding(CARDS)
    with pytest.raises(ValueError):
        enc(torch.zeros(4, 2, dtype=torch.long))
    with pytest.raises(TypeError):
        enc(torch.zeros(4, 3))
    with pytest.raises(ValueError):
        OneHotEncoding([3, 0])


# >>> ModelPack: construction and shapes


def test_pack_invariant_and_attributes() -> None:
    model = make_model(pack_size=5, n_blocks=[1, 2, 3, 2, 1], dropout=0.1)
    assert model.pack_size == 5
    assert isinstance(model.backbone, nn.Module)
    assert isinstance(model.head, nn.Module)
    assert model.head.weight.shape == (5, D_BLOCK, 1)
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        assert tensor.shape[0] == 5, name
    # The first block sees the numerical features + the one-hot columns.
    d_in = N_NUM + sum(CARDS)
    assert model.backbone.blocks[0].linear.weight.shape == (5, d_in, D_BLOCK)


@pytest.mark.parametrize(
    ('n_classes', 'tail'), [(None, ()), (2, ()), (3, (3,)), (7, (7,))]
)
@pytest.mark.parametrize('shared', [True, False])
def test_output_shapes(n_classes: int | None, tail: tuple, shared: bool) -> None:
    k, b = 3, 10
    model = make_model(n_classes=n_classes, pack_size=k)
    x_num, x_cat = make_inputs(b) if shared else make_inputs(k, b)
    out = model(x_num, x_cat)
    assert out.shape == (k, b, *tail)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ('n_num', 'cards'), [(N_NUM, []), (0, CARDS), (N_NUM, CARDS), (0, [5])]
)
@pytest.mark.parametrize('shared', [True, False])
def test_missing_num_or_cat(n_num: int, cards: list[int], shared: bool) -> None:
    k, b = 2, 6
    model = make_model(n_num_features=n_num, cat_cardinalities=cards, pack_size=k)
    x_num, x_cat = make_inputs(b, n_num=n_num, cards=cards)
    if not shared:
        x_num, x_cat = expand_k(x_num, k), expand_k(x_cat, k)
    assert (model.cat_encoding is None) == (not cards)
    assert model(x_num, x_cat).shape == (k, b)


def test_forward_rejects_inconsistent_inputs() -> None:
    model = make_model(pack_size=3)
    x_num, x_cat = make_inputs(4)
    assert x_num is not None and x_cat is not None
    with pytest.raises(ValueError):
        model(None, None)
    with pytest.raises(ValueError):
        model(None, x_cat)  # the model has numerical features
    with pytest.raises(ValueError):
        model(x_num, None)  # the model has categorical features
    with pytest.raises(ValueError):
        model(x_num[:, :2], x_cat)  # wrong number of numerical features
    with pytest.raises(ValueError):
        model(x_num.expand(2, -1, -1), x_cat)  # wrong pack size
    with pytest.raises(ValueError):
        model(x_num[0], x_cat[0])  # 1-D input


def test_constructor_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        make_model(n_num_features=0, cat_cardinalities=[])
    with pytest.raises(ValueError):
        make_model(n_classes=1)
    with pytest.raises(ValueError):
        make_model(n_num_features=-1)


# >>> ModelPack: shared vs per-member inputs


@pytest.mark.parametrize('n_classes', [None, 2, 3])
def test_shared_equals_per_member_with_same_rows(n_classes: int | None) -> None:
    k, b = 4, 12
    model = make_model(
        n_classes=n_classes, pack_size=k, n_blocks=[1, 2, 3, 2], dropout=0.3
    ).eval()
    x_num, x_cat = make_inputs(b)
    with torch.no_grad():
        shared = model(x_num, x_cat)
        per_member = model(expand_k(x_num, k), expand_k(x_cat, k))
        mixed_1 = model(x_num, expand_k(x_cat, k))
        mixed_2 = model(expand_k(x_num, k), x_cat)
    torch.testing.assert_close(per_member, shared)
    torch.testing.assert_close(mixed_1, shared)
    torch.testing.assert_close(mixed_2, shared)


def test_per_member_rows_are_routed_to_their_member() -> None:
    k, b = 3, 7
    model = make_model(pack_size=k, n_blocks=[3, 1, 2]).eval()
    x_num, x_cat = make_inputs(k, b)
    assert x_num is not None and x_cat is not None
    with torch.no_grad():
        per_member = model(x_num, x_cat)
        for i in range(k):
            torch.testing.assert_close(per_member[i], model(x_num[i], x_cat[i])[i])


@pytest.mark.parametrize('with_cat', [False, True])
def test_shared_input_is_expanded_without_copy(with_cat: bool) -> None:
    k, b = 5, 9
    cards = CARDS if with_cat else []
    model = make_model(cat_cardinalities=cards, pack_size=k).eval()
    x_num, x_cat = make_inputs(b, cards=cards)
    assert x_num is not None
    seen: list[Tensor] = []
    model.backbone.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    with torch.no_grad():
        model(x_num, x_cat)
    (x,) = seen
    assert x.shape == (k, b, N_NUM + sum(cards))
    assert x.stride(0) == 0  # a broadcast view: one (B, d_in) buffer for all members
    if not with_cat:
        assert x.data_ptr() == x_num.data_ptr()


# >>> ModelPack: heterogeneous members


def _manual_member_forward(
    model: ModelPack, k: int, x_num: Tensor, x_cat: Tensor, n_blocks: int
) -> Tensor:
    """Member k of the pack, written out as a plain MLP (eval mode, ReLU)."""
    assert model.cat_encoding is not None
    x = torch.cat([x_num, model.cat_encoding(x_cat)], -1)
    for block in list(model.backbone.blocks)[:n_blocks]:
        x = torch.relu(x @ block.linear.weight[k] + block.linear.bias[k])
    return (x @ model.head.weight[k] + model.head.bias[k]).squeeze(-1)


def test_heterogeneous_members_match_independent_mlps() -> None:
    n_blocks, dropout = [1, 3, 2, 1], [0.0, 0.5, 0.1, 0.25]
    model = make_model(pack_size=4, n_blocks=n_blocks, dropout=dropout).eval()
    x_num, x_cat = make_inputs(11)
    assert x_num is not None and x_cat is not None
    with torch.no_grad():
        out = model(x_num, x_cat)
        for k, nb in enumerate(n_blocks):
            expected = _manual_member_forward(model, k, x_num, x_cat, nb)
            torch.testing.assert_close(out[k], expected)


def test_heterogeneous_dropout_in_train_mode() -> None:
    model = make_model(pack_size=3, n_blocks=2, dropout=[0.0, 0.5, 0.0]).train()
    x_num, x_cat = make_inputs(64)
    torch.manual_seed(10)
    out_1 = model(x_num, x_cat)
    torch.manual_seed(11)
    out_2 = model(x_num, x_cat)
    # Members without dropout are deterministic; the member with p=0.5 is not.
    assert torch.equal(out_1[0], out_2[0])
    assert torch.equal(out_1[2], out_2[2])
    assert not torch.equal(out_1[1], out_2[1])
    # Eval mode switches dropout off for every member.
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x_num, x_cat), model(x_num, x_cat))


def test_in_place_member_removal_is_picked_up() -> None:
    # Mimics pack_ops.pack_select_: ModelPack must not cache anything derived from K.
    model = make_model(pack_size=4, n_blocks=[1, 3, 2, 2], dropout=0.2).eval()
    x_num, x_cat = make_inputs(10)
    keep = torch.tensor([3, 1])
    with torch.no_grad():
        before = model(x_num, x_cat)
        for module in model.modules():
            for p in module.parameters(recurse=False):
                p.data = p.data[keep]
            for name, buf in list(module.named_buffers(recurse=False)):
                setattr(module, name, buf[keep])
        assert model.pack_size == 2
        after = model(x_num, x_cat)
    torch.testing.assert_close(after, before[keep])


# >>> ModelPack: gradients and training


def test_gradients_reach_all_members() -> None:
    n_blocks = [1, 3, 2]
    k = len(n_blocks)
    model = make_model(pack_size=k, n_blocks=n_blocks, dropout=[0.0, 0.1, 0.2])
    x_num, x_cat = make_inputs(k, 16)
    model(x_num, x_cat).square().sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        for member in range(k):
            norm = p.grad[member].abs().sum().item()
            if name.startswith('backbone.blocks.'):
                i_block = int(name.split('.')[2])
                if i_block >= n_blocks[member]:
                    assert norm == 0.0, (name, member)  # the member skips this block
                    continue
            assert norm > 0.0, (name, member)


def test_members_are_independent() -> None:
    k = 3
    model = make_model(pack_size=k, n_blocks=[2, 1, 2])
    x_num, x_cat = make_inputs(k, 16)
    model(x_num, x_cat)[1].sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None
        assert p.grad[0].abs().sum() == 0, name
        assert p.grad[2].abs().sum() == 0, name


@pytest.mark.parametrize('n_classes', [2, 3])
def test_training_reduces_loss(n_classes: int) -> None:
    dataset = make_synthetic_dataset(n_train=256, n_val=16, n_test=16)
    assert dataset.x_num is not None and dataset.x_cat is not None
    x_num, x_cat, y = dataset.x_num['train'], dataset.x_cat['train'], dataset.y['train']
    if n_classes == 3:  # a learnable 3-class target derived from the binary one
        y = y + (x_num[:, 4] > 0.5).long()
    k, n, batch_size = 3, len(y), 64
    torch.manual_seed(0)
    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.cat_cardinalities,
        n_classes=n_classes,
        pack_size=k,
        d_block=32,
        n_blocks=[1, 2, 3],
        dropout=[0.0, 0.1, 0.0],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def loss_fn(logits: Tensor, target: Tensor) -> Tensor:
        # logits (K, B) or (K, B, C); target (K, B) or (B,).
        target = target.expand(logits.shape[:2])
        if n_classes == 2:
            return nn.functional.binary_cross_entropy_with_logits(
                logits, target.float()
            )
        return nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())

    def full_loss() -> Tensor:
        model.eval()
        with torch.no_grad():
            return loss_fn(model(x_num, x_cat), y)

    initial = full_loss()
    for _ in range(60):
        model.train()
        idx = torch.stack([torch.randperm(n)[:batch_size] for _ in range(k)])
        loss = loss_fn(model(x_num[idx], x_cat[idx]), y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = full_loss()
    assert final < 0.7 * initial
