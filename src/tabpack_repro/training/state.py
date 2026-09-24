"""Per-member training state (a20)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from tabpack_repro.types import PACK_DIM, PartKey


class PackState:
    """Book-keeping for the K running members (host-side numpy/int where cheap).

    Attributes (all indexed by current pack position, length K):
    * ids: int64 original member ids (0..n_models-1), never reused.
    * steps: optimizer steps taken by each member.
    * n_bad_updates: consecutive evaluations without strict val improvement.
    * best_val_score: float, -inf before the first evaluation.
    * best_step: step of the best evaluation (-1 before the first evaluation).
    * best_predictions: dict part -> Tensor (K, N[, C]) at each member's best epoch.
    * best_model_state: dict name -> Tensor (K, ...) (pack_state_dict at best epoch).

    Implementation notes:
    * ids / steps / n_bad_updates / best_step are int64 numpy arrays and
      best_val_score is a float64 numpy array, so the per-epoch book-keeping never
      touches the device.
    * best_predictions / best_model_state live on the device of the tensors passed
      to the first ``update``. They are cloned once there; later updates copy only
      the improved members' slices in place (``index_copy_``).
    * An update performs exactly one device -> host transfer (``val_scores``).
    """

    def __init__(self, pack_size: int) -> None:
        assert pack_size >= 0, pack_size
        self.ids: np.ndarray = np.arange(pack_size, dtype=np.int64)
        self.steps: np.ndarray = np.zeros(pack_size, dtype=np.int64)
        self.n_bad_updates: np.ndarray = np.zeros(pack_size, dtype=np.int64)
        self.best_val_score: np.ndarray = np.full(pack_size, -np.inf, dtype=np.float64)
        self.best_step: np.ndarray = np.full(pack_size, -1, dtype=np.int64)
        self.best_predictions: dict[PartKey, Tensor] = {}
        self.best_model_state: dict[str, Tensor] = {}
        # True once the first evaluation has been recorded (the device tensors exist).
        self._has_best = False

    @property
    def pack_size(self) -> int:
        return len(self.ids)

    def step(self) -> None:
        """All running members took one optimizer step."""
        self.steps += 1

    def update(
        self,
        val_scores: Tensor,
        predictions: dict[PartKey, Tensor],
        model_state: dict[str, Tensor],
    ) -> Tensor:
        """Record an evaluation. Returns bool (K,) mask of members that improved.

        Improvement is strict (score > best). First evaluation always improves.
        Improved members: copy predictions/model_state slices, reset n_bad_updates,
        set best_step = steps. Others: n_bad_updates += 1.

        The returned mask is a CPU tensor (it is computed on the host, like the rest
        of the book-keeping). As in the official code, a NaN score recorded by the
        first evaluation is never improved upon afterwards.
        """
        k = self.pack_size
        assert val_scores.shape == (k,), (tuple(val_scores.shape), k)
        # The one and only device -> host transfer of this method.
        scores = val_scores.detach().to(device='cpu', dtype=torch.float64).numpy()

        if self._has_best:
            # Check before mutating anything, so a bad call leaves the state intact.
            _check_like(self.best_predictions, predictions)
            _check_like(self.best_model_state, model_state)

        # Members that were never evaluated improve unconditionally.
        improved = (self.best_step < 0) | (scores > self.best_val_score)
        improved_idx = np.flatnonzero(improved)
        n_improved = len(improved_idx)

        self.n_bad_updates[improved_idx] = 0
        self.n_bad_updates[~improved] += 1
        self.best_step[improved_idx] = self.steps[improved_idx]
        self.best_val_score[improved_idx] = scores[improved_idx]

        with torch.no_grad():
            if not self._has_best:
                assert n_improved == k
                self.best_predictions = {
                    part: _clone_pack(x, k) for part, x in predictions.items()
                }
                self.best_model_state = {
                    name: _clone_pack(x, k) for name, x in model_state.items()
                }
                self._has_best = True
            elif n_improved == k:
                _copy_all_(self.best_predictions, predictions)
                _copy_all_(self.best_model_state, model_state)
            elif n_improved > 0:
                # Host -> device copies of the (small) index, one per device.
                idx_host = torch.from_numpy(improved_idx)
                idx_cache: dict[torch.device, Tensor] = {}
                _copy_members_(self.best_predictions, predictions, idx_host, idx_cache)
                _copy_members_(self.best_model_state, model_state, idx_host, idx_cache)

        return torch.from_numpy(improved)

    def select_(self, keep_idx: Tensor) -> None:
        """Keep only members keep_idx (same semantics as nn.pack_ops.pack_select_)."""
        keep_idx = torch.as_tensor(keep_idx)
        assert keep_idx.ndim == 1, tuple(keep_idx.shape)
        assert not keep_idx.dtype.is_floating_point and keep_idx.dtype != torch.bool, (
            keep_idx.dtype
        )
        keep_idx = keep_idx.to(torch.int64)
        # At most one device -> host transfer (none if keep_idx is already on CPU).
        keep_host = keep_idx.detach().cpu().numpy()
        k = self.pack_size
        assert np.all((keep_host >= 0) & (keep_host < k)), (keep_host, k)
        assert len(np.unique(keep_host)) == len(keep_host), keep_host

        self.ids = self.ids[keep_host]
        self.steps = self.steps[keep_host]
        self.n_bad_updates = self.n_bad_updates[keep_host]
        self.best_val_score = self.best_val_score[keep_host]
        self.best_step = self.best_step[keep_host]

        idx_cache = {
            keep_idx.device: keep_idx,
            torch.device('cpu'): torch.from_numpy(keep_host),
        }
        with torch.no_grad():
            self.best_predictions = {
                part: x.index_select(PACK_DIM, _idx_on(x.device, keep_idx, idx_cache))
                for part, x in self.best_predictions.items()
            }
            self.best_model_state = {
                name: x.index_select(PACK_DIM, _idx_on(x.device, keep_idx, idx_cache))
                for name, x in self.best_model_state.items()
            }

    def validate(self) -> None:
        """Assert that every per-member field is consistent with pack_size."""
        k = self.pack_size
        host = {
            'ids': (self.ids, np.int64),
            'steps': (self.steps, np.int64),
            'n_bad_updates': (self.n_bad_updates, np.int64),
            'best_val_score': (self.best_val_score, np.float64),
            'best_step': (self.best_step, np.int64),
        }
        for name, (array, dtype) in host.items():
            assert isinstance(array, np.ndarray), name
            assert array.shape == (k,), (name, array.shape, k)
            assert array.dtype == dtype, (name, array.dtype)

        assert np.all(self.ids >= 0)
        assert len(np.unique(self.ids)) == k
        assert np.all(self.steps >= 0)
        assert np.all(self.n_bad_updates >= 0)
        assert np.all(self.best_step >= -1)
        assert np.all(self.best_step <= self.steps)
        never_evaluated = self.best_step < 0
        assert np.all(self.best_val_score[never_evaluated] == -np.inf)
        assert np.all(self.n_bad_updates[never_evaluated] == 0)

        if self.best_predictions or self.best_model_state:
            assert self._has_best
        if self._has_best:
            assert not np.any(never_evaluated)
        for group in (self.best_predictions, self.best_model_state):
            for name, x in group.items():
                assert isinstance(x, Tensor), name
                assert x.ndim >= 1 and x.shape[PACK_DIM] == k, (name, x.shape, k)


def _clone_pack(x: Tensor, k: int) -> Tensor:
    assert x.ndim >= 1 and x.shape[PACK_DIM] == k, (tuple(x.shape), k)
    return x.detach().clone()


def _check_like(best: dict[str, Tensor], new: dict[str, Tensor]) -> None:
    assert best.keys() == new.keys(), (sorted(best), sorted(new))
    for name, dst in best.items():
        src = new[name]
        assert src.shape == dst.shape, (name, tuple(src.shape), tuple(dst.shape))


def _copy_all_(best: dict[str, Tensor], new: dict[str, Tensor]) -> None:
    for name, dst in best.items():
        dst.copy_(new[name])


def _copy_members_(
    best: dict[str, Tensor],
    new: dict[str, Tensor],
    idx_host: Tensor,
    idx_cache: dict[torch.device, Tensor],
) -> None:
    for name, dst in best.items():
        src = new[name]
        idx = _idx_on(dst.device, idx_host, idx_cache)
        if src.device != dst.device:
            src = src.to(dst.device)
        dst.index_copy_(PACK_DIM, idx, src.index_select(PACK_DIM, idx))


def _idx_on(
    device: torch.device, idx: Tensor, cache: dict[torch.device, Tensor]
) -> Tensor:
    if device not in cache:
        # A host -> device copy may be non-blocking (safe even for pageable memory,
        # and it avoids a sync); any other direction must stay blocking.
        non_blocking = idx.device.type == 'cpu' and device.type != 'cpu'
        cache[device] = idx.to(device, non_blocking=non_blocking)
    return cache[device]
