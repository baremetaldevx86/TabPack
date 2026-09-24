"""Per-member training state (a20)."""

from __future__ import annotations

from torch import Tensor

from tabpack_repro.types import PartKey


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
    """

    def __init__(self, pack_size: int) -> None:
        raise NotImplementedError

    @property
    def pack_size(self) -> int:
        raise NotImplementedError

    def step(self) -> None:
        """All running members took one optimizer step."""
        raise NotImplementedError

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
        """
        raise NotImplementedError

    def select_(self, keep_idx: Tensor) -> None:
        """Keep only members keep_idx (same semantics as nn.pack_ops.pack_select_)."""
        raise NotImplementedError
