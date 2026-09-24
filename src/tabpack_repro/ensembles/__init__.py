"""Ensemble aggregation and (online) greedy ensemble selection."""

from tabpack_repro.ensembles.aggregate import average_predictions
from tabpack_repro.ensembles.greedy import greedy_ensemble
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble

__all__ = ['OnlineGreedyEnsemble', 'average_predictions', 'greedy_ensemble']
