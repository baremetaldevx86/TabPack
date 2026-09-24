"""Run-to-run reproducibility on CPU (a50).

Three levels, all with fp32 on CPU:

1. ``train_pack``: the same seeds give a bitwise identical ``PackTrainResult``
   (AdamWPack, and MuonAdamWPack with an online ensemble). ``train_pack`` seeds only
   its batch generator, so the caller seeds the global RNG (initialization, dropout)
   before building the model; a different seed must change the result.
2. ``methods.<name>.run``: the same config twice in one process (with a run of
   another seed in between) gives identical reports (except wall-clock time and
   env), identical ``predictions.npz`` and identical ``config.toml``; a fresh
   interpreter with another ``PYTHONHASHSEED`` reproduces them too.
3. Sampler + config: the member configs sampled from a TabPack config are
   identical across calls, across a TOML round trip, and across interpreters with
   different ``PYTHONHASHSEED`` (no dependence on ``hash()`` / set order).

"Identical" is checked on a canonical form that keeps dict key order, the Python
type of every scalar (1 vs 1.0 vs True) and the exact bits of every float.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _helpers import make_synthetic_dataset

import tabpack_repro
from tabpack_repro.config import (
    AdamWConfig,
    DataConfig,
    HomogeneousEnsembleConfig,
    MLPMethodConfig,
    MLPModelConfig,
    OnlineEnsembleConfig,
    TabPackConfig,
    TrainingConfig,
    dump_config,
    load_config,
)
from tabpack_repro.data.pipeline import PreparedDataset
from tabpack_repro.ensembles.online import OnlineGreedyEnsemble
from tabpack_repro.methods import homogeneous, mlp, tabpack
from tabpack_repro.metrics import make_score_fn
from tabpack_repro.nn.model_pack import ModelPack
from tabpack_repro.optim.adamw_pack import AdamWPack
from tabpack_repro.optim.muon_adamw_pack import MuonAdamWPack
from tabpack_repro.optim.pack_utils import make_param_groups
from tabpack_repro.sampler import sample_configs
from tabpack_repro.training.trainer import PackTrainResult, train_pack
from tabpack_repro.types import PARTS

# The directory that contains the `tabpack_repro` package, for subprocesses.
SRC_DIR = Path(tabpack_repro.__file__).resolve().parents[1]
# Wall-clock fields and the environment description may differ between runs.
VOLATILE_KEYS = frozenset({'time', 'time_sec', 'env'})
SUBPROCESS_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def canonical(obj: Any, *, drop: frozenset[str] = frozenset()) -> Any:
    """A comparable form of `obj` that is equal only for bitwise-identical data.

    Dict key order is kept, scalars are tagged with their type (so that 1, 1.0 and
    True differ), floats are compared by their exact bits (float.hex, so NaN == NaN
    and 0.0 != -0.0) and arrays by dtype, shape and raw bytes. Dict keys in `drop`
    are left out at every level.
    """
    if isinstance(obj, dict):
        return (
            'dict',
            [(k, canonical(v, drop=drop)) for k, v in obj.items() if k not in drop],
        )
    if isinstance(obj, list | tuple):
        return ('list', [canonical(v, drop=drop) for v in obj])
    if isinstance(obj, np.ndarray):
        return ('ndarray', obj.dtype.str, obj.shape, obj.tobytes())
    if isinstance(obj, torch.Tensor):
        return canonical(obj.detach().cpu().numpy(), drop=drop)
    if isinstance(obj, bool | np.bool_):
        return ('bool', bool(obj))
    if isinstance(obj, int | np.integer):
        return ('int', int(obj))
    if isinstance(obj, float | np.floating):
        return ('float', float(obj).hex())
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f'Unexpected type in a result: {type(obj).__name__}')


def assert_identical(a: Any, b: Any, *, drop: frozenset[str] = frozenset()) -> None:
    # pytest's diff of the canonical forms points at the first differing entry.
    assert canonical(a, drop=drop) == canonical(b, drop=drop)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def subprocess_env(hash_seed: int) -> dict[str, str]:
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = str(hash_seed)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(SRC_DIR), *filter(None, [env.get('PYTHONPATH')])]
    )
    env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def run_python(script: str, *args: str, hash_seed: int) -> Any:
    """Run `script` in a fresh interpreter; it must print one JSON document."""
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script), *args],
        env=subprocess_env(hash_seed),
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 1. train_pack
# ---------------------------------------------------------------------------

PACK_SIZE = 4
BATCH_SIZE = 64
MAX_EPOCHS = 8


@pytest.fixture(scope='module')
def pack_dataset() -> PreparedDataset:
    return make_synthetic_dataset(n_train=256, n_val=96, n_test=96)


def _make_optimizer(model: ModelPack, kind: str) -> torch.optim.Optimizer:
    # Per-member hyperparameters, as in TabPack.
    if kind == 'adamw':
        return AdamWPack(
            make_param_groups(model, muon=False),
            lr=[3e-3, 1e-3, 5e-3, 2e-3],
            weight_decay=[1e-4, 1e-2, 0.0, 1e-3],
            pack_size=PACK_SIZE,
        )
    assert kind == 'muon'
    return MuonAdamWPack(
        make_param_groups(model, muon=True),
        lr=[3e-3, 1e-3, 5e-3, 2e-3],
        weight_decay=[1e-3, 1e-2, 1e-1, 1e-3],
        muon_lr=[2e-2, 1e-2, 5e-2, 3e-2],
        pack_size=PACK_SIZE,
    )


def _build(
    dataset: PreparedDataset,
    kind: str,
    *,
    init_seed: int,
    dropout: list[float] | None = None,
) -> tuple[ModelPack, torch.optim.Optimizer]:
    # train_pack does not seed the global RNG: seed it before building the model.
    torch.manual_seed(init_seed)
    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.cat_cardinalities,
        n_classes=dataset.task.n_classes,
        pack_size=PACK_SIZE,
        d_block=16,
        n_blocks=[1, 2, 3, 2],
        dropout=[0.0, 0.1, 0.2, 0.3] if dropout is None else dropout,
    )
    return model, _make_optimizer(model, kind)


def _fit(
    dataset: PreparedDataset,
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    with_ensemble: bool,
) -> PackTrainResult:
    online_ensemble = (
        OnlineGreedyEnsemble(
            score_fn=make_score_fn(dataset.y['val'], dataset.task),
            task=dataset.task,
            max_ensemble_size=4,
            patience=100,
        )
        if with_ensemble
        else None
    )
    return train_pack(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        patience=1,
        max_epochs=MAX_EPOCHS,
        seed=seed,
        online_ensemble=online_ensemble,
    )


def _train(
    dataset: PreparedDataset,
    kind: str,
    *,
    seed: int,
    init_seed: int | None = None,
    with_ensemble: bool = False,
) -> PackTrainResult:
    model, optimizer = _build(
        dataset, kind, init_seed=seed if init_seed is None else init_seed
    )
    return _fit(dataset, model, optimizer, seed=seed, with_ensemble=with_ensemble)


def _result_dict(result: PackTrainResult) -> dict[str, Any]:
    return {
        'ids': result.ids,
        'best_steps': result.best_steps,
        'predictions': {part: result.predictions[part] for part in PARTS},
        'members': result.members,
        'history': result.history,
        'n_epochs': result.n_epochs,
        'n_steps': result.n_steps,
        'online_ensemble': result.online_ensemble,
    }


PACK_CASES = [
    pytest.param('adamw', False, id='adamw'),
    pytest.param('muon', True, id='muon+online-ensemble'),
]


@pytest.mark.parametrize(('kind', 'with_ensemble'), PACK_CASES)
def test_train_pack_same_seed_is_bitwise_identical(
    pack_dataset: PreparedDataset, kind: str, with_ensemble: bool
) -> None:
    first = _train(pack_dataset, kind, seed=0, with_ensemble=with_ensemble)
    second = _train(pack_dataset, kind, seed=0, with_ensemble=with_ensemble)

    # The run is non-trivial: several epochs, members that finished at different
    # steps (not all at the epoch limit), dropout active in most members.
    assert first.n_epochs >= 3
    assert len(first.ids) > 0
    assert len(set(first.best_steps.tolist())) > 1
    if with_ensemble:
        assert first.online_ensemble is not None
        assert first.online_ensemble['size'] > 0

    np.testing.assert_array_equal(first.ids, second.ids)
    np.testing.assert_array_equal(first.best_steps, second.best_steps)
    for part in PARTS:
        assert first.predictions[part].dtype == second.predictions[part].dtype
        assert first.predictions[part].tobytes() == second.predictions[part].tobytes()
    losses = [record['train_loss'] for record in first.history]
    assert [record['train_loss'] for record in second.history] == losses
    assert_identical(_result_dict(first), _result_dict(second), drop=VOLATILE_KEYS)


@pytest.mark.parametrize(('kind', 'with_ensemble'), PACK_CASES)
def test_train_pack_different_seeds_differ(
    pack_dataset: PreparedDataset, kind: str, with_ensemble: bool
) -> None:
    reference = _train(pack_dataset, kind, seed=0, with_ensemble=with_ensemble)
    variants = {
        # Everything reseeded.
        'seed': _train(pack_dataset, kind, seed=1, with_ensemble=with_ensemble),
        # Same initialization and dropout seed, other batch order.
        'batch order': _train(
            pack_dataset, kind, seed=1, init_seed=0, with_ensemble=with_ensemble
        ),
        # Same batch order, other initialization.
        'initialization': _train(
            pack_dataset, kind, seed=0, init_seed=1, with_ensemble=with_ensemble
        ),
    }
    for name, result in variants.items():
        assert result.history[0]['train_loss'] != reference.history[0]['train_loss'], (
            name
        )
        assert canonical(result.predictions) != canonical(reference.predictions), name


def test_train_pack_batch_order_depends_only_on_its_seed(
    pack_dataset: PreparedDataset,
) -> None:
    """Without dropout, the global RNG state at the call does not matter (the
    batches come from train_pack's own generator); with dropout it does, which is
    why the caller must seed the global RNG before training."""

    def fit_after(global_seed: int, dropout: list[float]) -> PackTrainResult:
        model, optimizer = _build(pack_dataset, 'adamw', init_seed=0, dropout=dropout)
        torch.manual_seed(global_seed)
        return _fit(pack_dataset, model, optimizer, seed=0, with_ensemble=False)

    no_dropout = [0.0] * PACK_SIZE
    assert_identical(
        _result_dict(fit_after(123, no_dropout)),
        _result_dict(fit_after(456, no_dropout)),
        drop=VOLATILE_KEYS,
    )
    dropout = [0.2] * PACK_SIZE
    assert canonical(fit_after(123, dropout).predictions) != canonical(
        fit_after(456, dropout).predictions
    )


# ---------------------------------------------------------------------------
# 2. methods.<name>.run on a tiny on-disk dataset
# ---------------------------------------------------------------------------

N_TRAIN, N_VAL, N_TEST = 320, 120, 120
METHODS = ['mlp', 'homogeneous', 'tabpack']
RUN_FILES = {'report.json', 'predictions.npz', 'config.toml'}


def write_dataset(path: Path, *, seed: int = 0) -> Path:
    """A learnable binclass dataset in the official on-disk format: float32 x_num
    (with one two-valued column, moved to the categorical features by the
    pipeline), string x_cat, int64 y and splits/default/{train,val,test}.npy."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    x_num = rng.standard_normal((n, 4)).astype(np.float32)
    x_num[:, 3] = rng.integers(0, 2, n)
    colors = rng.choice(np.array(['red', 'green', 'blue']), n)
    sizes = rng.choice(np.array(['s', 'm']), n)
    x_cat = np.stack([colors, sizes], axis=1).astype(str)
    logit = (
        1.5 * x_num[:, 0]
        - x_num[:, 1]
        + 0.5 * x_num[:, 2] * x_num[:, 3]
        + (colors == 'red')
        + 0.5 * rng.standard_normal(n)
    )
    y = (logit > 0.3).astype(np.int64)

    path.mkdir(parents=True)
    (path / 'info.json').write_text(
        json.dumps({'task': {'type': 'binclass', 'score': 'accuracy'}})
    )
    np.save(path / 'x_num.npy', x_num)
    np.save(path / 'x_cat.npy', x_cat)
    np.save(path / 'y.npy', y)
    split_dir = path / 'splits' / 'default'
    split_dir.mkdir(parents=True)
    index = rng.permutation(n).astype(np.int64)
    np.save(split_dir / 'train.npy', index[:N_TRAIN])
    np.save(split_dir / 'val.npy', index[N_TRAIN : N_TRAIN + N_VAL])
    np.save(split_dir / 'test.npy', index[N_TRAIN + N_VAL :])
    return path


def make_config(
    method: str, data_dir: Path, seed: int
) -> MLPMethodConfig | HomogeneousEnsembleConfig | TabPackConfig:
    data = DataConfig(path=str(data_dir.resolve()))
    training = TrainingConfig(
        batch_size=64, patience=2, max_epochs=12, eval_batch_size=1024, device='cpu'
    )
    model = MLPModelConfig(n_blocks=2, d_block=16, dropout=0.1)
    optimizer = AdamWConfig(lr=3e-3, weight_decay=1e-4)
    if method == 'mlp':
        return MLPMethodConfig(
            seed=seed, data=data, model=model, optimizer=optimizer, training=training
        )
    if method == 'homogeneous':
        return HomogeneousEnsembleConfig(
            seed=seed,
            n_models=3,
            data=data,
            model=model,
            optimizer=optimizer,
            training=training,
        )
    assert method == 'tabpack'
    return TabPackConfig(
        seed=seed,
        n_models=4,
        d_block=16,
        data=data,
        training=training,
        online_ensemble=OnlineEnsembleConfig(patience=3, max_ensemble_size=4),
    )


RUNNERS = {'mlp': mlp.run, 'homogeneous': homogeneous.run, 'tabpack': tabpack.run}


def run_method(method: str, data_dir: Path, seed: int, output_dir: Path) -> Path:
    report = RUNNERS[method](make_config(method, data_dir, seed), output_dir)
    assert {p.name for p in output_dir.iterdir()} == RUN_FILES
    # The returned report is the one on disk.
    assert json.loads((output_dir / 'report.json').read_text()) == json.loads(
        json.dumps(report)
    )
    return output_dir


def assert_same_run(a: Path, b: Path) -> None:
    """Identical report (up to time fields and env), predictions and config file."""
    report_a = json.loads((a / 'report.json').read_text())
    report_b = json.loads((b / 'report.json').read_text())
    assert_identical(report_a, report_b, drop=VOLATILE_KEYS)
    predictions_a = load_npz(a / 'predictions.npz')
    predictions_b = load_npz(b / 'predictions.npz')
    assert list(predictions_a) == list(predictions_b) == ['val', 'test']
    assert_identical(predictions_a, predictions_b)
    assert (a / 'config.toml').read_bytes() == (b / 'config.toml').read_bytes()


@pytest.fixture(scope='module')
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_dataset(tmp_path_factory.mktemp('data') / 'toy')


@pytest.fixture(scope='module')
def reference_runs(
    data_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    """One seed-0 run of every method, in this process."""
    root = tmp_path_factory.mktemp('reference')
    return {
        method: run_method(method, data_dir, 0, root / method) for method in METHODS
    }


@pytest.mark.parametrize('method', METHODS)
def test_run_same_seed_is_identical(
    method: str, data_dir: Path, reference_runs: dict[str, Path], tmp_path: Path
) -> None:
    # A run with another seed in between must not leak state into the next run.
    other = run_method(method, data_dir, 1, tmp_path / 'seed-1')
    again = run_method(method, data_dir, 0, tmp_path / 'seed-0')
    reference = reference_runs[method]
    assert_same_run(reference, again)

    report = json.loads((reference / 'report.json').read_text())
    assert report['n_epochs'] >= 2
    assert report['metrics']['test']['accuracy'] > 0.6
    # Another seed gives another run.
    other_report = json.loads((other / 'report.json').read_text())
    assert other_report['seed'] == 1
    assert canonical(load_npz(other / 'predictions.npz')) != canonical(
        load_npz(reference / 'predictions.npz')
    )
    assert canonical(other_report['members'], drop=VOLATILE_KEYS) != canonical(
        report['members'], drop=VOLATILE_KEYS
    )


_RUN_SCRIPT = """
    import json
    import sys
    from importlib import import_module

    from tabpack_repro.config import load_config

    for config_path, output_dir in zip(sys.argv[1::2], sys.argv[2::2]):
        config = load_config(config_path)
        import_module(f'tabpack_repro.methods.{config.method}').run(config, output_dir)
    print(json.dumps({'hash': hash('tabpack')}))
"""


@pytest.mark.slow
def test_run_is_identical_in_a_fresh_interpreter(
    data_dir: Path, reference_runs: dict[str, Path], tmp_path: Path
) -> None:
    """Every method, seed 0, run from its dumped config.toml in a subprocess with
    another PYTHONHASHSEED, reproduces the in-process runs.

    Marked slow only because of the fresh interpreter's start-up (~10 s here: torch,
    sklearn and the lazily imported torch._dynamo); the runs themselves take ~2 s.
    """
    args: list[str] = []
    for method in METHODS:
        config = make_config(method, data_dir, 0)
        config_path = tmp_path / f'{method}.toml'
        dump_config(config, config_path)
        assert load_config(config_path) == config
        args += [str(config_path), str(tmp_path / 'runs' / method)]
    hash_seed = 12345
    output = run_python(_RUN_SCRIPT, *args, hash_seed=hash_seed)
    if os.environ.get('PYTHONHASHSEED') != str(hash_seed):
        # The subprocess really hashed strings differently from this process.
        assert output['hash'] != hash('tabpack')
    for method in METHODS:
        assert_same_run(reference_runs[method], tmp_path / 'runs' / method)


# ---------------------------------------------------------------------------
# 3. Sampler + config round trip
# ---------------------------------------------------------------------------

# The official Churn space plus the less common sampler features, so that the
# checks also cover nested dicts, plain lists, categorical choices, '$list', the
# '?' coin flip with a non-float default and step arguments.
RICH_SPACE: dict[str, Any] = {
    'model': {
        'n_blocks': ['_tune_', 'int', 1, 4],
        'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
        'd_hidden': ['_tune_', '$list', 3, 'int', 8, 64, 8],
        'activation': ['_tune_', 'categorical', ['ReLU', 'GELU', 'SiLU']],
        'nested': {'scale': ['_tune_', '?loguniform', 1, 0.1, 10.0], 'fixed': 7},
    },
    'optimizer': {
        'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
        'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
        'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
        'betas': [0.9, ['_tune_', 'uniform', 0.99, 0.999, 0.001]],
    },
}
N_SAMPLED = 24
SPACES = {'official': TabPackConfig().space, 'rich': RICH_SPACE}


def _tabpack_config(space_name: str, seed: int = 3) -> TabPackConfig:
    return TabPackConfig(seed=seed, n_models=N_SAMPLED, space=SPACES[space_name])


def _sample(config: TabPackConfig) -> list[dict[str, Any]]:
    return sample_configs(config.space, config.n_models, seed=config.seed)


@pytest.mark.parametrize('space_name', list(SPACES))
def test_sample_configs_same_across_calls(space_name: str) -> None:
    config = _tabpack_config(space_name)
    first = _sample(config)
    assert len(first) == N_SAMPLED
    assert_identical(first, _sample(config))
    assert canonical(_sample(_tabpack_config(space_name, seed=4))) != canonical(first)


@pytest.mark.parametrize('space_name', list(SPACES))
def test_member_configs_survive_the_toml_round_trip(
    space_name: str, tmp_path: Path
) -> None:
    config = _tabpack_config(space_name)
    expected = _sample(config)

    # The run config: the loaded config samples the same member configs.
    dump_config(config, tmp_path / 'tabpack.toml')
    loaded = load_config(tmp_path / 'tabpack.toml')
    assert loaded == config
    assert_identical(loaded.space, config.space)
    assert_identical(_sample(loaded), expected)

    # Explicit member configs (the conservative protocol): exact floats and types.
    explicit = TabPackConfig(seed=0, n_models=len(expected), configs=expected)
    dump_config(explicit, tmp_path / 'explicit.toml')
    reloaded = load_config(tmp_path / 'explicit.toml')
    assert isinstance(reloaded, TabPackConfig)
    assert_identical(reloaded.configs, expected)


_SAMPLE_SCRIPT = """
    import json
    import sys

    from tabpack_repro.config import load_config
    from tabpack_repro.sampler import sample_configs

    config = load_config(sys.argv[1])
    configs = sample_configs(config.space, config.n_models, seed=config.seed)
    # json.dumps writes floats with repr(), which round-trips exactly.
    print(json.dumps({'hash': hash('tabpack'), 'configs': configs}))
"""


@pytest.mark.parametrize('space_name', list(SPACES))
def test_member_configs_independent_of_hash_seed(
    space_name: str, tmp_path: Path
) -> None:
    config = _tabpack_config(space_name)
    expected = _sample(config)
    config_path = tmp_path / 'tabpack.toml'
    dump_config(config, config_path)

    outputs = [
        run_python(_SAMPLE_SCRIPT, str(config_path), hash_seed=hash_seed)
        for hash_seed in (0, 4242)
    ]
    # The two interpreters really used different string hashes.
    assert outputs[0]['hash'] != outputs[1]['hash']
    for output in outputs:
        assert_identical(output['configs'], expected)
