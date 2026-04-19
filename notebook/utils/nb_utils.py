"""Shared helpers for the notebook/ pipeline.

Provides a small layer on top of FuxiCTR so that every notebook can reuse the
same path conventions, dataset-config materialisation, train/valid split,
prediction persistence, and run discovery logic.

All paths assume the repository layout:

    <repo>/notebook/...           <- the notebooks live here
    <repo>/notebook/configs/...   <- generated YAML configs
    <repo>/notebook/artifacts/... <- run outputs (metrics, predictions, ckpts)
    <repo>/data/...               <- raw + processed data (FuxiCTR convention)
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


def _load_yaml(path: str | Path) -> Any:
    """Load YAML with FullLoader to match FuxiCTR's own convention
    (`fuxictr/utils.py:46`). FullLoader supports Python-specific tags such as
    `!!python/tuple`, which is how we express tuple-grouped fields (e.g. DIN's
    ``din_target_field: [!!python/tuple [item_id, cate_id]]``) in YAML without
    any model-specific handling in this module."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NOTEBOOK_ROOT = Path(__file__).resolve().parent.parent          # notebook/
PROJECT_ROOT = NOTEBOOK_ROOT.parent                             # repo root
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw_data"
CONFIG_ROOT = NOTEBOOK_ROOT / "configs"
ARTIFACT_ROOT = NOTEBOOK_ROOT / "artifacts"

for _d in (CONFIG_ROOT / "datasets", CONFIG_ROOT / "models", CONFIG_ROOT / "tuning",
           ARTIFACT_ROOT / "runs", ARTIFACT_ROOT / "predictions",
           ARTIFACT_ROOT / "tuning", ARTIFACT_ROOT / "analysis",
           ARTIFACT_ROOT / "figures"):
    _d.mkdir(parents=True, exist_ok=True)


def add_fuxictr_to_path() -> None:
    """Insert the project root on ``sys.path`` so ``from fuxictr import ...`` works."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# Column introspection helpers (used by 01_Build_dataset.ipynb)
# ---------------------------------------------------------------------------

def load_raw_columns(path: str | Path) -> list[str]:
    """Return the column list of a csv/parquet file without loading all rows."""
    path = Path(path)
    if path.suffix == ".parquet":
        return list(pd.read_parquet(path).columns) if path.is_file() else \
               list(pd.read_parquet(next(path.glob("*.parquet"))).columns)
    # csv
    return list(pd.read_csv(path, nrows=0).columns)


def column_overview(path: str | Path, sample_rows: int = 10_000) -> pd.DataFrame:
    """Read a sample of rows and summarise each column (dtype, samples, n_unique)."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path).head(sample_rows)
    else:
        df = pd.read_csv(path, nrows=sample_rows)
    rows = []
    for col in df.columns:
        s = df[col]
        samples = s.dropna().astype(str).head(3).tolist()
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_unique(sample)": s.nunique(dropna=True),
            "null_rate(sample)": round(s.isna().mean(), 4),
            "samples": ", ".join(samples),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Train/valid split — materialises files since FuxiCTR reads from disk
# ---------------------------------------------------------------------------

def split_train_valid(
    train_path: str | Path,
    out_dir: str | Path,
    valid_ratio: float = 0.1,
    seed: int = 42,
    group_col: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Split a csv into train/valid files. If ``group_col`` is provided, split by
    group so all rows for a given group go to one side."""
    train_path = Path(train_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train.csv"
    valid_out = out_dir / "valid.csv"
    if train_out.exists() and valid_out.exists() and not overwrite:
        print(f"[split] reuse existing split at {out_dir}")
        return train_out, valid_out

    df = pd.read_csv(train_path)
    rng = np.random.default_rng(seed)
    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
        rng.shuffle(groups)
        n_valid = max(1, int(len(groups) * valid_ratio))
        valid_groups = set(groups[:n_valid])
        mask = df[group_col].isin(valid_groups)
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_valid = max(1, int(len(idx) * valid_ratio))
        mask = np.zeros(len(df), dtype=bool)
        mask[idx[:n_valid]] = True

    df.loc[~mask].to_csv(train_out, index=False)
    df.loc[mask].to_csv(valid_out, index=False)
    print(f"[split] train={(~mask).sum():,}  valid={mask.sum():,}  -> {out_dir}")
    return train_out, valid_out


# ---------------------------------------------------------------------------
# Dataset YAML materialisation + FuxiCTR build wrapper
# ---------------------------------------------------------------------------

def _rel_to_data_root(path: str | Path) -> str:
    """FuxiCTR expects data paths relative to the working dir of the caller.
    We always call FuxiCTR with cwd == notebook/, so we express paths relative
    to the repo root via ../data/... when possible, and absolute otherwise."""
    p = Path(path).resolve()
    try:
        return "../" + str(p.relative_to(PROJECT_ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(p).replace(os.sep, "/")


def materialize_dataset_yaml(
    dataset_id: str,
    feature_cols: list[dict],
    label_col: dict,
    train_data: str | Path,
    valid_data: str | Path,
    test_data: str | Path,
    data_format: str = "csv",
    min_categr_count: int = 1,
    extra: dict | None = None,
    out_dir: Path = CONFIG_ROOT / "datasets",
) -> Path:
    """Write a FuxiCTR-style dataset_config.yaml into ``out_dir/<dataset_id>.yaml``
    and also create a sibling ``dataset_config.yaml`` wrapper that
    ``load_dataset_config`` can consume directly."""
    cfg = {
        "data_root": "../data/",
        "data_format": data_format,
        "train_data": _rel_to_data_root(train_data),
        "valid_data": _rel_to_data_root(valid_data),
        "test_data": _rel_to_data_root(test_data),
        "min_categr_count": min_categr_count,
        "feature_cols": feature_cols,
        "label_col": label_col,
    }
    if extra:
        cfg.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Individual record
    record_path = out_dir / f"{dataset_id}.yaml"
    with record_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({dataset_id: cfg}, fh, sort_keys=False, allow_unicode=True)
    # Merged dataset_config.yaml (so FuxiCTR utils.load_dataset_config works if reused)
    merged_path = out_dir / "dataset_config.yaml"
    merged: dict[str, Any] = {}
    if merged_path.exists():
        merged = yaml.safe_load(merged_path.read_text(encoding="utf-8")) or {}
    merged[dataset_id] = cfg
    with merged_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, sort_keys=False, allow_unicode=True)
    print(f"[yaml] wrote {record_path}")
    return record_path


def _list_dataset_artifacts(dataset_dir: Path) -> dict[str, list[Path]]:
    """Group existing artefacts in a processed-dataset folder."""
    feat_files = [p for p in [
        dataset_dir / "feature_map.json",
        dataset_dir / "feature_processor.pkl",
        dataset_dir / "feature_vocab.json",
    ] if p.exists()]
    parquet_files = sorted(dataset_dir.rglob("*.parquet")) if dataset_dir.exists() else []
    return {"feature_files": feat_files, "parquet_files": parquet_files}


def clear_dataset_build(dataset_id: str, data_root: Path = DATA_ROOT) -> list[Path]:
    """Remove feature_map / processor / parquet artefacts for a dataset_id so
    ``build_dataset`` will rebuild everything. Raw data under ``data/raw_data``
    is never touched."""
    ds_dir = Path(data_root) / dataset_id
    removed: list[Path] = []
    if not ds_dir.exists():
        return removed
    targets = list(ds_dir.rglob("*.parquet")) + [
        ds_dir / name for name in ("feature_map.json", "feature_processor.pkl", "feature_vocab.json")
    ]
    # Also remove subdirs created by block-size parquet output (train/, valid/, test/)
    for sub in ("train", "valid", "test"):
        d = ds_dir / sub
        if d.is_dir():
            for p in d.rglob("*"):
                if p.is_file():
                    targets.append(p)
    for t in targets:
        if t.exists() and t.is_file():
            t.unlink()
            removed.append(t)
    for sub in ("train", "valid", "test"):
        d = ds_dir / sub
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if removed:
        print(f"[clear] removed {len(removed)} file(s) from {ds_dir}")
    return removed


def build_from_yaml(
    dataset_id: str,
    config_dir: Path | None = None,
    force_rebuild: bool = False,
) -> dict:
    """Run FuxiCTR's FeatureProcessor + build_dataset using a generated YAML.

    Output artefacts land in ``data/<dataset_id>/``: feature_map.json,
    feature_processor.pkl, train.parquet/valid.parquet/test.parquet.

    Notes:
    - FuxiCTR's ``build_dataset`` silently skips the whole build (including
      parquet transform) if ``feature_map.json`` already exists. It only emits
      a ``logging.warn``, which is easy to miss. Pass ``force_rebuild=True``
      to wipe prior artefacts and rebuild from scratch.
    - After building, we verify that parquet files actually landed on disk and
      raise a clear error if they did not.
    """
    add_fuxictr_to_path()
    from fuxictr.preprocess import FeatureProcessor, build_dataset
    from fuxictr.utils import load_dataset_config, set_logger, print_to_json
    import logging

    config_dir = Path(config_dir) if config_dir else CONFIG_ROOT / "datasets"
    prev_cwd = Path.cwd()
    os.chdir(NOTEBOOK_ROOT)  # so '../data/...' in YAML resolves to <repo>/data
    try:
        params = load_dataset_config(str(config_dir), dataset_id)
        params.setdefault("model_root", str(ARTIFACT_ROOT / "runs"))
        params.setdefault("model_id", f"{dataset_id}_build")

        ds_dir = (NOTEBOOK_ROOT / params["data_root"] / dataset_id).resolve()
        before = _list_dataset_artifacts(ds_dir)
        if force_rebuild:
            clear_dataset_build(dataset_id, Path(params["data_root"]).resolve() if Path(params["data_root"]).is_absolute() else (NOTEBOOK_ROOT / params["data_root"]).resolve())
            before = _list_dataset_artifacts(ds_dir)
        elif (ds_dir / "feature_map.json").exists() and not before["parquet_files"]:
            raise RuntimeError(
                f"feature_map.json exists at {ds_dir} but no parquet files are present. "
                "FuxiCTR will skip rebuilding and leave you without parquet output. "
                "Re-run this cell with force_rebuild=True to clear and rebuild."
            )

        set_logger(params)
        logging.info("Build params: " + print_to_json(params))

        feature_encoder = FeatureProcessor(
            feature_cols=params["feature_cols"],
            label_col=params["label_col"],
            dataset_id=dataset_id,
            data_root=params["data_root"],
        )
        build_dataset(
            feature_encoder,
            train_data=params["train_data"],
            valid_data=params["valid_data"],
            test_data=params["test_data"],
        )

        after = _list_dataset_artifacts(ds_dir)
        print(f"[build] feature files: {len(after['feature_files'])}  "
              f"parquet files: {len(after['parquet_files'])}")
        for p in after["parquet_files"]:
            print(f"  - {p.relative_to(ds_dir.parent)}")
        if not after["parquet_files"]:
            raise RuntimeError(
                f"Build finished but no parquet files were written under {ds_dir}. "
                "Check the FuxiCTR log (set_logger writes to model_root). "
                "Try force_rebuild=True."
            )
        print(f"[build] done -> {ds_dir}")
        return params
    finally:
        os.chdir(prev_cwd)


# ---------------------------------------------------------------------------
# Feature map + run registry helpers (used by 02–06)
# ---------------------------------------------------------------------------

def load_feature_map(dataset_id: str) -> dict:
    fmap_path = DATA_ROOT / dataset_id / "feature_map.json"
    if not fmap_path.exists():
        raise FileNotFoundError(f"feature_map.json not found at {fmap_path}. "
                                f"Run 01_Build_dataset for '{dataset_id}' first.")
    return json.loads(fmap_path.read_text(encoding="utf-8"))


def make_run_id(model: str, dataset_id: str, seed: int) -> str:
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{model}_{dataset_id}_s{seed}"


def save_predictions(
    dataset_id: str,
    model: str,
    run_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_df: pd.DataFrame | None = None,
) -> Path:
    out_dir = ARTIFACT_ROOT / "predictions" / dataset_id / model / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1),
                       "y_pred": np.asarray(y_pred).reshape(-1)})
    if sample_df is not None:
        sample_df = sample_df.reset_index(drop=True)
        if len(sample_df) != len(df):
            raise ValueError(f"sample_df length {len(sample_df)} != predictions {len(df)}")
        df = pd.concat([sample_df.reset_index(drop=True), df], axis=1)
    path = out_dir / "preds.parquet"
    df.to_parquet(path, index=False)
    print(f"[preds] saved {len(df):,} rows -> {path}")
    return path


def load_predictions(dataset_id: str, model: str | None = None,
                     run_id: str | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    base = ARTIFACT_ROOT / "predictions" / dataset_id
    out: dict[tuple[str, str], pd.DataFrame] = {}
    if not base.exists():
        return out
    model_dirs = [base / model] if model else [p for p in base.iterdir() if p.is_dir()]
    for m_dir in model_dirs:
        run_dirs = [m_dir / run_id] if run_id else [p for p in m_dir.iterdir() if p.is_dir()]
        for r_dir in run_dirs:
            p = r_dir / "preds.parquet"
            if p.exists():
                out[(m_dir.name, r_dir.name)] = pd.read_parquet(p)
    return out


def list_runs(dataset_id: str | None = None) -> pd.DataFrame:
    """Scan ``artifacts/runs/{dataset_id}/*.metrics.json`` and return a DataFrame.

    Layout convention (matches FuxiCTR's ``{model_root}/{dataset_id}/{model_id}.model``):
        artifacts/runs/{dataset_id}/{model_id}.model           (checkpoint, written by FuxiCTR)
        artifacts/runs/{dataset_id}/{model_id}.metrics.json    (sidecar written by us)
        artifacts/runs/{dataset_id}/{model_id}.params.json     (sidecar written by us)
    """
    base = ARTIFACT_ROOT / "runs"
    rows: list[dict[str, Any]] = []
    if not base.exists():
        return pd.DataFrame(columns=["dataset", "model", "run_id", "model_id", "path"])
    dataset_dirs = [base / dataset_id] if dataset_id else [p for p in base.iterdir() if p.is_dir()]
    for ds_dir in dataset_dirs:
        if not ds_dir.is_dir():
            continue
        for mpath in ds_dir.glob("*.metrics.json"):
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
            params_path = mpath.with_name(mpath.name.replace(".metrics.json", ".params.json"))
            params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
            rows.append({
                "dataset": ds_dir.name,
                "model": metrics.get("model") or params.get("model") or "unknown",
                "run_id": metrics.get("run_id") or params.get("run_id") or mpath.stem.replace(".metrics", ""),
                "model_id": mpath.stem.replace(".metrics", ""),
                "path": str(ds_dir / mpath.stem.replace(".metrics", "")),
                **{f"metric.{k}": v for k, v in metrics.items()
                   if k not in {"model", "run_id", "dataset_id", "model_id"} and not isinstance(v, (list, dict))},
                **{f"param.{k}": v for k, v in params.items()
                   if k not in {"model", "run_id", "dataset_id", "model_id"} and not isinstance(v, (list, dict))},
            })
    if not rows:
        return pd.DataFrame(columns=["dataset", "model", "run_id", "model_id", "path"])
    return pd.DataFrame(rows).sort_values(["dataset", "model", "run_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optuna search-space → trial.suggest_*
# ---------------------------------------------------------------------------

def suggest_from_space(trial, search_space: dict) -> dict:
    """Translate a YAML-defined search space into concrete trial suggestions.

    Each entry in ``search_space`` is either a scalar (used as-is, no search)
    or a dict with a ``type`` key:

        type: categorical, choices: [a, b, c]
        type: uniform,     low: 0.0, high: 0.5
        type: loguniform,  low: 1.0e-5, high: 1.0e-2
        type: int,         low: 16, high: 128, step: 16
    """
    out: dict[str, Any] = {}
    for name, spec in search_space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            out[name] = spec
            continue
        t = spec["type"]
        if t == "categorical":
            choices = spec["choices"]
            # Optuna's SQLite storage only persists primitive choices (None/bool/int/
            # float/str). Lists/dicts raise a UserWarning and are stored as strings,
            # which breaks resume. Encode non-primitives to JSON strings and decode
            # back so YAML can still express `choices: [[256,128], [512,256,128]]`.
            if any(isinstance(c, (list, dict)) for c in choices):
                encoded = [json.dumps(c) if isinstance(c, (list, dict)) else c
                           for c in choices]
                picked = trial.suggest_categorical(name, encoded)
                out[name] = json.loads(picked) if isinstance(picked, str) and picked.startswith(("[", "{")) else picked
            else:
                out[name] = trial.suggest_categorical(name, choices)
        elif t == "uniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif t == "loguniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "int":
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                          step=int(spec.get("step", 1)))
        else:
            raise ValueError(f"Unknown search type '{t}' for '{name}'")
    return out


def decode_best_params(best_params: dict) -> dict:
    """Decode JSON-encoded values in Optuna's ``study.best_params``.

    ``suggest_from_space`` encodes non-primitive categorical choices (lists/dicts)
    as JSON strings for SQLite compatibility.  Optuna's ``best_params`` returns
    those raw strings — this function converts them back to Python objects."""
    out = {}
    for k, v in best_params.items():
        if isinstance(v, str) and v.startswith(("[", "{")):
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Training helpers (used by 03, 05)
# ---------------------------------------------------------------------------

def _default_fuxictr_params() -> dict:
    """BaseModel / data pipeline defaults that aren't model-specific."""
    return {
        "task": "binary_classification",
        "loss": "binary_crossentropy",
        "metrics": ["AUC", "logloss"],
        "optimizer": "adam",
        "monitor": "AUC",
        "monitor_mode": "max",
        "embedding_regularizer": 0,
        "net_regularizer": 0,
        "shuffle": True,
        "num_workers": 3,
        "verbose": 1,
        "early_stop_patience": 2,
        "pickle_feature_encoder": True,
        "save_best_only": True,
        "eval_steps": None,
        "debug_mode": False,
        "group_id": None,
        "use_features": None,
        "feature_specs": None,
        "feature_config": None,
        "gpu": 0,
    }


def load_base_config(model_name: str, dataset_id: str | None = None) -> dict:
    """Load ``notebook/configs/models/{model}_base.yaml`` and merge with defaults.

    Dataset-specific overrides are supported via sections keyed with
    ``Base@{dataset_id}`` and ``{model_name}@{dataset_id}`` in the same YAML.
    When ``dataset_id`` is passed, those sections are merged last so that
    sequence/target field names (e.g. ``bst_sequence_field``, ``feature_specs``)
    can differ per dataset without needing a separate file per combination.

    Merge order (later overrides earlier):
        defaults → Base → Base@{dataset_id} → {model_name} → {model_name}@{dataset_id}
    """
    path = CONFIG_ROOT / "models" / f"{model_name}_base.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model base config missing: {path}")
    data = _load_yaml(path) or {}
    base = data.get("Base", {})
    model_cfg = data.get(model_name, {})
    ds_base = data.get(f"Base@{dataset_id}", {}) if dataset_id else {}
    ds_model = data.get(f"{model_name}@{dataset_id}", {}) if dataset_id else {}
    merged = _default_fuxictr_params()
    merged.update(base)
    merged.update(ds_base)
    merged.update(model_cfg)
    merged.update(ds_model)
    merged["model"] = model_name
    return merged


def load_search_space(model_name: str) -> dict:
    path = CONFIG_ROOT / "tuning" / f"{model_name}_search.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Tuning search space missing: {path}")
    return _load_yaml(path) or {}


def load_dataset_runtime(dataset_id: str) -> dict:
    """Grab dataset_config entries (data_root, *_data, data_format, feature_cols, ...)
    that 01 wrote into ``notebook/configs/datasets/<dataset_id>.yaml``."""
    path = CONFIG_ROOT / "datasets" / f"{dataset_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"dataset config missing: {path}. Run 01 first.")
    data = _load_yaml(path) or {}
    if dataset_id not in data:
        raise KeyError(f"{dataset_id} not in {path}")
    cfg = data[dataset_id]
    cfg["dataset_id"] = dataset_id
    # data paths in YAML are relative to notebook/; resolve to absolute so we can
    # use them regardless of the current cwd.
    for k in ("train_data", "valid_data", "test_data", "data_root"):
        v = cfg.get(k)
        if v and not Path(v).is_absolute():
            cfg[k] = str((NOTEBOOK_ROOT / v).resolve())
    return cfg


def _import_model_class(model_name: str):
    add_fuxictr_to_path()
    import importlib
    mz = importlib.import_module("model_zoo")
    if not hasattr(mz, model_name):
        raise AttributeError(
            f"model_zoo has no attribute '{model_name}'. "
            f"Available (first 20): {[n for n in dir(mz) if not n.startswith('_')][:20]}"
        )
    return getattr(mz, model_name)


def run_training(
    dataset_id: str,
    model_name: str,
    params: dict,
    run_id: str,
    save_test_preds: bool = True,
    sample_feature_cols: list[str] | None = None,
    model_cls=None,
) -> dict:
    """End-to-end: fit → evaluate(test) → (optionally) save predictions & sidecars.

    Writes:
        artifacts/runs/{dataset_id}/{model_id}.model           (FuxiCTR best ckpt)
        artifacts/runs/{dataset_id}/{model_id}.metrics.json    (our sidecar)
        artifacts/runs/{dataset_id}/{model_id}.params.json     (our sidecar)
        artifacts/predictions/{dataset_id}/{model_name}/{run_id}/preds.parquet
    """
    add_fuxictr_to_path()
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.torch_utils import seed_everything
    from fuxictr.pytorch.dataloaders import RankDataLoader
    from fuxictr.utils import set_logger, print_to_json
    import logging
    import time

    ds_cfg = load_dataset_runtime(dataset_id)
    full = dict(params)
    full.update(ds_cfg)
    full["dataset_id"] = dataset_id

    model_id = f"{model_name}_{run_id}"
    full["model_id"] = model_id
    full["model_root"] = str(ARTIFACT_ROOT / "runs")
    full["model"] = model_name

    prev_cwd = Path.cwd()
    os.chdir(NOTEBOOK_ROOT)
    try:
        set_logger(full)
        logging.info(f"Run {run_id}: " + print_to_json(full))
        seed_everything(seed=full.get("seed", 2026))

        data_dir = os.path.join(full["data_root"], dataset_id)
        feature_map = FeatureMap(dataset_id, data_dir)
        feature_map.load(os.path.join(data_dir, "feature_map.json"), full)

        train_gen, valid_gen = RankDataLoader(
            feature_map,
            stage="train",
            train_data=full["train_data"],
            valid_data=full["valid_data"],
            batch_size=full["batch_size"],
            data_format=full["data_format"],
            shuffle=full["shuffle"],
        ).make_iterator()

        Model = model_cls or _import_model_class(model_name)
        model = Model(feature_map, **full)
        model.count_parameters()
        t0 = time.time()
        model.fit(train_gen, validation_data=valid_gen, epochs=full["epochs"])
        train_secs = time.time() - t0

        logging.info(f"*** Validation evaluation (run={run_id}) ***")
        valid_logs = model.evaluate(valid_gen)

        test_logs, preds_path = {}, None
        if full.get("test_data"):
            logging.info(f"*** Test evaluation (run={run_id}) ***")
            test_gen = RankDataLoader(
                feature_map,
                stage="test",
                test_data=full["test_data"],
                batch_size=full["batch_size"],
                data_format=full["data_format"],
                shuffle=False,
            ).make_iterator()
            test_logs = model.evaluate(test_gen)

            if save_test_preds:
                test_gen2 = RankDataLoader(
                    feature_map,
                    stage="test",
                    test_data=full["test_data"],
                    batch_size=full["batch_size"],
                    data_format=full["data_format"],
                    shuffle=False,
                ).make_iterator()
                y_pred = model.predict(test_gen2)
                test_df = _read_parquet_any(full["test_data"])
                if len(test_df) != len(y_pred):
                    logging.warning(
                        f"test_df rows {len(test_df)} != y_pred {len(y_pred)} — sample_df disabled"
                    )
                    sample_df = None
                    y_true = test_df[feature_map.labels[0]].values[: len(y_pred)]
                else:
                    y_true = test_df[feature_map.labels[0]].values
                    keep = sample_feature_cols or _default_sample_cols(feature_map, test_df)
                    sample_df = test_df[keep] if keep else None
                preds_path = save_predictions(
                    dataset_id, model_name, run_id, y_true, y_pred, sample_df
                )

        run_dir = ARTIFACT_ROOT / "runs" / dataset_id
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_name,
            "dataset_id": dataset_id,
            "run_id": run_id,
            "model_id": model_id,
            "train_seconds": round(train_secs, 2),
            **{f"valid_{k}": float(v) for k, v in valid_logs.items()},
            **{f"test_{k}": float(v) for k, v in test_logs.items()},
        }
        (run_dir / f"{model_id}.metrics.json").write_text(json.dumps(summary, indent=2))
        saved_params = {k: v for k, v in full.items() if _jsonable(v)}
        (run_dir / f"{model_id}.params.json").write_text(json.dumps(saved_params, indent=2))
        print(f"[run {run_id}] valid={valid_logs}  test={test_logs}")
        print(f"[run {run_id}] artifacts -> {run_dir / model_id}.*")
        if preds_path:
            print(f"[run {run_id}] preds    -> {preds_path}")
        return summary
    finally:
        os.chdir(prev_cwd)


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False


def _read_parquet_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_file():
        return pd.read_parquet(p)
    if p.is_dir():
        files = sorted(p.rglob("*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if (p.with_suffix(".parquet")).is_file():
        return pd.read_parquet(p.with_suffix(".parquet"))
    raise FileNotFoundError(f"no parquet at {path}")


def _default_sample_cols(feature_map, df: pd.DataFrame) -> list[str]:
    """Light-weight columns to persist with predictions for downstream analysis."""
    keep: list[str] = []
    for name, spec in feature_map.features.items():
        if name in df.columns and spec.get("type") in {"categorical", "numeric", "sequence"}:
            keep.append(name)
        if len(keep) >= 8:
            break
    return keep
