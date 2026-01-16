# =========================================================================
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import argparse
import yaml
import os
import gc
import pandas as pd
import optuna
from optuna.samplers import TPESampler

import fuxictr_version
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
from __init__ import *
from longctr_dataloader import LongCTRDataLoader

yaml.Dumper.ignore_aliases = lambda *args: True


class FuxiCTROptimizer:
    """Optuna-based hyperparameter optimizer for FuxiCTR models."""

    def __init__(self, base_params, tuner_space, data_root, dataset_id, monitor=None, sampler_type='tpe'):
        self.base_params = base_params.copy()
        self.tuner_space = tuner_space
        self.data_root = data_root
        self.dataset_id = dataset_id
        self.monitor = monitor if monitor else {'PRAUC': 1}

        if sampler_type == 'tpe':
            self.sampler = TPESampler(seed=self.base_params.get('seed', 42))
        else:
            self.sampler = TPESampler(seed=42)

        self.best_metrics = None
        self.best_scalability = None

    def _suggest_parameters(self, trial):
        """Convert tuner_space list format to Optuna suggestions."""
        params = {}

        for param_name, param_value in self.tuner_space.items():
            # Skip non-tunable parameters
            if param_name in ['model_root', 'monitor', 'batch_size', 'epochs', 'seed']:
                params[param_name] = param_value
                continue

            # Handle metrics specially: [[PRAUC, AUC, logloss]] -> [PRAUC, AUC, logloss]
            if param_name == 'metrics':
                if isinstance(param_value, list) and len(param_value) == 1 and isinstance(param_value[0], list):
                    params[param_name] = param_value[0]
                else:
                    params[param_name] = param_value
                continue

            # List format -> categorical suggestion
            if isinstance(param_value, list):
                # Convert list items to strings for Optuna, then convert back
                str_choices = [str(v) for v in param_value]
                selected = trial.suggest_categorical(param_name, str_choices)
                # Convert back to original type
                original_value = param_value[str_choices.index(selected)]
                params[param_name] = original_value
            else:
                # Single value -> use as is
                params[param_name] = param_value

        return params

    def _compute_monitor_score(self, metrics):
        """Compute combined score from metrics based on monitor configuration."""
        score = 0.0
        for metric_name, direction in self.monitor.items():
            if metric_name not in metrics:
                continue
            # direction=1 means maximize -> negate for Optuna minimization
            # direction=-1 means minimize -> use as is
            score += -direction * metrics[metric_name]
        return score

    def objective(self, trial):
        """Optuna objective function."""
        suggested_params = self._suggest_parameters(trial)
        params = self.base_params.copy()
        params.update(suggested_params)

        seed_everything(seed=params.get('seed', 42))

        # Build feature_map for each trial with current params
        feature_map = build_feature_map(params.copy(), self.data_root, self.dataset_id)

        # Create checkpoint directory
        checkpoint_dir = os.path.join(params.get('model_root', './checkpoints/'), self.dataset_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_class = eval(params['model'])
        model = model_class(feature_map, **params)

        params["data_loader"] = LongCTRDataLoader
        train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

        model.fit(train_gen, validation_data=valid_gen, **params)

        valid_result, valid_scalability = model.evaluate(valid_gen)

        del train_gen, valid_gen, feature_map
        gc.collect()

        # Cleanup checkpoint
        if hasattr(model, 'checkpoint') and os.path.exists(model.checkpoint):
            os.remove(model.checkpoint)

        monitor_score = self._compute_monitor_score(valid_result)

        # Store best metrics for later use
        if self.best_metrics is None or monitor_score < trial.study.best_value:
            self.best_metrics = valid_result
            self.best_scalability = valid_scalability

        # Log trial result
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in valid_result.items()])
        print(f"    Trial {trial.number}: {metrics_str}")

        return monitor_score

    def optimize(self, n_trials=20, timeout=None):
        """Run hyperparameter optimization."""
        study = optuna.create_study(
            direction='minimize',
            sampler=self.sampler
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        return study.best_params, study.best_value, study


def run_single_trial(params, data_root, dataset_id, return_test=False, seed_offset=0):
    """Run a single experiment with given parameters."""
    base_seed = params.get('seed', 42)
    seed_everything(seed=base_seed + seed_offset)

    # Build feature_map for this trial
    feature_map = build_feature_map(params.copy(), data_root, dataset_id)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(params.get('model_root', './checkpoints/'), dataset_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_class = eval(params['model'])
    model = model_class(feature_map, **params)

    params["data_loader"] = LongCTRDataLoader
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

    model.fit(train_gen, validation_data=valid_gen, **params)

    valid_result, valid_scalability = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    test_result = {}
    test_scalability = {"inference_time": 0, "peak_memory_mb": 0}

    if return_test and params.get("test_data"):
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result, test_scalability = model.evaluate(test_gen)

    num_params = sum(p.numel() for p in model.parameters())

    # Cleanup checkpoint
    if hasattr(model, 'checkpoint') and os.path.exists(model.checkpoint):
        os.remove(model.checkpoint)

    del feature_map
    gc.collect()

    return {
        'valid_result': valid_result,
        'test_result': test_result,
        'scalability': test_scalability,
        'num_params': num_params
    }


def save_final_results(results, dataset, model, max_seq_len, results_dir):
    """Save final repeated experiment results to CSV."""
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df['dataset'] = dataset
    df['model'] = model
    df['max_seq_len'] = max_seq_len

    # Reorder columns
    cols = ['dataset', 'model', 'max_seq_len', 'repeat'] + \
           [c for c in df.columns if c not in ['dataset', 'model', 'max_seq_len', 'repeat']]
    df = df[cols]

    csv_path = os.path.join(results_dir, f'final_{dataset}_{model}_{max_seq_len}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Final results saved to: {csv_path}")

    # Print summary
    print(f"\n  Summary (n={len(results)}):")
    for metric in ['PRAUC', 'AUC', 'logloss']:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"    {metric}: {mean_val:.6f} ± {std_val:.6f}")

    return csv_path


def load_multi_experiment_config(config_path):
    """Load multi-experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def prepare_base_params(dataset, model_name, max_seq_len, multi_config, gpu_id):
    """Prepare base parameters for experiment."""
    model_config = multi_config['model_base_configs'][model_name]
    dataset_config = multi_config['datasets'][dataset]
    tuner_space = multi_config['tuner_spaces'][model_name]

    data_root = multi_config['data_root_template'].format(
        dataset_name=dataset,
        max_user_seq_len=max_seq_len
    )

    dataset_id = f"{dataset}_maxlen{max_seq_len}"

    # Load base model config
    model_config_path = os.path.join(model_config['base_config'], 'model_config.yaml')

    with open(model_config_path, 'r') as f:
        base_model_config = yaml.safe_load(f)

    base_expid = model_config['base_expid']
    params = base_model_config.get(base_expid, {}).copy()

    # Set required default parameters
    params.setdefault('verbose', 1)
    params.setdefault('early_stop_patience', 2)
    params.setdefault('every_x_epochs', 1)
    params.setdefault('debug_mode', False)
    params.setdefault('save_best_only', True)

    # Update with dataset config
    params['dataset_id'] = dataset_id
    params['data_root'] = data_root
    params['data_format'] = dataset_config['data_format']
    params['train_data'] = f"{data_root}train_longctr.parquet"
    params['valid_data'] = f"{data_root}valid_longctr.parquet"
    params['test_data'] = f"{data_root}test_longctr.parquet"
    params['user_info'] = f"{data_root}user_info.parquet"
    params['item_info'] = f"{data_root}item_info.parquet"
    params['rebuild_dataset'] = dataset_config['rebuild_dataset']
    params['feature_cols'] = dataset_config['feature_cols']
    params['label_col'] = dataset_config['label_col']
    params['gpu'] = gpu_id

    return params, tuner_space, dataset_id, data_root


def build_feature_map(params, data_root, dataset_id):
    """Build feature map for the dataset."""
    data_dir = os.path.join(data_root)
    feature_map_json = os.path.join(data_dir, dataset_id, "feature_map.json")

    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, **params)

    feature_map = FeatureMap(dataset_id, data_dir)
    feature_map.load(feature_map_json, params)

    return feature_map


def run_single_experiment(dataset, model_name, max_seq_len, multi_config, gpu_list, n_repeats=3):
    """
    Run a single experiment for one (dataset, model, max_seq_len) combination.

    Args:
        dataset: Dataset name
        model_name: Model name
        max_seq_len: Maximum sequence length
        multi_config: Multi-experiment configuration dict
        gpu_list: List of GPU IDs
        n_repeats: Number of repeated experiments with best hyperparameters
    """
    gpu_id = gpu_list[0] if gpu_list else -1

    # Prepare parameters
    params, tuner_space, dataset_id, data_root = prepare_base_params(
        dataset, model_name, max_seq_len, multi_config, gpu_id
    )

    # Get optimization settings
    optimization = multi_config.get('optimization', {})
    n_trials = optimization.get('n_trials', 20)
    timeout = optimization.get('timeout', None)
    sampler_type = optimization.get('sampler', 'tpe')

    # Get monitor from tuner_space or optimization config
    monitor = tuner_space.get('monitor', optimization.get('monitor', {'PRAUC': 1}))

    results_dir = multi_config['results_root_template'].format(
        dataset_name=dataset,
        max_user_seq_len=max_seq_len
    )
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Optuna hyperparameter optimization
    print(f"\n  Step 1/2: Optuna hyperparameter optimization (n_trials={n_trials})")

    optimizer = FuxiCTROptimizer(
        base_params=params,
        tuner_space=tuner_space,
        data_root=data_root,
        dataset_id=dataset_id,
        monitor=monitor,
        sampler_type=sampler_type
    )

    best_params, best_score, study = optimizer.optimize(n_trials=n_trials, timeout=timeout)

    # Save optimization results
    optuna_results_path = os.path.join(results_dir, f"optuna_{dataset}_{model_name}_{max_seq_len}.csv")
    study_df = study.trials_dataframe()
    study_df.to_csv(optuna_results_path, index=False)
    print(f"  Optuna results saved to: {optuna_results_path}")
    print(f"  Best parameters: {best_params}")

    # Step 2: Repeated experiments with best hyperparameters
    print(f"\n  Step 2/2: Repeated experiments with best hyperparameters ({n_repeats} runs)")

    # Merge best params with base params
    final_params = params.copy()
    for key, value in best_params.items():
        # Convert string back to original type if needed
        if key in tuner_space and isinstance(tuner_space[key], list):
            # Find matching value in original list
            for orig_val in tuner_space[key]:
                if str(orig_val) == str(value):
                    final_params[key] = orig_val
                    break
        else:
            final_params[key] = value

    # Apply non-tunable params from tuner_space
    for key in ['batch_size', 'epochs', 'seed', 'model_root']:
        if key in tuner_space:
            final_params[key] = tuner_space[key]

    # Handle metrics: [[PRAUC, AUC, logloss]] -> [PRAUC, AUC, logloss]
    if 'metrics' in tuner_space:
        metrics_value = tuner_space['metrics']
        if isinstance(metrics_value, list) and len(metrics_value) == 1 and isinstance(metrics_value[0], list):
            final_params['metrics'] = metrics_value[0]
        else:
            final_params['metrics'] = metrics_value

    repeated_results = []
    for i in range(n_repeats):
        print(f"  Repeat {i+1}/{n_repeats}")

        result = run_single_trial(final_params, data_root, dataset_id, return_test=True, seed_offset=i)

        repeated_results.append({
            'repeat': i + 1,
            **result['test_result'],
            'inference_time': result['scalability'].get('inference_time', 0),
            'peak_memory_mb': result['scalability'].get('peak_memory_mb', 0),
            'num_params': result['num_params']
        })

    # Save final results
    save_final_results(repeated_results, dataset, model_name, max_seq_len, results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='./config/multi_experiment_config.yaml',
                       help='Path to multi-experiment configuration file')
    parser.add_argument('--n_repeats', type=int, default=3,
                       help='Number of repeated experiments with best hyperparameters')
    args = vars(parser.parse_args())

    multi_config = load_multi_experiment_config(args['config'])
    gpu_list = multi_config.get('gpu_list')
    n_repeats = args['n_repeats']

    datasets = multi_config['dataset_list']
    models = multi_config['model_list']
    max_seq_lens = multi_config['max_user_seq_len_list']

    total = len(datasets) * len(models) * len(max_seq_lens)
    count = 0

    print(f"Starting multi-experiment run: {total} total experiments")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Max sequence lengths: {max_seq_lens}")
    print(f"Repeated experiments per best config: {n_repeats}")
    print("=" * 80)

    for dataset in datasets:
        for model in models:
            for max_seq_len in max_seq_lens:
                count += 1
                print(f"\n[{count}/{total}] Running: {dataset} + {model} + maxlen{max_seq_len}")
                print("-" * 80)
                run_single_experiment(dataset, model, max_seq_len, multi_config, gpu_list, n_repeats)

    print("\n" + "=" * 80)
    print(f"All {total} experiments completed!")


if __name__ == '__main__':
    main()
