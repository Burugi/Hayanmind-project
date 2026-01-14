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
import re
import shutil
import subprocess
import pandas as pd
from pathlib import Path
import fuxictr_version
from fuxictr import autotuner

yaml.Dumper.ignore_aliases = lambda *args: True


def parse_tuner_csv(csv_path):
    """Parse tuner results CSV and return list of experiment results."""
    results = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse exp_id (stop before comma or [)
            exp_id_match = re.search(r'\[exp_id\]\s*([^,\[\]]+)', line)
            exp_id = exp_id_match.group(1).strip() if exp_id_match else None

            # Parse test metrics: "PRAUC: 0.123456 - AUC: 0.234567 - logloss: 0.345678"
            test_match = re.search(r'\[test\]\s*(.+?)\s*,?\[scalability\]', line)
            metrics = {}
            if test_match:
                test_str = test_match.group(1)
                for part in test_str.split(' - '):
                    if ':' in part:
                        key, val = part.split(':')
                        metrics[key.strip()] = float(val.strip())

            # Parse scalability
            scalability = {}
            infer_match = re.search(r'inference_time=([\d.]+)s', line)
            mem_match = re.search(r'peak_memory=([\d.]+)MB', line)
            params_match = re.search(r'num_params=(\d+)', line)

            if infer_match:
                scalability['inference_time'] = float(infer_match.group(1))
            if mem_match:
                scalability['peak_memory_mb'] = float(mem_match.group(1))
            if params_match:
                scalability['num_params'] = int(params_match.group(1))

            if exp_id and metrics:
                results.append({
                    'exp_id': exp_id,
                    'metrics': metrics,
                    'scalability': scalability
                })

    return results


def find_best_expid(csv_path, metric='PRAUC'):
    """Find the best experiment ID based on specified metric."""
    results = parse_tuner_csv(csv_path)

    best_result = max(results, key=lambda x: x['metrics'].get(metric, 0))
    return best_result['exp_id']


def run_single_expid(config_dir, expid, gpu_id):
    """Run a single experiment with specific expid."""
    cmd = f"python -u run_expid.py --config {config_dir} --expid {expid} --gpu {gpu_id}"
    subprocess.run(cmd.split(), check=True)


def run_repeated_experiments(config_dir, best_expid, gpu_id, n_repeats=3):
    """Run repeated experiments with the best hyperparameter combination."""
    results = []
    for i in range(n_repeats):
        print(f"  Repeat {i+1}/{n_repeats}: Running {best_expid}")
        run_single_expid(config_dir, best_expid, gpu_id)

        # Parse the latest result from CSV
        csv_path = os.path.basename(config_dir) + '.csv'
        all_results = parse_tuner_csv(csv_path)

        # Get the last result (most recent)
        latest = all_results[-1]
        results.append({
            'repeat': i + 1,
            **latest['metrics'],
            **latest['scalability']
        })

    return results


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


def generate_tuner_config(dataset_name, model_name, max_seq_len, multi_config):
    """
    Generate tuner config dict for a single experiment combination.

    Args:
        dataset_name: Name of the dataset (e.g., 'redkiwi_1')
        model_name: Name of the model (e.g., 'mirrn', 'twin')
        max_seq_len: Maximum sequence length (e.g., 500)
        multi_config: Multi-experiment configuration dict

    Returns:
        tuple: (tuner_config dict, dataset_id string)
    """
    model_config = multi_config['model_base_configs'][model_name]
    dataset_config = multi_config['datasets'][dataset_name]
    tuner_space = multi_config['tuner_spaces'][model_name]

    data_root = multi_config['data_root_template'].format(
        dataset_name=dataset_name,
        max_user_seq_len=max_seq_len
    )

    dataset_id = f"{dataset_name}_maxlen{max_seq_len}"

    tuner_config = {
        'base_config': model_config['base_config'],
        'base_expid': model_config['base_expid'],
        'dataset_id': dataset_id,
        'dataset_config': {
            dataset_id: {
                'data_root': data_root,
                'data_format': dataset_config['data_format'],
                'train_data': f"{data_root}train_longctr.parquet",
                'valid_data': f"{data_root}valid_longctr.parquet",
                'test_data': f"{data_root}test_longctr.parquet",
                'user_info': f"{data_root}user_info.parquet",
                'item_info': f"{data_root}item_info.parquet",
                'rebuild_dataset': dataset_config['rebuild_dataset'],
                'feature_cols': dataset_config['feature_cols'],
                'label_col': dataset_config['label_col']
            }
        },
        'tuner_space': tuner_space
    }

    return tuner_config, dataset_id


def run_single_experiment(dataset, model, max_seq_len, multi_config, gpu_list, n_repeats=3):
    """
    Run a single experiment for one (dataset, model, max_seq_len) combination.

    Args:
        dataset: Dataset name
        model: Model name
        max_seq_len: Maximum sequence length
        multi_config: Multi-experiment configuration dict
        gpu_list: List of GPU IDs
        n_repeats: Number of repeated experiments with best hyperparameters
    """
    tuner_config, dataset_id = generate_tuner_config(
        dataset, model, max_seq_len, multi_config
    )

    temp_config_path = f"./config/temp_config/temp_tuner_{dataset}_{model}_{max_seq_len}.yaml"
    os.makedirs("./config/temp_config", exist_ok=True)

    with open(temp_config_path, 'w') as f:
        yaml.dump(tuner_config, f, default_flow_style=None, indent=4)

    config_dir = autotuner.enumerate_params(temp_config_path)

    # Step 1: Grid search
    print(f"\n  Step 1/2: Grid search for hyperparameter tuning")
    autotuner.grid_search(config_dir, gpu_list)

    results_dir = multi_config['results_root_template'].format(
        dataset_name=dataset,
        max_user_seq_len=max_seq_len
    )
    os.makedirs(results_dir, exist_ok=True)

    # CSV filename is based on config_dir name (generated by run_expid.py)
    config_dir_name = os.path.basename(config_dir)
    source_csv = f"{config_dir_name}.csv"
    tuner_csv = os.path.join(results_dir, f"tuner_{dataset}_{model}_{max_seq_len}.csv")

    shutil.copy(source_csv, tuner_csv)
    print(f"  Tuner results saved to: {tuner_csv}")

    # Step 2: Find best expid and run repeated experiments
    print(f"\n  Step 2/2: Repeated experiments with best hyperparameters")
    best_expid = find_best_expid(source_csv, metric='PRAUC')
    print(f"  Best experiment ID (by PRAUC): {best_expid}")

    gpu_id = gpu_list[0] if gpu_list else -1
    repeated_results = run_repeated_experiments(config_dir, best_expid, gpu_id, n_repeats)

    # Save final results
    save_final_results(repeated_results, dataset, model, max_seq_len, results_dir)

    # Cleanup
    os.remove(source_csv)
    os.remove(temp_config_path)


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
