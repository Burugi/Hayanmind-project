import argparse
import yaml
import os
import time
import pandas as pd
import torch
from optimization.hyperopt_binary import HyperparameterOptimizer
from exp.exp_binary_classification import Exp_Binary_Classification
from utils.print_args import print_args
import copy


class Args:
    """Simple args container"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_multi_experiment_config(config_path):
    """Load multi-experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_base_args(dataset_config, model_config, common_settings, seq_len, root_path):
    """Create base arguments for experiment with all run.py arguments."""
    args_dict = {
        # basic config
        'task_name': dataset_config['task_name'],
        'is_training': 1,
        'model_id': f"{model_config['model']}_{dataset_config['data']}_{seq_len}",
        'model': model_config['model'],

        # data loader
        'data': dataset_config['data'],
        'root_path': root_path,
        'data_path': 'ETTh1.csv',
        'features': dataset_config.get('features', 'S'),
        'target': dataset_config.get('target', 'label'),
        'freq': 'h',
        'checkpoints': './checkpoints/',

        # forecasting task
        'seq_len': seq_len,
        'label_len': 0,
        'pred_len': 0,
        'seasonal_patterns': 'Monthly',
        'inverse': False,

        # imputation task
        'mask_rate': 0.25,

        # anomaly detection task
        'anomaly_ratio': 0.25,

        # model define
        'expand': 2,
        'd_conv': 4,
        'top_k': 5,
        'num_kernels': 6,
        'enc_in': 1,
        'dec_in': 1,
        'c_out': 1,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'moving_avg': 25,
        'factor': model_config.get('factor', 1),
        'distil': model_config.get('distil', True),
        'dropout': 0.1,
        'embed': model_config.get('embed', 'timeF'),
        'activation': 'gelu',
        'channel_independence': 1,
        'decomp_method': 'moving_avg',
        'use_norm': 1,
        'down_sampling_layers': 0,
        'down_sampling_window': 1,
        'down_sampling_method': None,
        'seg_len': model_config.get('seg_len', 96),

        # optimization
        'num_workers': common_settings.get('num_workers', 0),
        'itr': 1,
        'train_epochs': common_settings.get('train_epochs', 100),
        'batch_size': common_settings.get('batch_size', 256),
        'patience': common_settings.get('patience', 10),
        'learning_rate': common_settings.get('learning_rate', 0.001),
        'des': 'test',
        'loss': 'MSE',
        'lradj': 'type1',
        'use_amp': False,

        # GPU
        'use_gpu': common_settings.get('use_gpu', True),
        'gpu': common_settings.get('gpu', 0),
        'gpu_type': 'cuda',
        'use_multi_gpu': False,
        'devices': '0,1,2,3',
        'device_ids': [0],

        # de-stationary projector params
        'p_hidden_dims': [128, 128],
        'p_hidden_layers': 2,

        # metrics (dtw)
        'use_dtw': False,

        # Augmentation
        'augmentation_ratio': 0,
        'seed': 2,
        'jitter': False,
        'scaling': False,
        'permutation': False,
        'randompermutation': False,
        'magwarp': False,
        'timewarp': False,
        'windowslice': False,
        'windowwarp': False,
        'rotation': False,
        'spawner': False,
        'dtwwarp': False,
        'shapedtwwarp': False,
        'wdba': False,
        'discdtw': False,
        'discsdtw': False,
        'extra_tag': '',

        # TimeXer / patch
        'patch_len': 16,

        # GCN
        'node_dim': 10,
        'gcn_depth': 2,
        'gcn_dropout': 0.3,
        'propalpha': 0.3,
        'conv_channel': 32,
        'skip_channel': 32,

        # DLinear
        'individual': model_config.get('individual', False),

        # TimeFilter
        'alpha': 0.1,
        'top_p': 0.5,
        'pos': 1,

        # SCINet specific
        'hid_size': model_config.get('hid_size', 1.0),
        'num_levels': model_config.get('num_levels', 3),
        'num_decoder_layer': 1,
        'concat_len': 0,
        'groups': model_config.get('groups', 1),
        'kernel': model_config.get('kernel', 5),
        'positionalE': False,
        'modified': True,
        'RIN': False,

        # Custom for binary classification
        'padding': 'pre',
    }

    return Args(**args_dict)


def run_optimization(dataset, model, seq_len, multi_config):
    """Run hyperparameter optimization for one experiment."""
    dataset_config = multi_config['datasets'][dataset]
    model_config = multi_config['model_configs'][model]
    common_settings = multi_config['common_settings']
    optimization_config = multi_config['optimization']

    root_path = multi_config['data_root_template'].format(
        dataset_name=dataset,
        seq_len=seq_len
    )

    args = create_base_args(dataset_config, model_config, common_settings, seq_len, root_path)

    save_dir = f'./hyperopt_results/{dataset}/{seq_len}/{model}'
    os.makedirs(save_dir, exist_ok=True)

    common_config_path = './configs/hyperopt_config.yaml'
    model_config_path = f'./configs/models/{model}.yaml'

    optimizer = HyperparameterOptimizer(
        args=args,
        model_name=model,
        common_config_path=common_config_path,
        model_config_path=model_config_path,
        save_dir=save_dir,
        sampler_type=optimization_config.get('sampler', 'tpe')
    )

    best_params, best_prauc = optimizer.optimize(
        n_trials=optimization_config.get('n_trials', 20),
        timeout=optimization_config.get('timeout', None)
    )

    return best_params, best_prauc


def run_final_training(dataset, model, seq_len, best_params, multi_config):
    """Run final training with best hyperparameters and collect scalability metrics."""
    dataset_config = multi_config['datasets'][dataset]
    model_config = multi_config['model_configs'][model]
    common_settings = multi_config['common_settings']

    root_path = multi_config['data_root_template'].format(
        dataset_name=dataset,
        seq_len=seq_len
    )

    args = create_base_args(dataset_config, model_config, common_settings, seq_len, root_path)

    for key, value in best_params.items():
        setattr(args, key, value)

    setting = f'{model}_{dataset}_seqlen{seq_len}_final'
    exp = Exp_Binary_Classification(args)

    print(f'\n{"="*80}')
    print(f'Final Training: {dataset} + {model} + seq_len={seq_len}')
    print(f'{"="*80}')

    start_train_time = time.time()
    exp.train(setting)
    train_time = time.time() - start_train_time

    test_metrics, inference_time = exp.test(setting, test=1, return_metrics=True)

    num_params = sum(p.numel() for p in exp.model.parameters())
    num_trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)

    model_size_mb = num_params * 4 / (1024 ** 2)

    results = {
        'dataset': dataset,
        'model': model,
        'seq_len': seq_len,
        'PRAUC': test_metrics['PRAUC'],
        'AUC': test_metrics['AUC'],
        'LogLoss': test_metrics['LogLoss'],
        'train_time_sec': train_time,
        'inference_time_sec': inference_time,
        'num_params': num_params,
        'num_trainable_params': num_trainable_params,
        'model_size_mb': model_size_mb,
        **best_params
    }

    return results


def save_results(results_list, dataset, seq_len, multi_config):
    """Save results to CSV file."""
    results_dir = multi_config['results_root_template'].format(
        dataset_name=dataset,
        seq_len=seq_len
    )
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(results_list)
    csv_path = os.path.join(results_dir, f'results_seqlen{seq_len}.csv')

    df.to_csv(csv_path, index=False)
    print(f'\nResults saved to: {csv_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='./configs/multi_experiment_config.yaml',
                       help='Path to multi-experiment configuration file')
    args = parser.parse_args()

    multi_config = load_multi_experiment_config(args.config)

    datasets = multi_config['dataset_list']
    models = multi_config['model_list']
    seq_lens = multi_config['seq_len_list']

    total = len(datasets) * len(models) * len(seq_lens)
    count = 0

    print(f"Starting multi-experiment run: {total} total experiments")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Sequence lengths: {seq_lens}")
    print("=" * 80)

    for dataset in datasets:
        for seq_len in seq_lens:
            results_list = []

            for model in models:
                count += 1
                print(f"\n[{count}/{total}] Running: {dataset} + {model} + seq_len={seq_len}")
                print("-" * 80)

                try:
                    print(f"\nStep 1: Hyperparameter Optimization")
                    best_params, best_prauc = run_optimization(dataset, model, seq_len, multi_config)

                    print(f"\nStep 2: Final Training with Best Hyperparameters")
                    results = run_final_training(dataset, model, seq_len, best_params, multi_config)
                    results_list.append(results)

                    print(f"\nCompleted {dataset} + {model} + seq_len={seq_len}")
                    print(f"Best PRAUC: {results['PRAUC']:.6f}")
                    print(f"AUC: {results['AUC']:.6f}")
                    print(f"LogLoss: {results['LogLoss']:.6f}")

                except Exception as e:
                    print(f"\nError in {dataset} + {model} + seq_len={seq_len}: {str(e)}")
                    continue

            save_results(results_list, dataset, seq_len, multi_config)

    print("\n" + "=" * 80)
    print(f"All {total} experiments completed!")


if __name__ == '__main__':
    main()
