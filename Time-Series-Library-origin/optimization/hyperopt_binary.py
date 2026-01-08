import os
import yaml
import json
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from exp.exp_binary_classification import Exp_Binary_Classification
from utils.tools import EarlyStopping
import copy


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna for binary classification.
    """
    def __init__(self, args, model_name, common_config_path, model_config_path,
                 save_dir, sampler_type='tpe'):
        self.base_args = copy.deepcopy(args)
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        with open(common_config_path, 'r') as f:
            self.common_config = yaml.safe_load(f)

        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)

        if sampler_type == 'tpe':
            self.sampler = TPESampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        self.device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')

    def _suggest_parameters(self, trial):
        """Suggest hyperparameters for a trial."""
        params = {}

        for param_name, param_config in self.common_config['common_params'].items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )

        if 'model_params' in self.model_config and self.model_config['model_params']:
            for param_name, param_config in self.model_config['model_params'].items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['min'],
                        param_config['max'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

        if 'seg_len' in params:
            seq_len = self.base_args.seq_len
            # seg_len must divide seq_len evenly
            if seq_len % params['seg_len'] != 0:
                # Find the largest valid seg_len that divides seq_len
                valid_seg_lens = [sl for sl in [8, 16, 32, 64] if sl <= seq_len and seq_len % sl == 0]
                params['seg_len'] = valid_seg_lens[-1] if valid_seg_lens else 8

        return params

    def objective(self, trial):
        """Optuna objective function."""
        args = copy.deepcopy(self.base_args)
        params = self._suggest_parameters(trial)

        for key, value in params.items():
            setattr(args, key, value)

        exp = Exp_Binary_Classification(args)
        setting = f'{self.model_name}_trial_{trial.number}'

        train_data, train_loader = exp._get_data(flag='train')
        vali_data, vali_loader = exp._get_data(flag='val')

        model_optim = exp._select_optimizer()
        criterion = exp._select_criterion()

        patience = self.common_config.get('patience', 10)
        max_epochs = self.common_config.get('max_epochs', 50)

        early_stopping = EarlyStopping(patience=patience, verbose=False)

        for epoch in range(max_epochs):
            exp.model.train()
            train_loss = []

            for batch_x, label, padding_mask in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(exp.device)
                padding_mask = padding_mask.float().to(exp.device)
                label = label.to(exp.device)

                outputs = exp.model(batch_x, padding_mask, None, None, None)
                loss = criterion(outputs.squeeze(), label.squeeze())
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            avg_train_loss = np.mean(train_loss)
            vali_loss, vali_metrics = exp.vali(vali_data, vali_loader, criterion)

            print(f"    Trial {trial.number} | Epoch {epoch+1}/{max_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {vali_loss:.4f} | "
                  f"Val PRAUC: {vali_metrics['PRAUC']:.4f}")

            temp_path = os.path.join(self.save_dir, f'temp_trial_{trial.number}')
            os.makedirs(temp_path, exist_ok=True)
            early_stopping(-vali_metrics['PRAUC'], exp.model, temp_path)

            if early_stopping.early_stop:
                break

            trial.report(-vali_metrics['PRAUC'], epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        checkpoint_path = os.path.join(temp_path, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(temp_path) and not os.listdir(temp_path):
            os.rmdir(temp_path)

        return -vali_metrics['PRAUC']

    def optimize(self, n_trials=None, timeout=None):
        """
        Run hyperparameter optimization.
        """
        if n_trials is None:
            n_trials = self.common_config.get('n_trials', 20)
        if timeout is None:
            timeout = self.common_config.get('timeout', None)

        study = optuna.create_study(
            direction='minimize',
            sampler=self.sampler,
            study_name=f'{self.model_name}_optimization'
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        results = {
            'best_params': study.best_params,
            'best_value': -study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials)
        }

        result_path = os.path.join(self.save_dir, 'hyperopt_results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f'\nOptimization completed!')
        print(f'Best params: {results["best_params"]}')
        print(f'Best PRAUC: {results["best_value"]:.6f}')
        print(f'Results saved to: {result_path}')

        return results['best_params'], results['best_value']
