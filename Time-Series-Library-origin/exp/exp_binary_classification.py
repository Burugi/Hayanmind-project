from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import binary_classification_metrics
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Binary_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Binary_Classification, self).__init__(args)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='train')
        self.args.enc_in = 1
        self.args.dec_in = 1
        self.args.c_out = 1
        self.args.pred_len = 0
        self.args.num_class = 1

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None, None)

                pred = outputs.detach()
                loss = criterion(pred.squeeze(), label.squeeze())
                total_loss.append(loss.item())

                preds.append(torch.sigmoid(pred.squeeze()).cpu().numpy())
                trues.append(label.squeeze().cpu().numpy())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        metrics = binary_classification_metrics(trues, preds)

        self.model.train()
        return total_loss, metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None, None)
                loss = criterion(outputs.squeeze(), label.squeeze())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss, vali_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.3f} "
                f"Vali Loss: {vali_loss:.3f} Vali PRAUC: {vali_metrics['PRAUC']:.4f} Vali AUC: {vali_metrics['AUC']:.4f} "
                f"Test Loss: {test_loss:.3f} Test PRAUC: {test_metrics['PRAUC']:.4f} Test AUC: {test_metrics['AUC']:.4f}"
            )

            early_stopping(-vali_metrics['PRAUC'], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0, return_metrics=False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        start_time = time.time()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None, None)

                preds.append(torch.sigmoid(outputs.squeeze()).cpu().numpy())
                trues.append(label.squeeze().cpu().numpy())

        inference_time = time.time() - start_time
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)

        metrics = binary_classification_metrics(trues, preds)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        print(f'PRAUC: {metrics["PRAUC"]:.6f}')
        print(f'AUC: {metrics["AUC"]:.6f}')
        print(f'LogLoss: {metrics["LogLoss"]:.6f}')
        print(f'Inference time: {inference_time:.2f}s')

        if return_metrics:
            return metrics, inference_time

        file_name = 'result_binary_classification.txt'
        with open(os.path.join(folder_path, file_name), 'a') as f:
            f.write(setting + "  \n")
            f.write(f'PRAUC: {metrics["PRAUC"]:.6f}\n')
            f.write(f'AUC: {metrics["AUC"]:.6f}\n')
            f.write(f'LogLoss: {metrics["LogLoss"]:.6f}\n')
            f.write(f'Inference time: {inference_time:.2f}s\n')
            f.write('\n')
