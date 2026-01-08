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


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import torch


class ParquetDataset(Dataset):
    def __init__(self, data_path, sampling_method=None, target_pos_ratio=None, random_seed=2024):
        self.column_index = dict()
        self.sampling_method = sampling_method
        self.target_pos_ratio = target_pos_ratio
        self.random_seed = random_seed
        self.darray, self.label_col_idx = self.load_data(data_path)
        self.original_indices = np.arange(len(self.darray))
        self.sampled_indices = None
        if sampling_method is not None and sampling_method.lower() != 'none':
            self.apply_sampling()

    def __getitem__(self, index):
        if self.sampled_indices is not None:
            actual_index = self.sampled_indices[index]
        else:
            actual_index = index
        return self.darray[actual_index, :]

    def __len__(self):
        if self.sampled_indices is not None:
            return len(self.sampled_indices)
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        data_arrays = []
        idx = 0
        label_col_idx = None
        for col in df.columns:
            if col == 'label':
                label_col_idx = idx
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
                seq_len = array.shape[1]
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            else:
                array = df[col].to_numpy()
                self.column_index[col] = idx
                idx += 1
            data_arrays.append(array)
        return np.column_stack(data_arrays), label_col_idx

    def apply_sampling(self):
        """Apply oversampling or undersampling based on sampling_method."""
        if self.label_col_idx is None:
            return

        labels = self.darray[:, self.label_col_idx]
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        pos_count = len(pos_indices)
        neg_count = len(neg_indices)

        if pos_count == 0 or neg_count == 0:
            return

        np.random.seed(self.random_seed)

        if self.sampling_method.lower() == 'oversample':
            if self.target_pos_ratio is not None:
                target_pos_count = int(neg_count * self.target_pos_ratio / (1 - self.target_pos_ratio))
            else:
                target_pos_count = neg_count
            sampled_pos_indices = np.random.choice(pos_indices, size=target_pos_count, replace=True)
            self.sampled_indices = np.concatenate([sampled_pos_indices, neg_indices])
        elif self.sampling_method.lower() == 'undersample':
            if self.target_pos_ratio is not None:
                target_neg_count = int(pos_count * (1 - self.target_pos_ratio) / self.target_pos_ratio)
            else:
                target_neg_count = pos_count
            sampled_neg_indices = np.random.choice(neg_indices, size=min(target_neg_count, neg_count), replace=False)
            self.sampled_indices = np.concatenate([pos_indices, sampled_neg_indices])

        if self.sampled_indices is not None:
            np.random.shuffle(self.sampled_indices)

    def resample(self):
        """Resample data with a new random seed for each epoch."""
        if self.sampling_method is not None and self.sampling_method.lower() != 'none':
            self.random_seed += 1
            self.apply_sampling()


class LongCTRDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, user_info, item_info, batch_size=32, shuffle=False,
                 num_workers=1, max_len=50, padding="pre", split='train', stage=None,
                 sampling_method=None, target_pos_ratio=None,
                 **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"

        if stage is None:
            stage = split

        train_sampling = sampling_method if stage == 'train' else None
        self.dataset = ParquetDataset(data_path, sampling_method=train_sampling,
                                     target_pos_ratio=target_pos_ratio)
        column_index = self.dataset.column_index
        self.stage = stage

        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=BatchCollator(feature_map, max_len, column_index,
                                     user_info, item_info, padding, stage=stage)
        )
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches

    def resample(self):
        """Resample dataset for next epoch (only for training with sampling)."""
        self.dataset.resample()


class BatchCollator(object):
    def __init__(self, feature_map, max_len, column_index, user_info, item_info, padding="pre",
                 stage='train'):
        self.feature_map = feature_map
        self.user_info = pd.read_parquet(user_info)["full_item_seq"].values
        self.item_info = pd.read_parquet(item_info).set_index("item_index")
        self.max_len = max_len
        self.column_index = column_index
        self.padding = padding
        self.stage = stage

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]

        user_index = batch_dict["user_index"].long().numpy()
        user_seqs = self.user_info[user_index]
        seq_lens = batch_dict["seq_len"].long().numpy()
        batch_seqs = self.padding_seqs(user_seqs, seq_lens)
        mask = (torch.from_numpy(batch_seqs) > 0).float()
        item_index = batch_dict["item_index"].long().numpy().reshape(-1, 1)
        batch_items = np.hstack([batch_seqs, item_index]).flatten()
        item_info = self.item_info.loc[batch_items]
        item_dict = dict()
        for col in item_info.columns:
            if col in all_cols:
                item_dict[col] = torch.from_numpy(np.array(item_info[col].to_list()))

        return batch_dict, item_dict, mask

    def padding_seqs(self, user_seqs, seq_lens):
        batch_seqs = []
        for seq, l in zip(user_seqs, seq_lens):
            batch_seqs.append(seq[:l])
        batch_seqs = pad_sequences(batch_seqs, maxlen=self.max_len,
                                   value=0, padding=self.padding, truncating=self.padding)
        return batch_seqs
