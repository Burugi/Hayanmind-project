import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_longctr_dataset(data_root, dataset_name, max_user_seq_len=1000):
    """
    Build LongCTR format dataset from behavior sequence data.

    Args:
        data_root: Root directory containing dataset
        dataset_name: Name of the dataset (e.g., 'redkiwi', 'alipay')
        max_user_seq_len: Maximum length of user sequence to keep

    Output files:
        - {data_root}/{dataset_name}/maxlen{max_user_seq_len}/user_info.parquet
        - {data_root}/{dataset_name}/maxlen{max_user_seq_len}/item_info.parquet
        - {data_root}/{dataset_name}/maxlen{max_user_seq_len}/train_longctr.parquet
        - {data_root}/{dataset_name}/maxlen{max_user_seq_len}/valid_longctr.parquet
        - {data_root}/{dataset_name}/maxlen{max_user_seq_len}/test_longctr.parquet
    """

    input_dir = os.path.join(data_root, dataset_name) 

    # 처리된 데이터를 저장할 폴더: {data_root}/{dataset_name}/maxlen{max_user_seq_len}
    output_dir = os.path.join(data_root, dataset_name, f"maxlen{max_user_seq_len}")
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Processing dataset: {dataset_name}")
    logging.info(f"Data directory: {input_dir, output_dir}")

    # Read all CSV files
    train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(input_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'))

    logging.info(f"Train samples: {len(train_df)}")
    logging.info(f"Valid samples: {len(valid_df)}")
    logging.info(f"Test samples: {len(test_df)}")

    # Combine all data to build complete user sequences
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # Build user sequences (chronological order based on data order)
    user_sequences = defaultdict(list)

    for idx, row in all_df.iterrows():
        user_id = row['user_id']
        seq = row['behavior_sequence']

        if pd.isna(seq) or seq == '':
            continue

        items = [int(x) for x in str(seq).split('^')]
        user_sequences[user_id].extend(items)

    logging.info(f"Total users: {len(user_sequences)}")

    # Create user_id to user_index mapping
    unique_users = sorted(user_sequences.keys())
    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}

    # Create item_id to item_index mapping (0 is reserved for padding)
    all_items = set()
    for items in user_sequences.values():
        all_items.update(items)

    unique_items = sorted(all_items)
    item_id_to_index = {iid: idx + 1 for idx, iid in enumerate(unique_items)}  # +1 for padding

    logging.info(f"Total unique items: {len(unique_items)}")

    # Build user_info: user_index -> full_item_seq
    user_info_data = []
    for user_id in unique_users:
        user_idx = user_id_to_index[user_id]
        item_seq = user_sequences[user_id]

        # Convert item_id to item_index
        item_indices = [item_id_to_index[iid] for iid in item_seq]

        # Limit sequence length
        if len(item_indices) > max_user_seq_len:
            item_indices = item_indices[-max_user_seq_len:]

        user_info_data.append({
            'user_index': user_idx,
            'full_item_seq': np.array(item_indices, dtype=np.int32)
        })

    user_info_df = pd.DataFrame(user_info_data)
    # Ensure correct data type for user_index
    user_info_df['user_index'] = user_info_df['user_index'].astype('int32')

    # Build item_info: item_index -> item features
    # For now, we only have item_index itself as a feature
    item_info_data = []
    for item_id in unique_items:
        item_idx = item_id_to_index[item_id]
        item_info_data.append({
            'item_index': int(item_idx),
            'item_id': int(item_idx)  # Use item_index for embedding (0 to N range)
        })

    # Add padding item (index 0)
    item_info_data.insert(0, {'item_index': 0, 'item_id': 0})
    item_info_df = pd.DataFrame(item_info_data)
    # Ensure correct data types
    item_info_df['item_index'] = item_info_df['item_index'].astype('int32')
    item_info_df['item_id'] = item_info_df['item_id'].astype('int32')

    # Process train/valid/test data
    def process_split(df, split_name):
        processed_data = []

        for idx, row in df.iterrows():
            user_id = row['user_id']
            seq = row['behavior_sequence']
            label = row['label']

            if pd.isna(seq) or seq == '':
                continue

            user_idx = user_id_to_index[user_id]
            items = [int(x) for x in str(seq).split('^')]

            # Last item is the target
            target_item_id = items[-1]
            target_item_idx = item_id_to_index[target_item_id]

            # Calculate sequence length (items seen before target)
            # This represents position in the full user sequence
            full_seq = user_sequences[user_id]

            # Find position of this interaction in user's full sequence
            # We use the number of items seen before the target item
            seq_len = len(items) - 1  # Exclude target item

            processed_data.append({
                'user_index': int(user_idx),  # Ensure integer type
                'item_index': int(target_item_idx),  # Ensure integer type
                'seq_len': int(min(seq_len, max_user_seq_len)),  # Ensure integer type
                'label': float(label)
            })

        return pd.DataFrame(processed_data)

    train_longctr = process_split(train_df, 'train')
    valid_longctr = process_split(valid_df, 'valid')
    test_longctr = process_split(test_df, 'test')

    logging.info(f"Processed train samples: {len(train_longctr)}")
    logging.info(f"Processed valid samples: {len(valid_longctr)}")
    logging.info(f"Processed test samples: {len(test_longctr)}")

    # Ensure correct data types
    for df in [train_longctr, valid_longctr, test_longctr]:
        df['user_index'] = df['user_index'].astype('int32')
        df['item_index'] = df['item_index'].astype('int32')
        df['seq_len'] = df['seq_len'].astype('int32')
        df['label'] = df['label'].astype('float32')

    # Save files
    user_info_path = os.path.join(output_dir, 'user_info.parquet')
    item_info_path = os.path.join(output_dir, 'item_info.parquet')
    train_path = os.path.join(output_dir, 'train_longctr.parquet')
    valid_path = os.path.join(output_dir, 'valid_longctr.parquet')
    test_path = os.path.join(output_dir, 'test_longctr.parquet')

    user_info_df.to_parquet(user_info_path, index=False)
    item_info_df.to_parquet(item_info_path, index=False)
    train_longctr.to_parquet(train_path, index=False)
    valid_longctr.to_parquet(valid_path, index=False)
    test_longctr.to_parquet(test_path, index=False)

    logging.info(f"Saved user_info to: {user_info_path}")
    logging.info(f"Saved item_info to: {item_info_path}")
    logging.info(f"Saved train data to: {train_path}")
    logging.info(f"Saved valid data to: {valid_path}")
    logging.info(f"Saved test data to: {test_path}")

    # Calculate vocab_size for config
    num_users = len(user_info_df)
    num_items = len(item_info_df)  # Includes padding (index 0)

    # Save metadata for config
    import json
    metadata = {
        'num_users': num_users,
        'num_items': num_items,
        'vocab_size_user_index': num_users,
        'vocab_size_item_index': num_items,
        'vocab_size_item_id': num_items,
        'max_seq_len': max_user_seq_len
    }

    metadata_path = os.path.join(output_dir, 'longctr_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved metadata to: {metadata_path}")

    # Print statistics
    logging.info("\n=== Dataset Statistics ===")
    logging.info(f"Number of users: {num_users}")
    logging.info(f"Number of items (including padding): {num_items}")
    logging.info(f"Average user sequence length: {user_info_df['full_item_seq'].apply(len).mean():.2f}")
    logging.info(f"Max user sequence length: {user_info_df['full_item_seq'].apply(len).max()}")
    logging.info(f"Min user sequence length: {user_info_df['full_item_seq'].apply(len).min()}")

    logging.info("\n=== Config Information ===")
    logging.info(f"Add these vocab_size values to your dataset_config.yaml:")
    logging.info(f"  - user_index vocab_size: {num_users}")
    logging.info(f"  - item_index vocab_size: {num_items}")
    logging.info(f"  - item_id vocab_size: {num_items}")

    return {
        'user_info': user_info_path,
        'item_info': item_info_path,
        'train': train_path,
        'valid': valid_path,
        'test': test_path,
        'metadata': metadata_path,
        'vocab_sizes': metadata
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build LongCTR format dataset')
    parser.add_argument('--data_root', type=str, default='../data/',
                        help='Root directory containing datasets')
    parser.add_argument('--dataset', type=str, default='redkiwi',
                        help='Dataset name (e.g., redkiwi, alipay)')
    parser.add_argument('--max_seq_len', type=int, nargs='+', default=[36, 48, 60, 96, 192, 336, 720], 
                    help='Maximum user sequence length(s). e.g. --max_seq_len 200 500 1000')

    args = parser.parse_args()

    all_results = {}

    for m in args.max_seq_len:
        paths = build_longctr_dataset(
            data_root=args.data_root,
            dataset_name=args.dataset,
            max_user_seq_len=m
        )
        all_results[m] = paths

    print("\n=== Generated Files (by max_seq_len) ===")
    for m, paths in all_results.items():
        print(f"\n--- max_seq_len={m} ---")
        for key, path in paths.items():
            print(f"{key}: {path}")