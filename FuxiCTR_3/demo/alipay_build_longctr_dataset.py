import sys
sys.path.append('../')
import os

# Reuse the same build function from redkiwi
from redkiwi_build_longctr_dataset import build_longctr_dataset

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build LongCTR format dataset for Alipay')
    parser.add_argument('--data_root', type=str, default='../data/',
                        help='Root directory containing datasets')
    parser.add_argument('--max_seq_len', type=int, default=1000,
                        help='Maximum user sequence length')

    args = parser.parse_args()

    paths = build_longctr_dataset(
        data_root=args.data_root,
        dataset_name='alipay',
        max_user_seq_len=args.max_seq_len
    )

    print("\n=== Generated Files ===")
    for key, path in paths.items():
        print(f"{key}: {path}")
