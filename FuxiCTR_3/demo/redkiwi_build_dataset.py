import sys
sys.path.append('../')
import os
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger
from fuxictr.preprocess import FeatureProcessor, build_dataset

if __name__ == '__main__':
    config_dir = './config/redkiwi_config'
    experiment_id = 'DIN_redkiwi'
    params = load_config(config_dir, experiment_id)

    set_logger(params)
    feature_cols = params['feature_cols']
    label_col = params['label_col']

    feature_processor = FeatureProcessor(
        feature_cols=feature_cols,
        label_col=label_col,
        dataset_id=params['dataset_id'],
        data_root=params['data_root']
    )

    params['train_data'] = build_dataset(
        feature_processor,
        train_data=params['train_data'],
        valid_data=params['valid_data'],
        test_data=params['test_data'],
        data_format=params.get('data_format', 'csv'),
        min_categr_count=params.get('min_categr_count', 1)
    )
