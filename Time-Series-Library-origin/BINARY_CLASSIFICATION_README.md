# Binary Classification for User Behavior Sequences

Time-Series-Library를 확장하여 FuxiCTR 스타일의 사용자 행동 시퀀스 기반 이진 분류를 지원합니다.

## 개요

이 구현은 사용자의 과거 행동 시퀀스와 타겟 아이템을 입력으로 받아 구매 여부(0/1)를 예측하는 이진 분류 시스템입니다.

### 주요 기능

1. **FuxiCTR 호환 데이터 로더**: Parquet 형식의 데이터 지원
2. **Binary Classification Metrics**: PRAUC, AUC, LogLoss 지원
3. **Optuna 기반 하이퍼파라미터 튜닝**: 자동화된 최적화
4. **Multi-Experiment Runner**: 여러 데이터셋, 모델, 시퀀스 길이 조합 실험
5. **Scalability Metrics**: 학습 시간, 추론 시간, 모델 크기 등 추적

## 데이터 형식

### 필수 파일

각 데이터셋은 다음 파일들을 포함해야 합니다:

```
data/{dataset_name}/maxlen{seq_len}/
├── train_longctr.parquet
├── valid_longctr.parquet
├── test_longctr.parquet
├── user_info.parquet
└── item_info.parquet
```

### 파일 구조

**train/valid/test_longctr.parquet**:
- `user_index`: 사용자 인덱스 (int)
- `item_index`: 타겟 아이템 인덱스 (int)
- `seq_len`: 실제 시퀀스 길이 (int)
- `label`: 레이블 (0 또는 1)

**user_info.parquet**:
- `full_item_seq`: 사용자의 전체 행동 시퀀스 (list of int)

**item_info.parquet**:
- `item_index`: 아이템 인덱스 (int)
- `item_id`: 아이템 ID (int)

## 설치

필요한 패키지 설치:

```bash
pip install pandas pyarrow scikit-learn optuna keras-preprocessing
```

## 사용 방법

### 1. Config 파일 설정

`configs/multi_experiment_config.yaml`을 편집하여 실험 설정:

```yaml
dataset_list:
  - redkiwi32

model_list:
  - DLinear
  - WPMixer
  - SegRNN
  - SCINet
  - Informer

seq_len_list:
  - 64
  - 128
  - 256
  - 512

data_root_template: "./data/{dataset_name}/maxlen{seq_len}/"
results_root_template: "./results/{dataset_name}/{seq_len}/"
```

### 2. Multi-Experiment 실행

```bash
python run_multi_experiment.py --config configs/multi_experiment_config.yaml
```

이 스크립트는:
1. 각 (데이터셋, 모델, seq_len) 조합에 대해 Optuna로 하이퍼파라미터 최적화
2. 최적 하이퍼파라미터로 최종 학습
3. 테스트 세트에서 평가
4. 결과를 CSV로 저장

### 3. 결과 확인

결과는 다음 위치에 저장됩니다:

```
results/{dataset_name}/{seq_len}/results_seqlen{seq_len}.csv
```

CSV 파일에는 다음 정보가 포함됩니다:
- `PRAUC`, `AUC`, `LogLoss`: 성능 지표
- `train_time_sec`, `inference_time_sec`: Scalability 지표
- `num_params`, `model_size_mb`: 모델 크기 정보
- 최적 하이퍼파라미터 값들

## 지원 모델

현재 다음 모델들이 구성되어 있습니다:

1. **DLinear**: 선형 모델
2. **WPMixer**: Wavelet + MLP-Mixer
3. **SegRNN**: Segment-wise RNN
4. **SCINet**: Sample Convolution and Interaction Network
5. **Informer**: Transformer 기반 모델

각 모델의 하이퍼파라미터는 `configs/models/{model_name}.yaml`에 정의되어 있습니다.

## 새 모델 추가

1. `configs/models/NewModel.yaml` 생성:

```yaml
model_params:
  param1:
    type: categorical
    choices: [option1, option2]
  param2:
    type: float
    min: 0.0
    max: 1.0
```

2. `configs/multi_experiment_config.yaml`의 `model_list`에 추가
3. `model_configs`에 기본 설정 추가

## 새 데이터셋 추가

1. FuxiCTR 형식으로 데이터 준비
2. `configs/multi_experiment_config.yaml`에 데이터셋 설정 추가:

```yaml
datasets:
  new_dataset:
    data: longctr
    task_name: binary_classification
    features: S
    target: label
```

3. `dataset_list`에 추가

## 주요 차이점: FuxiCTR vs Time-Series-Library

| 특징 | FuxiCTR | 이 구현 |
|------|---------|---------|
| Time Encoding | 사용 안 함 | 사용 안 함 (요청사항) |
| Padding | keras pad_sequences (pre) | 동일 |
| Metrics | PRAUC, AUC, LogLoss | 동일 |
| Optimization | Grid Search | Optuna (Bayesian) |
| 결과 저장 | CSV | CSV (FuxiCTR 스타일) |

## 파일 구조

```
Time-Series-Library-origin/
├── configs/
│   ├── hyperopt_config.yaml           # 공통 하이퍼파라미터
│   ├── multi_experiment_config.yaml   # 멀티 실험 설정
│   └── models/
│       ├── DLinear.yaml
│       ├── WPMixer.yaml
│       ├── SegRNN.yaml
│       ├── SCINet.yaml
│       └── Informer.yaml
├── data_provider/
│   ├── data_loader.py                 # Dataset_LongCTR 추가됨
│   └── data_factory.py                # binary_classification 지원
├── exp/
│   └── exp_binary_classification.py   # Binary Classification Experiment
├── optimization/
│   └── hyperopt_binary.py             # Optuna 최적화
├── utils/
│   └── metrics.py                     # PRAUC, AUC, LogLoss 추가됨
├── run_multi_experiment.py            # 멀티 실험 실행
└── BINARY_CLASSIFICATION_README.md    # 이 파일
```

## 예제: 단일 실험 실행

Optuna 최적화 없이 단일 실험만 실행하려면:

```python
import torch
from exp.exp_binary_classification import Exp_Binary_Classification
from run_multi_experiment import Args, create_base_args

# Arguments 설정
args = Args(
    task_name='binary_classification',
    model='DLinear',
    data='longctr',
    root_path='./data/redkiwi32/maxlen128/',
    seq_len=128,
    batch_size=256,
    learning_rate=0.001,
    train_epochs=100,
    patience=10,
    use_gpu=True,
    gpu=0,
    # ... 기타 필요한 인자들
)

# Experiment 생성 및 실행
exp = Exp_Binary_Classification(args)
setting = 'DLinear_redkiwi32_128'

# 학습
exp.train(setting)

# 테스트
exp.test(setting, test=1)
```

## 참고사항

1. **Padding 방식**: FuxiCTR과 동일하게 `keras.preprocessing.sequence.pad_sequences`의 `pre` 패딩 사용
2. **Time Encoding 없음**: 유저 행동 시퀀스에는 datetime 정보가 없으므로 time encoding 비활성화
3. **Scalability 지표**: 모든 실험에서 학습/추론 시간, 모델 크기 자동 기록
4. **재현성**: 랜덤 시드는 Optuna sampler에서 42로 고정

## 문제 해결

### 메모리 부족
- `batch_size` 감소
- `seq_len` 감소
- 모델 파라미터 감소 (d_model, d_ff 등)

### 학습이 느림
- `n_trials` 감소
- `max_epochs` 감소
- `num_workers` 증가 (데이터 로딩 병렬화)

### 데이터 로딩 오류
- Parquet 파일 경로 확인
- 데이터 형식 확인 (user_index, item_index, seq_len, label)
- user_info.parquet의 full_item_seq가 리스트 형식인지 확인
