# FuxiCTR Notebook Pipeline — 개발 노트

여러 세션에 걸쳐 `notebook/` 아래 6개 실험 노트북 + 공용 헬퍼(`utils/nb_utils.py`)를
구축하며 만난 함정과 설계 원칙을 정리한 문서. 새 세션은 **이 문서를 먼저 읽은 뒤**
아래 "노트북별 가이드"의 해당 섹션을 확장하는 방식으로 작업한다.

---

## 0. 공통 기반

### 디렉토리 규약
```
notebook/
  01_Build_dataset.ipynb     # raw → FuxiCTR parquet
  02_EDA.ipynb               # 전처리 산출물 EDA
  03_Tuning_Run.ipynb        # Optuna + N_REPEATS 최종 학습
  04_Results_analysis.ipynb  # subgroup/오분류/시퀀스 분석
  05_New_model_test.ipynb    # 새 모델 실험 (셀 안에서 정의)
  06_Results_viz.ipynb       # run 메트릭 시각화
  configs/
    datasets/{dataset_id}.yaml   # 01이 생성
    models/{MODEL}_base.yaml     # 모델별 기본 하이퍼
    tuning/{MODEL}_search.yaml   # Optuna 탐색공간
  utils/
    nb_utils.py              # 공용 헬퍼
  artifacts/
    runs/{dataset_id}/{model_id}.{model,metrics.json,params.json}
    predictions/{dataset_id}/{model_name}/{run_id}/preds.parquet
    tuning/{dataset_id}/{model_name}/study.db   # Optuna SQLite
    analysis/, figures/
data/
  raw_data/{dataset}/...      # 원본 CSV/parquet
  {dataset_id}/               # FuxiCTR 산출물
    train.parquet, valid.parquet, test.parquet
    feature_map.json, feature_processor.pkl, feature_vocab.json
```

### `nb_utils.py` 설계 원칙
- **모델/데이터셋 특이 로직은 절대 넣지 않는다.** 모든 특이성은 YAML로 표현.
  → 과거 "DIN 전용 list→tuple 변환"을 nb_utils에 넣었다가 사용자 지적 후
    YAML `!!python/tuple` + `yaml.FullLoader`로 대체한 이력 있음.
- FuxiCTR 산출물 경로, 러너(`run_training`), Optuna 헬퍼, 예측 저장 등
  **라이브러리 수준 공용 함수**만 둔다.
- 경로 상수: `PROJECT_ROOT`, `NOTEBOOK_ROOT`, `DATA_ROOT`, `CONFIG_ROOT`, `ARTIFACT_ROOT`.

### 실행 환경
- 사용자는 **원격 서버**에서 실행, 로컬에서는 정적 검증만. GPU 가정(`gpu=0`).
- Claude 측은 YAML 파싱/경로/구문 검증까지. 런타임 검증은 사용자 로그 공유로.
- 각 노트북은 독립 실행 가능하되 산출물 경로 규약으로 느슨히 결합.

### Python 환경 주의
- scikit-learn 1.4+ 가정 → `log_loss(eps=...)` 사용 불가.
  `fuxictr/metrics.py:31` 은 이미 `np.clip(..., 1e-7, 1-1e-7)` 로 패치됨.
  **새로 clone한 FuxiCTR를 쓰면 이 패치가 사라지므로 재적용 필요.**
- pandas 2.x 가정.
- yaml FullLoader 사용 (`!!python/tuple` 지원 위함).

---

## 1. 지난 세션 시행착오 요약

향후 작업에서 **재발 방지**를 위해 꼭 기억할 항목.

| # | 증상 | 원인 | 해결 |
|---|------|------|------|
| 1 | `optuna UserWarning: choices ... is of type list` + resume 깨짐 | Optuna SQLite는 categorical choice로 primitive만 영속화 | `suggest_from_space`가 list/dict choice를 JSON 인코딩 → `suggest_categorical` → 디코딩 |
| 2 | DIN `RuntimeError: size of tensor a vs b mismatch` (여러 번) | 시퀀스 필드가 `FeatureProcessor`에서 기본 `MaskedAveragePooling`으로 pre-pool 되어 DIN attention이 받을 raw `[B, L, D]`가 아닌 `[B, D]` 수령 | 모델 YAML의 `Base`에 `feature_specs: [{name: <seq>, feature_encoder: null}]` 추가. 이는 FuxiCTR 표준(`fuxictr/features.py:54`, `demo/example6_config`) |
| 3 | DIN `din_target_field`에 list로 써도 여전히 attention 실패 | DIN.py `type(field) == tuple` strict 체크 (`model_zoo/DIN/src/DIN.py:105`) | YAML에 `!!python/tuple [a, b]` + `yaml.FullLoader` |
| 4 | `TypeError: got unexpected keyword argument 'eps'` | sklearn 1.4에서 `log_loss(eps=)` 제거 | `fuxictr/metrics.py` 에 `np.clip` 로 대체 |
| 5 | best params 적용시 `TypeError: can only concatenate list (not "str") to list` | `study.best_params`는 JSON 인코딩된 choice를 raw 문자열로 반환 | `U.decode_best_params(study.best_params)` 후 `params.update` |
| 6 | 04 분석의 `pd.notna(row[SEQ_FIELD])` → `ValueError: truth value of array is ambiguous` | test.parquet의 시퀀스 컬럼은 numpy 배열(0-padding)로 저장됨 | `seq_to_tokens(val, sep)` 헬퍼로 배열/문자열 둘 다 처리 |
| 7 | 01 재빌드 시 조용히 skip | FuxiCTR `build_dataset`이 `feature_map.json` 있으면 skip (logging.warn 한 줄) | `build_from_yaml(..., force_rebuild=True)` |

### 공통 교훈
- **모델 디버깅 시 `model_zoo/{MODEL}/src/{MODEL}.py`의 `__init__`과 `forward`를 먼저 읽어라.** 시그니처/field 처리 방식에 따라 YAML이 완전히 달라진다.
- **FuxiCTR 전처리는 모든 시퀀스에 기본 encoder를 붙인다.** DIN/DIEN/BST 계열은 반드시 `feature_specs`로 해제.
- **FuxiCTR는 cwd 의존.** `run_training`·`build_from_yaml` 안에서 `os.chdir(NOTEBOOK_ROOT)` 후 복원. 노트북에서 직접 FuxiCTR API 호출 시 주의.

---

## 2. 노트북별 가이드

### 01_Build_dataset.ipynb — 다양한 raw 포맷 처리

#### 현재 지원 흐름
1. `DATASET_REGISTRY` 에 데이터셋 엔트리(raw 경로, 포맷, 임베딩 파일, 분리 규칙).
2. 컬럼 미리보기 → FEATURE_COLS 편집 → valid split → YAML 생성 → `build_from_yaml`.

#### raw 포맷별 처리 포인트
- **CSV, 단일 파일, 시퀀스 `^` 구분** (AmazonElectronics_x1 유형)
  → `data_format: csv`, `splitter: "^"`, 바로 처리 가능.
- **CSV, 시퀀스가 공백/콤마 구분** (MicroVideo 유형)
  → raw를 사전 정규화(`^`로 치환)하거나, `FEATURE_COLS[i]["splitter"]`를 맞춰 지정.
- **Parquet raw** (TAAC2026 유형 등)
  → `data_format: parquet`. 이때 `_read_parquet_any` 로 읽힘.
  → 단, FuxiCTR가 parquet raw 입력도 기대하는 스키마가 CSV와 다를 수 있음 — 검증 필요.
- **valid가 없음**
  → `U.split_train_valid(train_path, out_dir, valid_ratio=0.1, group_col="user_id")`.
  → user 단위 분리가 디폴트(리크 방지). group_col 없으면 row 단위.
- **결측 시퀀스 다수** (KuaiVideo `pos_items`/`neg_items` 등)
  → FEATURE_COLS에 `"na_value": ""`, 빈 시퀀스 허용 옵션 명시.
- **Pretrained embedding 사용** (KuaiVideo `item_visual_emb` 등)
  → FEATURE_COLS의 해당 feature에 `pretrained_emb`, `embedding_dim`, `freeze_emb: True/False`.
  → `demo/example5_DeepFM_with_pretrained_emb_as_weights.py` 참고.
- **embedding 공유** (item_history ↔ item_id)
  → sequence 피처에 `"share_embedding": "item_id"`.

#### 주의사항
- **재빌드**: `feature_map.json`이 있으면 FuxiCTR는 조용히 skip. `build_from_yaml(..., force_rebuild=True)`.
- **경로**: YAML의 `train_data` 등은 **`notebook/` 기준 상대경로**(`../data/...`)여야 `run_training`·`build_from_yaml`에서 cwd 변경 후 해석 가능. `materialize_dataset_yaml`이 `_rel_to_data_root`로 자동 처리.
- **시퀀스 `max_len`**: 너무 짧으면 정보 손실, 너무 길면 메모리 폭발. 02에서 분포 확인 후 결정.
- **`min_categr_count`**: 기본 1. 너무 낮으면 vocab 폭발, 너무 높으면 long-tail 소실. 데이터셋별 튜닝.

---

### 02_EDA.ipynb — 전처리 후 데이터 형태

#### 핵심: FuxiCTR 전처리 후 데이터는 모두 정수 인코딩됨
`data/{dataset_id}/{train,valid,test}.parquet`의 구조:
- **categorical 컬럼**: `int64` vocab index. 0은 padding/UNK.
- **numeric 컬럼**: float (필요 시 normalize).
- **sequence 컬럼**: **numpy ndarray (dtype=int), 0-padding된 고정 길이 `max_len`**.
- **label 컬럼**: float (`binary_classification` 기준 0/1).

원본 문자열을 보려면:
- **raw CSV/parquet을 병행 로드** (02에서 샘플 분석용)
- 또는 `feature_processor.pkl` 로드 후 `feature_encoder.feature_map.features[name]['vocab']` 로 int→str 매핑

#### `feature_map.json` 구조 (분석에 유용)
```jsonc
{
  "dataset_id": "AmazonElectronics_x1",
  "num_fields": N,
  "total_features": M,
  "labels": ["label"],
  "features": {
    "item_id":      {"type": "categorical", "vocab_size": 63002, ...},
    "item_history": {"type": "sequence", "max_len": 50, "vocab_size": 63002,
                     "padding_idx": 0, "feature_encoder": null, ...},
    ...
  }
}
```
- `vocab_size` 포함 여부로 long-tail 진단 가능.
- `max_len`이 실제 데이터 분포와 맞는지 02에서 재검증.

#### 시각화 아이디어 (이미 있는 것 + 확장)
- (기본) 레코드 수, 라벨 분포, feature별 missing.
- (기본) 카테고리 top-20 빈도 + 카테고리별 CTR.
- (기본) 시퀀스 길이 히스토그램 + p50/p90/p99.
- (기본) 시퀀스 내 bigram top-K, 히트맵.
- **확장 제안**:
  - 시퀀스 길이 × 타겟 CTR (길이 bin별 conversion rate).
  - 유저 활동량 분포(power law 여부).
  - time decay: timestamp 있는 데이터셋이면 week/day별 CTR 추이.
  - item cold-start 분석: train에만 있는 item이 test에서 등장할 때 CTR 변화.
  - 시퀀스 중복도(unique/length) 분포.
  - category-item 교차 분포 (`cate_id` 내 `item_id` 다양성).

#### 시각화 시 유의점
- 시퀀스 컬럼은 배열 → `seq_to_tokens`로 flatten한 후 `Counter` 등으로 집계.
- 카테고리 vocab_size가 매우 크면 seaborn heatmap은 렌더링 불가. top-K로 자름.
- FuxiCTR 전처리 결과로 string 원본이 사라진 경우, raw CSV를 별도 로드해 조인하면 해석 쉬움.

---

### 03_Tuning_Run.ipynb — 새 모델 추가 시 체크리스트

**반드시 이 순서대로 한다.**

#### 1) 모델 소스 읽기
`model_zoo/{MODEL}/src/{MODEL}.py` 를 열어 다음을 파악:
- `__init__` 시그니처의 필수 파라미터 (예: DIN의 `din_target_field`, `din_sequence_field`).
- 파라미터의 **기대 타입**: list? tuple? list-of-tuple?
  → `type(x) == tuple` 같은 strict 체크 있으면 YAML에 `!!python/tuple` 필요.
- `forward`에서 시퀀스를 raw `[B, L, D]`로 받는지, pre-pool된 `[B, D]`로 받는지.
  → raw를 받으면 **반드시 `feature_specs`로 기본 encoder 해제**.
- 어떤 `feature_map` 메서드를 호출하는지 (예: `sum_emb_out_dim`).

#### 2) `configs/models/{MODEL}_base.yaml` 작성
- `Base:` 섹션: `num_workers`, `early_stop_patience`, `pickle_feature_encoder`, `save_best_only`, `feature_specs`, `feature_config`.
  - **시퀀스 attention 모델은 `feature_specs`로 해당 시퀀스 필드의 `feature_encoder: null` 지정**.
- `{MODEL}:` 섹션: 모델별 하이퍼 + `task`, `loss`, `metrics`, `optimizer`, `monitor`, `monitor_mode`, `seed`, `gpu`.
- tuple 필요 시 `!!python/tuple [a, b]`. (`_load_yaml`이 FullLoader 사용)
- `dnn_hidden_units` 같은 list 값도 YAML에서 그대로 OK.

#### 3) `configs/tuning/{MODEL}_search.yaml` 작성
- `type: loguniform|uniform|categorical|int`.
- list/dict choice 가능 (예: `dnn_hidden_units` choices) → `suggest_from_space`가 JSON 인코딩해 저장.
- 탐색 대상이 아닌 파라미터는 `base.yaml` 값이 유지됨.

#### 4) 스위치 설정 & smoke test
- `N_TRIALS=3`, `N_REPEATS=1`, `TUNE_EPOCHS=3`로 먼저 돌려 에러 잡고, 본 탐색으로.
- smoke에서 `tune_t000_s<seed>` 체크포인트/metrics가 생기면 OK.

#### 5) 모델별 특이사항 (앞으로 쌓아갈 섹션)
- **DIN 계열 (DIEN, DMR, BST, DMIN 등)**: 시퀀스 attention → `feature_specs` 필수. target/sequence 필드가 tuple이면 `!!python/tuple`.
- **LongCTR/DIN 변형**: `din_target_field`/`din_sequence_field` 없이 처리하는 변형도 존재 → 해당 모델 소스 별도 분석.
- **DeepFM, DCN, DCNv2, AutoInt**: 시퀀스 attention 없음, 기본 pooling으로 OK. `feature_specs` 불필요.
- **FinalMLP, MaskNet**: `Base` 하이퍼가 많음, demo config 참조.

#### 6) Optuna 관련 함정
- **resume**: 동일 `study_name` + `storage=sqlite:///...` 면 자동 resume.
- **list/dict choice**: `suggest_from_space` 자동 JSON 인코딩, `decode_best_params`로 디코딩.
- **failed trial**: `optuna.TrialPruned()` raise → best 계산에서 제외됨.
- **seed 다양화**: 튜닝 중에는 `base.yaml`의 seed 고정, 최종 N_REPEATS에서만 seed 바꿈.

#### 7) checkpoint/예측 파일
- `artifacts/runs/{dataset_id}/{MODEL}_{run_id}.model` — FuxiCTR best ckpt.
- `{MODEL}_{run_id}.metrics.json` / `.params.json` — `run_training` sidecar.
- 예측은 `artifacts/predictions/{dataset_id}/{MODEL}/{run_id}/preds.parquet`.
  컬럼: `y_true`, `y_pred`, + `_default_sample_cols` 가 고른 피처(최대 8개).

---

### 04_Results_analysis.ipynb — 데이터 형태와 분석 확장

#### 예측 parquet 스키마
`preds.parquet`:
- `y_true` (float, 0/1), `y_pred` (float).
- 샘플 feature 컬럼들 (`_default_sample_cols` 선정): categorical은 int, sequence는 **numpy 배열(int, 0-padding)**, numeric은 float.
  - 즉 원본 string이 아닌 전처리 후 int ID.
  - 원본 string이 필요하면 `feature_processor.pkl` 로드해 vocab 역매핑.

#### 시퀀스 처리 헬퍼 (이미 노트북 안에 정의됨)
```python
def seq_to_tokens(val, sep='^'):
    # numpy array (0 padding) / list / string / NaN 모두 처리
    # 배열이면 0 제외하고 str(x) 리스트 반환
    # 문자열이면 sep로 split
```
새 분석 추가 시 **시퀀스 컬럼에는 반드시 이 헬퍼를 경유**. 직접 `pd.notna(seq_val)`·`seq_val.split()` 금지 (배열에선 터짐).

#### 현재 구현된 분석
- subgroup AUC (시퀀스 길이 / 아이템 인기도 / 유저 활동량).
- 오분류(FN/FP) 피처 분포, 시퀀스 길이 분포.
- 시퀀스 패턴: target_in_history, seq_diversity, last_is_target.
- 모델 간 예측 상관(pearson, heatmap).

#### 확장 아이디어
- **calibration**: bin별 예측 확률 vs 실제 CTR plot.
- **per-user AUC 분포**: `groupby('user_id').apply(roc_auc)` histogram.
- **lift curve / PR curve**.
- **model disagreement 분석**: 두 모델 예측 차이가 큰 샘플의 특징.
- **시퀀스 recency**: 뒤쪽 k개만 남기고 AUC 변화 (cut-off별).
- **category mismatch**: target `cate_id` ∉ history cate 집합인 케이스의 CTR/AUC.
- **gAUC / avgAUC**: 이미 `fuxictr.metrics`에 구현되어 있음 → `group_id` 셋업 후 사용.

#### 저장 규약
복잡한 분석 결과는 `artifacts/analysis/{dataset_id}/` 아래 parquet로 저장 → 06 에서 재사용.

#### 시각화 주의점
- 한글 폰트: 서버 matplotlib에 한글 폰트 없으면 박스로 표시됨. Latin 축 라벨 권장.
- 모델이 많으면 line plot legend 범람 — `model` 단위로 그룹핑 후 반복 실험은 mean+band로.
- sequence length 분포는 log scale이 유리 (heavy tail).

---

### 05_New_model_test.ipynb — 새 모델 설계 가이드

#### BaseModel 상속 규약
```python
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block

class MyModel(BaseModel):
    def __init__(self, feature_map, model_id='MyModel', gpu=-1,
                 learning_rate=1e-3, embedding_dim=32, ...,
                 embedding_regularizer=None, net_regularizer=None,
                 **kwargs):
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         embedding_regularizer=embedding_regularizer,
                         net_regularizer=net_regularizer, **kwargs)
        # layers...
        self.compile(kwargs['optimizer'], kwargs['loss'], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # ... custom logic ...
        return {'y_pred': y_pred}
```

#### params dict 필수 키
`run_training`은 `full = params + dataset_runtime`을 모델 `__init__`에 `**`로 넘김.
BaseModel이 요구하는 키(둘 다 필수 — 빠지면 KeyError):
- `task`, `loss`, `metrics`, `optimizer`, `monitor`, `monitor_mode`,
  `save_best_only`, `early_stop_patience`, `embedding_regularizer`, `net_regularizer`,
  `verbose`, `seed`, `gpu`, `batch_size`, `epochs`, `shuffle`, `num_workers`,
  `eval_steps`, `debug_mode`, `group_id`, `use_features`, `feature_specs`,
  `feature_config`, `pickle_feature_encoder`, `model_root`, `model_id`.
- 기본값은 `nb_utils._default_fuxictr_params()` 참조.
- 05 노트북 템플릿은 이걸 전부 포함. 복붙 후 모델 특수 파라미터만 바꿔라.

#### 시퀀스 사용 모델이라면
- 새 모델이 raw sequence `[B, L, D]`를 직접 처리하면 **반드시 `params['feature_specs']`에 시퀀스 필드의 `feature_encoder: null`**. (03에서 YAML로 하던 것을 05에서는 dict로.)
  ```python
  params['feature_specs'] = [
      {'name': 'item_history', 'feature_encoder': None},
      {'name': 'cate_history', 'feature_encoder': None},
  ]
  ```
- target/sequence를 튜플로 받는 설계면 Python tuple 리터럴 그대로 `params['my_field'] = [('item_id', 'cate_id')]`.

#### 흔한 초기화 에러
- `self.compile(...)` 빼먹음 → optimizer 없음 → `fit` 중 AttributeError.
- `self.model_to_device()` 빼먹음 → CPU에서만 돌아 GPU 무용.
- `self.reset_parameters()` 빼먹음 → 초기화 불균일로 학습 불안정.
- `**kwargs` 를 super()에 안 넘김 → BaseModel이 `kwargs["verbose"]` 등에서 KeyError.

#### Optuna 탐색
05 노트북 하단에 주석으로 템플릿 있음. `run_training(model_cls=MyModel)` 에 `model_cls` 넘기면 됨. 03과 달리 `_import_model_class`를 건너뛴다.

#### 권장 검증 플로우
1. `epochs=1`, `N_REPEATS=1` smoke.
2. loss가 감소하는지 + valid AUC > 0.5 확인.
3. predictions parquet 생성 확인.
4. 04/06에서 다른 모델과 함께 자연스럽게 비교되는지 확인.

---

### 06_Results_viz.ipynb — 메트릭 시각화 확장

#### 입력 데이터
- `U.list_runs(dataset_id=None)` → 모든 run 메트릭 DataFrame.
  컬럼: `dataset`, `model`, `run_id`, `model_id`, `path`, `metric.*`, `param.*`.
  - `metric.*` 에는 scalar만 포함 (list/dict는 제외).
  - 튜닝 trial(`run_id`에 `tune_t` 포함)도 섞임 → 최종 run만 보려면 필터.
- Optuna study: `artifacts/tuning/{ds}/{model}/study.db`.
  `optuna.load_study(study_name=..., storage=f'sqlite:///{path}')` 로 로드.

#### 현재 구현된 뷰
- 데이터셋×모델 AUC 히트맵 (mean±std).
- 반복실험 boxplot + 개별 점.
- Optuna progress curve (best-so-far).
- `optuna.importance.get_param_importances` (fanova — completed trial ≥ 3 필요).
- 데이터셋별 best 모델 요약 표.

#### 확장 아이디어
- **Pareto front**: AUC vs train_seconds (또는 params count) scatter.
- **hyperparameter parallel coordinate**: `optuna.visualization.plot_parallel_coordinate` (plotly 필요).
- **seed별 분산**: violin plot by (dataset, model), y=AUC.
- **튜닝 vs 최종 간 gap**: 튜닝 best valid AUC vs 최종 test AUC.
- **시간 추이**: run 타임스탬프 기준 성능 개선 곡선 (실험 로그 관점).
- **모델 × 데이터셋 상대 순위 heatmap**: 절대값 대신 rank.

#### 주의점
- `metric.test_AUC` 가 없으면 `metric.valid_AUC`로 폴백 (현재 구현).
- `optuna.visualization` 은 plotly 기반 — matplotlib-only 환경이면 `matplotlib_importance` 계열 함수 필요.
- 반복 실험 시 seed가 1개만 있으면 std=NaN → `fillna(0)` 로 표시.

---

## 3. 암묵적 규칙 / 자주 간과되는 것들

사용자가 명시하지 않았으나 알고 있어야 할 것:

- **cwd 민감성**: FuxiCTR는 `load_dataset_config`·`set_logger`·데이터 경로 해석 모두 cwd 기반.
  → `run_training`과 `build_from_yaml`이 `os.chdir(NOTEBOOK_ROOT)` ↔ `prev_cwd` 로 감쌈.
  → **노트북에서 FuxiCTR API를 직접 호출**하면 경로가 깨질 수 있음. 가급적 `nb_utils` 경유.
- **`make_run_id` 포맷**: `{timestamp}_{MODEL}_{DATASET_ID}_s{seed}`.
  → tuning trial은 별도 포맷 (`tune_t{NNN}_s{seed}`) — list_runs 필터링에 사용.
- **checkpoint 파일명**: `{model_id}.model` (FuxiCTR 규약). `model_id = f"{MODEL}_{run_id}"`.
- **`save_best_only=False` 동안 IO 폭증**: 튜닝 trial에서는 False 권장(속도↑), 최종 run에서 True.
- **`feature_processor.pkl` 로 원본 string 복원**:
  ```python
  import pickle
  with open('data/{ds}/feature_processor.pkl', 'rb') as f:
      fp = pickle.load(f)
  vocab = fp.feature_map.features['item_id']['vocab']  # {str: int}
  inv   = {v: k for k, v in vocab.items()}             # {int: str}
  ```
- **feature_map.json vs feature_processor.pkl**: 전자는 스키마, 후자는 vocab/scaler 등 상태 포함. 분석엔 pkl이 풍부.
- **예측 parquet 저장 컬럼 상한**: `_default_sample_cols` 기본 8개. 더 필요하면 `run_training(sample_feature_cols=[...])`로 명시.
- **test_gen 재생성**: `run_training` 내부에서 evaluate용/predict용으로 **RankDataLoader 2번 생성**. iterator 1회 소비 후 재사용 불가 때문.
- **seed 적용 시점**: `seed_everything`은 `run_training` 진입 직후 실행. 모델 `__init__` 이전이어야 동일 seed 보장.
- **Windows 경로**: 로컬은 Windows, 서버는 Linux. YAML에 저장되는 경로는 forward slash(`/`). `_rel_to_data_root`가 이미 처리.
- **`list_runs`에서 list/dict param 제외**: scalar만 컬럼으로 평탄화. `dnn_hidden_units` 같은 건 `.params.json` 직접 열어서 봐야 함.
- **Optuna resume**: 같은 storage URL + study_name 이면 자동 append. 다른 search space로 재개하면 히스토리 섞임 — 새 study_name 쓰기.

---

## 4. 다음 세션 진입 프로토콜

새 세션에서 위 6개 카테고리 중 하나 작업할 때:

1. 이 파일(`notebook/NOTES.md`) 전체를 읽는다.
2. 해당 노트북을 연다 (`03_Tuning_Run.ipynb` 등).
3. `nb_utils.py` 의 관련 함수 시그니처를 확인한다 (`run_training`, `load_base_config`, `suggest_from_space`, `decode_best_params`, `list_runs`, `load_predictions` 등).
4. 필요하다면 `model_zoo/{MODEL}/src/{MODEL}.py` 를 먼저 읽는다 (3번 카테고리).
5. 기존 YAML들(`configs/models/DIN_base.yaml`, `configs/tuning/DIN_search.yaml`)을 템플릿 삼는다.
6. 변경은 YAML/노트북 셀에 한정. `nb_utils.py` 에 모델·데이터셋 특이 로직을 **절대** 추가하지 않는다.

---

## 5. 열린 항목 / 미확정

- **TAAC2026**: 전처리 파이프라인 완성 (`00_preprocessing.ipynb` + `01_embedding.ipynb`). 01의 `DATASET_REGISTRY` 및 `FEATURE_TEMPLATES` 등록 완료. 상세 사항은 `notebook/NOTES_TAAC.md` 참조. full 데이터 적용 시 k값, max_len 재조정 필요.
- **KuaiVideo_x1, MicroVideo1.7M_x1**: 01의 FEATURE 템플릿 초기 작성만 되어 있음. 실제 학습 검증은 AmazonElectronics_x1에서만 완료. 다른 데이터셋 추가 시 위 "raw 포맷별 처리 포인트" 참조.
- **추가 모델 base config**: 현재 DIN, DeepFM만. DCN, DCNv2, AutoInt, FinalMLP 등은 요청 시 추가.
- **04/06의 분석 저장 포맷**: `artifacts/analysis/`에 저장 규약을 더 엄격히 정할지 (04 → 06 재사용을 위한 스키마)는 미정.
