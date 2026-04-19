# TAAC2026 데이터셋 — 전처리 & 임베딩 개발 노트

TAAC2026 competition 데이터를 FuxiCTR 파이프라인에 맞추기 위한 전처리 과정을 정리한 문서.
`notebook/NOTES.md`의 범용 가이드와 병행하여 TAAC2026 고유 사항을 기록한다.

---

## 0. 파일 구조

```
data/raw_data/TAAC2026/
  raw/
    demo_1000.parquet          # 원본 데이터 (1000행 x 120열)
    README.md                  # 공식 스키마 설명
  00_preprocessing.ipynb       # raw parquet → train.csv / test.csv
  01_embedding.ipynb           # unpaired dense → KMeans pseudo-category + h5
  train.csv                    # 전처리 결과 (700행 x 103열, fid_61/fid_87 포함)
  test.csv                     # 전처리 결과 (300행 x 103열)
  fid_61_k{K}.h5               # user_dense_feats_61 centroid embedding (256d)
  fid_87_k{K}.h5               # user_dense_feats_87 centroid embedding (320d)
  preprocessing.ipynb          # 사용자 자체 실험 노트북 (별도 접근법)
```

---

## 1. 데이터 개요

| 항목 | 값 |
|------|-----|
| 원본 파일 | `demo_1000.parquet` (1000행 x 120열, ~39MB) |
| Label | `label_type`: 1 → 0 (negative, 876개), 2 → 1 (positive, 124개) |
| Positive rate | ~12.4% |
| Train/Test split | 7:3 stratified, random_state=42 |
| 최종 CSV 컬럼 수 | 103 (101 from preprocessing + 2 pseudo-category from embedding) |

---

## 2. 컬럼 분류 (120열 → 6 카테고리)

### 2.1 CSV에 포함 (00_preprocessing.ipynb)

| 카테고리 | 컬럼 수 | 처리 | FuxiCTR type |
|----------|---------|------|-------------|
| label (label_type→label) | 1 | 1→0, 2→1 | label |
| user_id, item_id | 2 | 그대로 유지 | categorical |
| timestamp | 1 | 그대로 유지 | numeric |
| User Int scalar | 35 | 그대로 유지 | categorical |
| Item Int scalar | 13 | 그대로 유지 | categorical |
| Unpaired User Int list (fid 15,60,80) | 3 | ^-join | sequence (max_len=15) |
| Item Int list (fid 11) | 1 | ^-join | sequence (max_len=20) |
| Domain Sequence (4 domains, 45 cols) | 45 | 0 제거 → ^-join | sequence (max_len=50) |
| Pseudo-category (fid_61, fid_87) | 2 | 01_embedding에서 추가 | categorical + pretrained_emb |

### 2.2 h5 임베딩으로 분리 (01_embedding.ipynb)

| 컬럼 | 타입 | 길이 | 처리 |
|------|------|------|------|
| user_dense_feats_61 | float | 256 고정 | KMeans → fid_61 pseudo-category + centroid h5 |
| user_dense_feats_87 | float | 320 고정 | KMeans → fid_87 pseudo-category + centroid h5 |

### 2.3 제외

| 컬럼 | 이유 |
|------|------|
| label_time | timestamp과 중복 |
| Paired fid (62-66, 89-91) int+dense | 가변 길이 category-value pair, 직접 h5 변환 불가. 탐색적 분석만 수행 |

---

## 3. Paired fid vs Unpaired 구분

### 핵심 개념
`user_int_feats_{fid}`와 `user_dense_feats_{fid}`가 **동일한 fid를 공유**하면 paired.
Paired features는 동일한 null 패턴, 동일한 행별 길이를 가진다 (entity 단위 aligned).

- **Paired fid**: 62, 63, 64, 65, 66, 89, 90, 91
  - int = category ID, dense = associated value (count/amount 또는 normalized)
  - fid 62-66: dense 값이 count/amount (mean ~13만, 큰 범위)
  - fid 89-91: dense 값이 정규화됨 (mean ≈ 0, std ≈ 0.32, vocab 5~7)
- **Unpaired int list**: 15, 60, 80 (dense 대응 없음)
- **Unpaired dense**: 61, 87 (int 대응 없음, true embedding 벡터)

### Paired fid 처리 결정
- **방법 A (Zero-pad + Concat)**: max_len으로 pad, int와 dense를 concat하여 고정 길이 벡터 생성
- 01_embedding.ipynb에서 방법 A, B(Weighted sum), C(Dense-only agg)를 비교 → **방법 A 채택**
- 현재는 탐색적 분석만 수행. full 데이터에서 본격 적용 시 추가 노트북 필요

---

## 4. Unpaired Dense → Pseudo-category 방식

### FuxiCTR pretrained embedding 메커니즘
- h5 파일: `key` (categorical values) + `value` (embedding matrix)
- `pretrained_embedding.py`의 `load_pretrained_embedding`이 h5를 로드하여 `nn.Embedding` 초기 가중치로 설정
- **제약**: 1개의 feature column에 1개의 h5만 연결 가능

### 설계: clustering pseudo-category
1. Unpaired dense 벡터 (user_dense_feats_61: 256d, user_dense_feats_87: 320d)를 KMeans 클러스터링
2. 최적 k를 silhouette score로 자동 탐색 (후보: k=4,6,8,...,32)
3. 각 user에게 cluster label 할당 → CSV에 `fid_61`, `fid_87` pseudo-category 컬럼 추가
4. Centroid 벡터를 h5에 저장: key=cluster_label(0..k-1), value=centroid

### FuxiCTR 연결
```python
# 01_Build_dataset.ipynb DATASET_REGISTRY
'pretrained_emb': {
    'fid_61': {'path': '...TAAC2026/fid_61_k{K}.h5', 'dim': 256},
    'fid_87': {'path': '...TAAC2026/fid_87_k{K}.h5', 'dim': 320},
},
# FEATURE_TEMPLATES
{'name': 'fid_61', 'active': True, 'dtype': 'str', 'type': 'categorical'}
```

### Embedding 특성 (demo 1000 샘플 기준)
- user_dense_feats_61 (256d): L2 norm ≈ 1.0, L2-normalized embedding
- user_dense_feats_87 (320d): mean ≈ 0, std ≈ 0.11, pretrained embedding

---

## 5. 00_preprocessing.ipynb 구조

| Cell | 내용 |
|------|------|
| 1 | 데이터 로드 + 6카테고리 자동 분류 (paired fid 감지 포함) |
| 2 | EDA: Label 분포 bar chart, ID nunique, timestamp 히스토그램 |
| 3 | EDA: Scalar features (null count, nunique 요약) |
| 4 | EDA: List features 길이 분포, paired fid 정렬 검증 |
| 5 | EDA: Domain sequence 실제 길이 (0-pad 제거 후), 도메인별 boxplot |
| 6 | 전처리: `list_to_str()`, ^-join, paired/dense/label_time 제거 |
| 7 | Label 변환 (label_type 2→1, else→0), 컬럼 순서 정리 |
| 8 | train_test_split(7:3, stratified, random_state=42) |
| 9 | CSV 저장 + reload 검증 |

---

## 6. 01_embedding.ipynb 구조

| Cell | 내용 |
|------|------|
| 1 (md) | 노트북 개요 + 처리 대상 정리 |
| 2 | 데이터 로드 + UNPAIRED_DENSE / PAIRED_FIDS 정의 + 기본 통계 |
| 3 (md) | Clustering → pseudo-category + h5 흐름 설명 |
| 4 | KMeans 최적 k 탐색 (silhouette), h5 저장, silhouette curve plot |
| 5 (md) | Evaluation 섹션 설명 |
| 6 | PCA 시각화 (cluster/label 색상) + Logistic Regression AUC 비교 |
| 7 (md) | Paired fid 분석 섹션 설명 |
| 8 | Helper 함수 정의 (paired_to_padded, paired_to_dense_agg) |
| 9 | Paired fid 텍스트 통계 + PCA 시각화 + dense value 히스토그램 |
| 10 (md) | Embedding method 후보 비교표 |
| 11 | Method A vs C AUC 비교 (fid=62 대표) → **방법 A 채택** |
| 12 (md) | CSV update 설명 |
| 13 | fid_61/fid_87 pseudo-category 컬럼을 train.csv/test.csv에 merge |
| 14 (md) | H5 검증 + Build_dataset reference 설명 |
| 15 | h5 파일 검증, CSV pseudo-category 확인, DATASET_REGISTRY 참조 출력 |

---

## 7. 실행 순서

```
1. 00_preprocessing.ipynb  → train.csv, test.csv (101 cols)
2. 01_embedding.ipynb      → fid_61/fid_87 columns added to CSV (103 cols) + h5 files
3. 01_Build_dataset.ipynb  → DATASET='taac2026' → FuxiCTR parquet + feature_map.json
```

### 주의사항
- 01_embedding.ipynb는 00_preprocessing.ipynb의 출력(train.csv, test.csv)에 의존
- h5 파일명에 최적 k가 포함됨 (`fid_61_k{K}.h5`). 01_Build_dataset.ipynb의 DATASET_REGISTRY에서 h5 경로의 k값을 실제 결과에 맞춰 수정 필요
- 두 노트북 모두 working directory가 `data/raw_data/TAAC2026/`이어야 함 (상대 경로 `raw/demo_1000.parquet` 사용)

---

## 8. 01_Build_dataset 연결

### DATASET_REGISTRY
```python
'taac2026': {
    'dataset_id': 'TAAC2026',
    'raw_dir':    U.RAW_DATA_ROOT / 'TAAC2026',
    'train_file': 'train.csv',
    'test_file':  'test.csv',
    'valid_file': None,       # train에서 10% 분리
    'data_format': 'csv',
    'label_name': 'label',
    'group_col':  'user_id',
    'pretrained_emb': {
        'fid_61': {'path': ...'TAAC2026/fid_61_k{K}.h5', 'dim': 256},
        'fid_87': {'path': ...'TAAC2026/fid_87_k{K}.h5', 'dim': 320},
    },
}
```

### FEATURE_TEMPLATES 요약 (103 features)
- categorical: user_id, item_id, 35 user_int scalar, 13 item_int scalar, fid_61, fid_87 = **53개**
- numeric: timestamp = **1개**
- sequence: 3 unpaired user_int list + 1 item_int list + 45 domain seq = **49개**

---

## 9. 열린 항목 / 미확정

- **Full 데이터**: 현재 demo 1000 샘플로만 검증. full 데이터에서 k값, max_len, min_categr_count 재조정 필요
- **Paired fid 활용**: 방법 A(pad+concat)로 결정했으나 아직 FuxiCTR 파이프라인에 통합하지 않음. 별도 처리 노트북 또는 커스텀 feature encoder 필요
- **Domain sequence max_len**: demo에서 50으로 설정. full 데이터 p90/p99 기준으로 조정 필요
- **Sequence share_embedding**: 같은 도메인 내 시퀀스 간 embedding 공유 여부 미정
- **Pretrained embedding freeze**: 현재 freeze=True 기본. fine-tune 여부는 모델 실험 후 결정
- **timestamp 처리**: 현재 numeric으로 등록. bucketing, cyclical encoding 등 추가 처리 가능
