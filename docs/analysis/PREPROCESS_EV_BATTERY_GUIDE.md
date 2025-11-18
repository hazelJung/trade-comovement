# EV 배터리 공급망 데이터 전처리 파이프라인 가이드

이 문서는 `preprocess_ev_battery.ipynb` 노트북의 전체 전처리 과정을 상세히 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [전처리 단계별 상세 설명](#전처리-단계별-상세-설명)
3. [최종 결과물](#최종-결과물)
4. [사용 방법](#사용-방법)

---

## 개요

### 목적
무역 데이터를 **EV 배터리 공급망 도메인** 관점에서 전처리하여, 공행성 쌍 예측 모델에 사용할 수 있는 고품질 피처를 생성합니다.

### 입력 데이터
- **파일**: `data/train.csv`
- **형식**: 무역 거래 데이터 (item_id, year, month, hs4, weight, quantity, value 등)
- **기간**: 2022년 1월 ~ 2025년 7월

### 출력 데이터
- **파일**: `analysis/data/monthly_features.csv`
- **형식**: 월별 집계 데이터 (27개 컬럼)
- **행 수**: 3,776행 (100개 item_id × 평균 37.76개 월)

---

## 전처리 단계별 상세 설명

### 1️⃣ 기본 로드 & 보조 컬럼 생성

#### 목적
원본 데이터를 로드하고, 시계열 분석을 위한 datetime 컬럼을 생성합니다.

#### 처리 과정
```python
# 1. train.csv 로드
df_train = pd.read_csv('../data/train.csv')

# 2. year, month를 이용해 datetime 컬럼 생성
df_train['ym'] = pd.to_datetime(
    df_train['year'].astype(str) + "-" + 
    df_train['month'].astype(str).str.zfill(2) + "-01"
)
```

#### 결과
- **입력**: 10,836행 (원본 거래 데이터)
- **출력 컬럼**: `ym` (datetime 형식, 예: 2022-01-01)
- **용도**: 시계열 분석 및 월별 집계의 기준점

---

### 2️⃣ item_id-hs4 매핑 확보

#### 목적
각 `item_id`에 대응하는 고유한 `hs4` 코드를 매핑 테이블로 생성합니다.

#### 가정
- **1:1 관계**: 하나의 `item_id`는 항상 동일한 `hs4` 코드를 가집니다.
- **역관계**: 하나의 `hs4` 코드는 여러 `item_id`를 가질 수 있습니다.

#### 처리 과정
```python
# (item_id, hs4) 유일 조합으로 매핑 테이블 생성
item_hs4_map = (
    df_train[['item_id', 'hs4']]
    .drop_duplicates()
    .set_index('item_id')['hs4']
    .to_dict()
)
```

#### 결과
- **매핑 테이블**: 100개 item_id → hs4 코드
- **예시**: `{'DEWLVASR': 3038, 'ELQGMQWE': 2002, ...}`
- **용도**: 이후 단계에서 hs4 정보를 빠르게 조회하기 위한 lookup table

---

### 3️⃣ (item_id, year, month) 단위 월별 집계

#### 목적
같은 `item_id`의 같은 월에 발생한 모든 거래를 하나의 행으로 집계합니다.

#### 처리 과정
```python
# value, weight, quantity를 합산하여 월별 집계
monthly_features = (
    df_train.groupby(['item_id', 'year', 'month'], as_index=False)
    .agg({
        'value': 'sum',      # 무역금액 합계
        'weight': 'sum',     # 중량 합계
        'quantity': 'sum',   # 수량 합계
        'ym': 'first'        # datetime 정보
    })
    .rename(columns={
        'value': 'value_sum',
        'weight': 'weight_sum',
        'quantity': 'quantity_sum'
    })
)

# item_id-hs4 매핑 적용
monthly_features['hs4'] = monthly_features['item_id'].map(item_hs4_map)
```

#### 결과
- **입력**: 10,836행 (원본 거래 데이터)
- **출력**: 3,776행 (월별 집계 데이터)
- **집계율**: 약 35% (같은 item_id의 같은 월에 평균 2.87건의 거래)
- **새 컬럼**: `value_sum`, `weight_sum`, `quantity_sum`, `hs4`

#### 왜 필요한가?
- **데이터 축소**: 거래 레벨 → 월별 레벨로 축소하여 시계열 분석 용이
- **노이즈 감소**: 같은 월의 여러 거래를 합산하여 안정적인 패턴 추출
- **모델 입력**: 공행성 쌍 예측 모델은 월별 데이터를 사용

---

### 4️⃣ EV 배터리 공급망 도메인 클러스터 라벨링

#### 목적
HS4 코드를 EV 배터리 공급망 관점에서 3개의 클러스터로 분류합니다.

#### 클러스터 정의

| 클러스터 | HS2 코드 | 의미 | 예시 |
|---------|---------|------|------|
| **battery_materials** | 27, 28, 29, 31, 38, 39, 72, 79, 81 | 배터리 소재·화학·금속 | 화학제품, 금속제품 |
| **ev_machinery_electrical** | 84, 85, 87, 90 | EV·설비·전기기기 | 전기모터, 기계류 |
| **other_industries** | 나머지 | 기타 산업 | 기타 제품 |

#### 처리 과정
```python
# HS2 코드 추출 (HS4의 앞 2자리)
hs4_str = monthly_features['hs4'].astype(str).str.zfill(4)
monthly_features['hs2'] = hs4_str.str[:2]

# 클러스터 라벨링 함수
def assign_cluster(hs2):
    if hs2 in ["27", "28", "29", "31", "38", "39", "72", "79", "81"]:
        return "battery_materials"
    elif hs2 in ["84", "85", "87", "90"]:
        return "ev_machinery_electrical"
    else:
        return "other_industries"

monthly_features['cluster'] = monthly_features['hs2'].apply(assign_cluster)
```

#### 결과
- **클러스터 분포**:
  - `battery_materials`: 1,818행 (48.1%)
  - `other_industries`: 1,364행 (36.1%)
  - `ev_machinery_electrical`: 594행 (15.7%)
- **새 컬럼**: `hs2`, `cluster`

#### 왜 필요한가?
- **도메인 지식 활용**: EV 배터리 공급망의 실제 구조를 반영
- **공행성 쌍 탐색**: 같은 클러스터 내 품목 간 공행성이 높을 가능성
- **가중치 적용**: 클러스터 정보를 활용하여 상관계수에 가중치 부여 가능

---

### 5️⃣ quantity 처리 (구조적 결측 반영)

#### 목적
`quantity` 값이 의미 있는지 판단하고, 구조적 결측을 플래그로 표시합니다.

#### 문제 상황
- 일부 HS4 코드는 **항상** `quantity_sum = 0`입니다.
- 이는 "수량 정보가 없는 HS4"를 의미합니다 (구조적 결측).
- 반면, 일부 HS4는 때때로 `quantity_sum > 0`인 경우가 있습니다.

#### 처리 과정
```python
# 1. hs4별로 quantity_sum > 0인 row가 하나라도 있는지 확인
hs4_has_quantity = (
    monthly_features.groupby('hs4')['quantity_sum']
    .apply(lambda x: (x > 0).any())
    .to_dict()
)

# 2. 해당 hs4에서 quantity 정보가 의미 있는지 플래그
monthly_features['has_quantity_info_hs4'] = (
    monthly_features['hs4'].map(hs4_has_quantity).astype(int)
)

# 3. 현재 row의 quantity_sum > 0 여부
monthly_features['has_quantity_this_row'] = (
    (monthly_features['quantity_sum'] > 0).astype(int)
)
```

#### 결과
- **수량 정보 의미 있는 HS4**: 746행 (19.8%)
- **수량 존재하는 행**: 703행 (18.6%)
- **새 컬럼**: `has_quantity_info_hs4`, `has_quantity_this_row`

#### 왜 필요한가?
- **모델 학습**: `quantity`가 의미 없는 경우와 있는 경우를 구분하여 학습
- **피처 선택**: `has_quantity_info_hs4 = 0`인 경우 `quantity` 피처를 제외할 수 있음
- **데이터 품질**: 구조적 결측을 명시적으로 표시하여 모델이 올바르게 처리

---

### 6️⃣ 0값 기반 "사실상 결측" 플래그

#### 목적
0값을 "사실상 결측"으로 간주하고, 각 상황에 맞는 플래그를 생성합니다.

#### 문제 상황
- **명시적 결측 (NaN)**: 데이터에 없음
- **사실상 결측 (0)**: 데이터는 있지만 실제로는 정보가 없음
  - 예: `weight = 0`이지만 `value > 0` → 무게 정보 미기입 가능성

#### 처리 과정
```python
# 1. 기본 0값 플래그
monthly_features['weight_zero_flag'] = (
    (monthly_features['weight_sum'] == 0).astype(int)
)
monthly_features['value_zero_flag'] = (
    (monthly_features['value_sum'] == 0).astype(int)
)

# 2. 사실상 결측 플래그
# weight가 0인데 value는 있음 → 무게 정보 미기입 가능성
monthly_features['weight_missing_flag'] = (
    (monthly_features['weight_sum'] == 0) & 
    (monthly_features['value_sum'] > 0)
).astype(int)

# value가 0인데 weight는 있음 → 금액 정보 미기입 가능성
monthly_features['value_missing_flag'] = (
    (monthly_features['value_sum'] == 0) & 
    (monthly_features['weight_sum'] > 0)
).astype(int)
```

#### 결과
- **weight_zero_flag**: 91행 (2.4%)
- **value_zero_flag**: 0행 (0%)
- **weight_missing_flag**: 91행 (2.4%)
- **value_missing_flag**: 0행 (0%)
- **새 컬럼**: `weight_zero_flag`, `value_zero_flag`, `weight_missing_flag`, `value_missing_flag`

#### 왜 필요한가?
- **데이터 품질 관리**: 0값의 의미를 명확히 구분
- **보정 전략**: `weight_missing_flag = 1`인 경우 다음 단계에서 보정 수행
- **모델 학습**: 플래그를 피처로 사용하여 모델이 결측 패턴을 학습

---

### 7️⃣ weight_sum / value_sum 결측성 보정

#### 목적
"사실상 결측"인 경우 계층적 중앙값으로 보정합니다.

#### 보정 전략 (계층적 접근)
1. **1차**: 같은 `item_id`의 중앙값 사용
2. **2차**: 같은 `hs4`의 중앙값 사용 (1차 실패 시)
3. **3차**: 전체 데이터의 중앙값 사용 (2차 실패 시)

#### 처리 과정
```python
# 1. 중앙값 사전 계산 (효율성을 위해 dict로 저장)
item_weight_median = monthly_features.groupby('item_id')['weight_sum'].median().to_dict()
hs4_weight_median = monthly_features.groupby('hs4')['weight_sum'].median().to_dict()
global_weight_median = monthly_features['weight_sum'].median()

# 2. weight_filled 보정
monthly_features['weight_filled'] = monthly_features['weight_sum'].copy()
missing_weight_mask = monthly_features['weight_missing_flag'] == 1

if missing_weight_mask.sum() > 0:
    # item_id 중앙값으로 먼저 시도
    monthly_features.loc[missing_weight_mask, 'weight_filled'] = (
        monthly_features.loc[missing_weight_mask, 'item_id'].map(item_weight_median)
    )
    
    # item_id 중앙값이 없거나 0인 경우 hs4 중앙값으로
    still_missing = (
        (monthly_features['weight_filled'].isna()) | 
        (monthly_features['weight_filled'] == 0)
    )
    monthly_features.loc[still_missing & missing_weight_mask, 'weight_filled'] = (
        monthly_features.loc[still_missing & missing_weight_mask, 'hs4'].map(hs4_weight_median)
    )
    
    # 그것도 안 되면 전체 중앙값
    still_missing = (
        (monthly_features['weight_filled'].isna()) | 
        (monthly_features['weight_filled'] == 0)
    )
    monthly_features.loc[still_missing & missing_weight_mask, 'weight_filled'] = (
        global_weight_median
    )

# 3. value_filled도 동일한 방식으로 보정
```

#### 결과
- **weight_filled 보정**: 91행 (2.4%)
- **value_filled 보정**: 0행 (0%)
- **전체 weight 중앙값**: 27,536
- **전체 value 중앙값**: 382,516
- **새 컬럼**: `weight_filled`, `value_filled`

#### 왜 필요한가?
- **모델 학습**: 결측값이 있으면 모델 학습이 어려움
- **계층적 접근**: 가장 가까운 그룹의 중앙값을 사용하여 현실적인 보정
- **원본 보존**: `weight_sum`, `value_sum`은 그대로 유지하고, 보정된 버전을 별도 컬럼으로 생성

---

### 8️⃣ 극단값(이상치) 처리: 클리핑 + 플래그

#### 목적
99 분위수를 초과하는 극단값을 클리핑하고, 플래그로 표시합니다.

#### 문제 상황
- 무역 데이터는 **스케일 차이가 매우 큼** (예: 100원 ~ 100억원)
- 극단값이 모델 학습에 악영향을 줄 수 있음
- 하지만 극단값 자체도 정보이므로 플래그로 보존

#### 처리 과정
```python
# 1. 99 분위수 계산
q99_value = monthly_features['value_filled'].quantile(0.99)
q99_weight = monthly_features['weight_filled'].quantile(0.99)
q99_quantity = monthly_features['quantity_sum'].quantile(0.99)

# 2. 극단값 플래그
monthly_features['value_extreme_flag'] = (
    (monthly_features['value_filled'] > q99_value).astype(int)
)
monthly_features['weight_extreme_flag'] = (
    (monthly_features['weight_filled'] > q99_weight).astype(int)
)
monthly_features['quantity_extreme_flag'] = (
    (monthly_features['quantity_sum'] > q99_quantity).astype(int)
)

# 3. 클리핑된 컬럼 생성
monthly_features['value_clip'] = np.minimum(
    monthly_features['value_filled'], q99_value
)
monthly_features['weight_clip'] = np.minimum(
    monthly_features['weight_filled'], q99_weight
)
monthly_features['quantity_clip'] = np.minimum(
    monthly_features['quantity_sum'], q99_quantity
)
```

#### 결과
- **99 분위수**:
  - `value`: 103,704,968
  - `weight`: 76,572,396
  - `quantity`: 1,545,930
- **극단값 플래그**:
  - `value_extreme_flag`: 38행 (1.0%)
  - `weight_extreme_flag`: 38행 (1.0%)
  - `quantity_extreme_flag`: 38행 (1.0%)
- **새 컬럼**: `value_extreme_flag`, `weight_extreme_flag`, `quantity_extreme_flag`, `value_clip`, `weight_clip`, `quantity_clip`

#### 왜 필요한가?
- **모델 안정성**: 극단값이 모델 학습을 방해하지 않도록 클리핑
- **정보 보존**: 극단값 플래그를 통해 모델이 극단값 패턴을 학습 가능
- **스케일 정규화**: 로그 변환 전에 클리핑하여 더 안정적인 변환

---

### 9️⃣ 로그 변환 피처

#### 목적
스케일 차이를 줄이고, 정규분포에 가깝게 변환합니다.

#### 왜 로그 변환인가?
- **스케일 차이**: 무역 데이터는 스케일 차이가 매우 큼 (예: 100원 ~ 100억원)
- **정규분포**: 로그 변환 후 정규분포에 가까워져 모델 성능 향상
- **0값 처리**: `log1p`를 사용하여 0값도 안전하게 처리

#### 처리 과정
```python
# log1p = log(1 + x) → 0값도 안전하게 처리
monthly_features['log_value'] = np.log1p(monthly_features['value_clip'])
monthly_features['log_weight'] = np.log1p(monthly_features['weight_clip'])
monthly_features['log_quantity'] = np.log1p(monthly_features['quantity_clip'])
```

#### 결과
- **새 컬럼**: `log_value`, `log_weight`, `log_quantity`
- **범위**: 대략 0 ~ 18 (로그 스케일)
- **용도**: 모델 학습 시 스케일이 정규화된 피처로 사용

#### 왜 필요한가?
- **모델 성능**: 스케일이 정규화된 피처는 모델 학습이 더 안정적
- **상관계수 계산**: 로그 변환 후 상관계수가 더 의미 있게 계산됨
- **이상치 영향 감소**: 로그 변환으로 극단값의 영향이 줄어듦

---

### 🔟 최종 컬럼 구성 및 저장

#### 목적
최종 데이터프레임을 정리하고 CSV 파일로 저장합니다.

#### 최종 컬럼 구성 (27개)

**1. 식별자/키 (7개)**
- `item_id`: 품목 ID
- `hs4`: 4단위 HS 코드
- `hs2`: 2단위 HS 코드
- `year`: 연도
- `month`: 월
- `ym`: datetime (년-월)
- `cluster`: EV 배터리 공급망 클러스터

**2. 집계 원본 (3개)**
- `value_sum`: 무역금액 합계
- `weight_sum`: 중량 합계
- `quantity_sum`: 수량 합계

**3. 결측/0값/플래그 (7개)**
- `has_quantity_info_hs4`: 수량 정보 의미 있는 HS4 여부
- `has_quantity_this_row`: 현재 행에 수량 존재 여부
- `weight_zero_flag`: 중량이 0인지 여부
- `value_zero_flag`: 금액이 0인지 여부
- `weight_missing_flag`: 사실상 중량 결측 여부
- `value_missing_flag`: 사실상 금액 결측 여부
- `value_extreme_flag`: 극단값 여부 (value)
- `weight_extreme_flag`: 극단값 여부 (weight)
- `quantity_extreme_flag`: 극단값 여부 (quantity)

**4. 보정·클리핑 (6개)**
- `weight_filled`: 보정된 중량
- `value_filled`: 보정된 금액
- `value_clip`: 클리핑된 금액
- `weight_clip`: 클리핑된 중량
- `quantity_clip`: 클리핑된 수량

**5. 로그 피처 (3개)**
- `log_value`: 로그 변환된 금액
- `log_weight`: 로그 변환된 중량
- `log_quantity`: 로그 변환된 수량

#### 처리 과정
```python
# 최종 컬럼 순서 정리
final_columns = [
    # 식별자/키
    'item_id', 'hs4', 'hs2', 'year', 'month', 'ym', 'cluster',
    # 집계 원본
    'value_sum', 'weight_sum', 'quantity_sum',
    # 결측/0값/플래그
    'has_quantity_info_hs4', 'has_quantity_this_row',
    'weight_zero_flag', 'value_zero_flag',
    'weight_missing_flag', 'value_missing_flag',
    'value_extreme_flag', 'weight_extreme_flag', 'quantity_extreme_flag',
    # 보정·클리핑
    'weight_filled', 'value_filled',
    'value_clip', 'weight_clip', 'quantity_clip',
    # 로그 피처
    'log_value', 'log_weight', 'log_quantity'
]

monthly_features = monthly_features[final_columns]

# CSV 저장
output_path = 'analysis/data/monthly_features.csv'
monthly_features.to_csv(output_path, index=False)
```

#### 결과
- **파일**: `analysis/data/monthly_features.csv`
- **행 수**: 3,776행
- **컬럼 수**: 27개
- **파일 크기**: 약 0.63 MB

---

## 최종 결과물

### 데이터 요약

| 항목 | 값 |
|------|-----|
| **행 수** | 3,776행 |
| **컬럼 수** | 27개 |
| **item_id 수** | 100개 |
| **기간** | 2022-01 ~ 2025-07 (평균 37.76개 월/item) |
| **클러스터 분포** | battery_materials: 48.1%, other_industries: 36.1%, ev_machinery_electrical: 15.7% |

### 주요 특징

1. **도메인 특화**: EV 배터리 공급망 클러스터 정보 포함
2. **결측 처리**: 계층적 중앙값 보정으로 결측값 처리
3. **이상치 처리**: 99 분위수 클리핑 + 플래그
4. **스케일 정규화**: 로그 변환으로 스케일 차이 해소
5. **정보 보존**: 원본 값과 보정/변환 값 모두 보존

---

## 사용 방법

### 1. 노트북 실행

```bash
cd trade-comovement/analysis
jupyter notebook preprocess_ev_battery.ipynb
```

### 2. Python 스크립트 실행

```bash
cd trade-comovement
python analysis/preprocess_ev_battery.py
```

### 3. 결과 확인

```python
import pandas as pd

# 전처리 결과 로드
monthly_features = pd.read_csv('analysis/data/monthly_features.csv')
print(f"Shape: {monthly_features.shape}")
print(f"Columns: {monthly_features.columns.tolist()}")
```

### 4. 다음 단계

전처리 결과는 `train_model_ev_battery.py`에서 사용됩니다:

```python
# train_model_ev_battery.py에서
monthly_features = pd.read_csv('analysis/data/monthly_features.csv')
```

---

## 참고 사항

### 데이터 품질 체크리스트

- ✅ 모든 item_id에 hs4 매핑 존재
- ✅ 월별 집계 완료 (중복 없음)
- ✅ 클러스터 라벨링 완료
- ✅ 결측값 보정 완료
- ✅ 이상치 클리핑 완료
- ✅ 로그 변환 완료

### 주의사항

1. **원본 보존**: `value_sum`, `weight_sum`, `quantity_sum`은 원본 값 유지
2. **보정 값 사용**: 모델 학습 시 `value_filled`, `weight_filled` 사용 권장
3. **클리핑 값 사용**: 로그 변환 전에 클리핑된 값 사용 (`value_clip`, `weight_clip`)
4. **플래그 활용**: 결측/이상치 플래그를 피처로 사용하여 모델 성능 향상 가능

---

**작성일**: 2024-11-18  
**버전**: 1.0  
**관련 파일**: `analysis/preprocess_ev_battery.ipynb`, `analysis/preprocess_ev_battery.py`

