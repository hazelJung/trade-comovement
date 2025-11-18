# preprocessing.ipynb 전처리 과정 가이드

이 문서는 `preprocessing.ipynb` 노트북의 전체 전처리 과정을 상세히 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [전처리 단계별 상세 설명](#전처리-단계별-상세-설명)
3. [EDA 및 분석](#eda-및-분석)
4. [최종 결과물](#최종-결과물)
5. [사용 방법](#사용-방법)

---

## 개요

### 목적
무역 데이터를 **공행성(Comovement) 분석**에 적합한 형태로 전처리하여, 시계열 패널 데이터를 생성합니다.

### 핵심 개념
- **Panel Completion**: 모든 item_id × 모든 month 조합을 생성하여 시계열 길이를 일치시킴
- **Zero/Missing Flag**: 0값과 결측값을 플래그로 표시하여 패턴을 피처로 활용
- **시계열 특성 추출**: 이동평균, 변화율, 계절성 등 시계열 특성 생성

### 입력 데이터
- **파일**: `data/train.csv`
- **형식**: 무역 거래 데이터 (item_id, year, month, hs4, weight, quantity, value 등)
- **기간**: 2022년 1월 ~ 2025년 7월

### 출력 데이터
- **데이터프레임**: `df_panel` (모든 item_id × 모든 month 조합)
- **행 수**: 약 4,200행 (100개 item_id × 42개월)
- **컬럼 수**: 약 30개 이상 (원본 + 플래그 + 공급망 정보 + 시계열 특성)

---

## 전처리 단계별 상세 설명

### 1️⃣ Zero/Missing Flag 생성

#### 목적
0값과 결측값을 제거하지 않고, **패턴을 피처로 인식**하기 위해 플래그를 생성합니다.

#### 처리 과정
```python
# Zero flag 생성
df_train['value_zero'] = (df_train['value'] == 0).astype(int)
df_train['weight_zero'] = (df_train['weight'] == 0).astype(int)
df_train['quantity_zero'] = (df_train['quantity'] == 0).astype(int)

# Missing flag 생성
df_train['value_missing'] = df_train['value'].isna().astype(int)
df_train['weight_missing'] = df_train['weight'].isna().astype(int)
df_train['quantity_missing'] = df_train['quantity'].isna().astype(int)
```

#### 결과
- **새 컬럼**: `value_zero`, `weight_zero`, `quantity_zero`, `value_missing`, `weight_missing`, `quantity_missing`
- **용도**: 모델 학습 시 0값/결측값 패턴을 피처로 활용

#### 왜 필요한가?
- **0값의 의미**: 0값은 "거래 없음"을 의미할 수 있지만, 때로는 "정보 미기입"일 수도 있음
- **패턴 학습**: 어떤 item_id가 자주 0값을 가지는지, 어떤 시기에 0값이 많은지 등 패턴 학습
- **결측 처리**: 결측값을 보간하지 않고 플래그로 표시하여 모델이 결측 패턴을 학습

---

### 2️⃣ 월별 집계 (Monthly Aggregation)

#### 목적
같은 `item_id`의 같은 월에 발생한 모든 거래를 하나의 행으로 집계합니다.

#### 처리 과정
```python
# item_id + hs4 + date 기준으로 value/weight/quantity 합산
agg_cols = ["value", "weight", "quantity"]

df_train_monthly = (
    df_train
    .groupby(["item_id", "hs4", "date"], as_index=False)[agg_cols]
    .sum()
)
```

#### 결과
- **입력**: 10,836행 (원본 거래 데이터)
- **출력**: 약 3,776행 (월별 집계 데이터)
- **집계율**: 약 35% (같은 item_id의 같은 월에 평균 2.87건의 거래)

#### 왜 필요한가?
- **시계열 분석**: 공행성 분석은 월별 시계열 데이터를 사용
- **데이터 축소**: 거래 레벨 → 월별 레벨로 축소하여 분석 용이
- **노이즈 감소**: 같은 월의 여러 거래를 합산하여 안정적인 패턴 추출

---

### 3️⃣ Panel Completion (패널 완성)

#### 목적
**모든 item_id × 모든 month 조합**을 생성하여 시계열 길이를 일치시킵니다.

#### 문제 상황
- **시계열 길이 불일치**: 
  - A item: 2022.01 ~ 2025.07 모든 달에 존재 = 42개월
  - C item: 2022.03 ~ 2023.05 일부 달에 존재 = 15개월
- **상관계수 계산 불가**: 시계열 길이가 다르면 공행성 상관계수를 계산할 수 없음

#### 처리 과정
```python
# 1. 전체 month 범위 & item_id 목록 만들기
min_month = df_train_monthly["date"].min()
max_month = df_train_monthly["date"].max()
full_months = pd.date_range(min_month, max_month, freq="MS")
all_items = df_train_monthly["item_id"].unique()

# 2. item_id × date 전체 조합 (풀 패널 인덱스)
panel = pd.MultiIndex.from_product(
    [all_items, full_months],
    names=["item_id", "date"]
).to_frame(index=False)

# 3. 집계된 df_train_monthly와 merge해서 패널 완성
df_panel = (
    panel.merge(df_train_monthly, on=["item_id", "date"], how="left")
         .sort_values(["item_id", "date"])
         .reset_index(drop=True)
)

# 4. 거래 없던 month → NaN을 0으로 채우기
for c in agg_cols:
    df_panel[c] = df_panel[c].fillna(0)
```

#### 결과
- **행 수**: 약 4,200행 (100개 item_id × 42개월)
- **거래 없는 월**: 0으로 채움
- **시계열 길이**: 모든 item_id가 동일한 42개월 시계열을 가짐

#### 왜 필요한가?
- **공행성 분석 필수**: 시계열 길이가 일치해야 상관계수 계산 가능
- **시계열 분석**: 모든 item_id가 같은 기간의 데이터를 가져야 비교 가능
- **모델 학습**: 시계열 모델은 일정한 길이의 시계열을 요구

---

### 4️⃣ HS4 매핑 보완

#### 목적
Panel Completion 과정에서 발생한 HS4 NaN 값을 채웁니다.

#### 가정
- **1:1 관계**: 하나의 `item_id`는 항상 동일한 `hs4` 코드를 가집니다.
- **역관계**: 하나의 `hs4` 코드는 여러 `item_id`를 가질 수 있습니다.

#### 처리 과정
```python
# item_id는 항상 같은 HS4를 가지므로 (1:1 관계), Panel에서 NaN인 HS4를 채움
item_hs4_mapping = (
    df_panel[df_panel['hs4'].notna()]
    .groupby('item_id')['hs4']
    .first()  # 모든 값이 같으므로 first() 사용
    .to_dict()
)

# Panel의 NaN 채우기
df_panel['hs4'] = df_panel['item_id'].map(item_hs4_mapping).fillna(df_panel['hs4'])
```

#### 결과
- **HS4 NaN 개수**: 524개 → 0개
- **HS4별 평균 item_id 수**: 1.41개
- **최대 item_id 수를 가진 HS4**: 2805 (6개)

#### 왜 필요한가?
- **데이터 완전성**: HS4 정보가 없으면 공급망 분석 불가
- **1:1 관계 활용**: item_id는 항상 같은 HS4를 가지므로 매핑 가능
- **후속 분석**: HS4 기반 공급망 분석을 위해 필수

---

### 5️⃣ HS2 코드 및 이름 매핑

#### 목적
HS4 코드에서 HS2 코드를 추출하고, 한국어 이름을 매핑합니다.

#### 처리 과정
```python
# HS2 코드 추출 (HS4의 앞 2자리)
df_panel["hs2"] = df_panel["hs4"].astype(str).str[:2]

# HS2 한국어 이름 매핑
hs2_map = {
    "27": "광물성 연료·에너지",
    "28": "무기화합물",
    "29": "유기화합물",
    # ... (97개 HS2 코드)
}
df_panel["hs2_name_kr"] = df_panel["hs2"].map(hs2_map)
```

#### 결과
- **새 컬럼**: `hs2`, `hs2_name_kr`
- **용도**: HS2 코드는 더 넓은 범주로, 공급망 분석에 유용

---

### 6️⃣ Zero/Missing Flag 재생성 (Panel 기준)

#### 목적
Panel Completion 후 `df_panel`에 Zero/Missing Flag를 생성합니다.

#### 처리 과정
```python
# df_panel에는 flag가 없으므로 직접 생성
df_panel['value_zero'] = (df_panel['value'] == 0).astype(int)
df_panel['weight_zero'] = (df_panel['weight'] == 0).astype(int)
df_panel['quantity_zero'] = (df_panel['quantity'] == 0).astype(int)

df_panel['value_missing'] = df_panel['value'].isna().astype(int)
df_panel['weight_missing'] = df_panel['weight'].isna().astype(int)
df_panel['quantity_missing'] = df_panel['quantity'].isna().astype(int)
```

#### 결과
- **새 컬럼**: `value_zero`, `weight_zero`, `quantity_zero`, `value_missing`, `weight_missing`, `quantity_missing`
- **용도**: Panel 데이터 기준으로 0값/결측값 패턴 분석

---

### 7️⃣ 공급망 정보 추가

#### 목적
HS4 코드를 기반으로 공급망 계층(Tier) 정보를 추가합니다.

#### 공급망 계층 정의

| Tier | HS4 범위 | 의미 | 예시 |
|------|---------|------|------|
| **Tier 1** | 2800-3899 | 원자재 (화학제품) | 화학제품, 플라스틱 |
| **Tier 2** | 7200-8399 | 중간재 (금속제품) | 철강, 금속제품 |
| **Tier 3** | 8400-8599 | 최종제품 (기계류) | 기계류, 전기기기 |
| **기타** | 나머지 | 기타 산업 | 기타 제품 |

#### 처리 과정
```python
def get_supply_chain_tier(hs4):
    """HS4 코드를 기반으로 공급망 계층 반환"""
    if pd.isna(hs4):
        return 0
    hs4_int = int(hs4)
    if 2800 <= hs4_int <= 3899:  # 화학제품
        return 1  # Tier 1 (원자재)
    elif 7200 <= hs4_int <= 8399:  # 금속제품
        return 2  # Tier 2 (중간재)
    elif 8400 <= hs4_int <= 8599:  # 기계류
        return 3  # Tier 3 (최종제품)
    else:
        return 0  # 기타

# 공급망 계층 추가
df_panel['supply_chain_tier'] = df_panel['hs4'].apply(get_supply_chain_tier)

# HS4별 item_id 수 (같은 HS4를 가진 item_id가 몇 개인지)
hs4_item_count = df_panel.groupby('hs4')['item_id'].nunique().to_dict()
df_panel['hs4_item_count'] = df_panel['hs4'].map(hs4_item_count).fillna(0)

# HS4별 평균 거래량 (같은 HS4 그룹의 평균과 비교)
hs4_avg_value = df_panel.groupby('hs4')['value'].mean().to_dict()
df_panel['hs4_avg_value'] = df_panel['hs4'].map(hs4_avg_value).fillna(0)
df_panel['value_vs_hs4_avg'] = df_panel['value'] / (df_panel['hs4_avg_value'] + 1e-8)
```

#### 결과
- **공급망 계층 분포**:
  - Tier 0 (기타): 1,591행
  - Tier 1 (원자재): 1,849행
  - Tier 2 (중간재): 344행
  - Tier 3 (최종제품): 516행
- **Tier별 item_id 수**:
  - 기타: 37개
  - Tier 1 (원자재): 43개
  - Tier 2 (중간재): 8개
  - Tier 3 (최종제품): 12개
- **새 컬럼**: `supply_chain_tier`, `hs4_item_count`, `hs4_avg_value`, `value_vs_hs4_avg`

#### 왜 필요한가?
- **도메인 지식 활용**: 공급망 계층 정보를 활용하여 공행성 쌍 탐색
- **공급망 방향성**: Tier 1 → Tier 2 → Tier 3 방향으로 공행성이 높을 가능성
- **상대적 비교**: `value_vs_hs4_avg`로 같은 HS4 그룹 내에서의 상대적 위치 파악

---

### 8️⃣ 시계열 특성 추출

#### 목적
시계열 분석에 유용한 특성들을 생성합니다.

#### 추출된 특성

**1. 이동평균 (Moving Average)**
```python
for window in [3, 6, 12]:  # 3개월, 6개월, 12개월 이동평균
    df_panel[f'value_ma{window}'] = (
        df_panel.groupby('item_id')['value']
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
```
- **용도**: 단기/중기/장기 추세 파악
- **특성**: `value_ma3`, `value_ma6`, `value_ma12`

**2. 변화율 (Percentage Change)**
```python
df_panel['value_pct_change'] = (
    df_panel.groupby('item_id')['value'].pct_change()
)
df_panel['value_pct_change'] = df_panel['value_pct_change'].fillna(0)
```
- **용도**: 전월 대비 변화율로 급증/급감 탐지
- **특성**: `value_pct_change`

**3. 계절성 (Seasonality)**
```python
df_panel['month_sin'] = np.sin(2 * np.pi * df_panel['month'] / 12)
df_panel['month_cos'] = np.cos(2 * np.pi * df_panel['month'] / 12)
df_panel['quarter'] = df_panel['month'].apply(lambda x: (x-1)//3 + 1)
```
- **용도**: 월별/분기별 계절성 패턴 학습
- **특성**: `month_sin`, `month_cos`, `quarter`

**4. 추세 (Trend)**
```python
def calculate_trend(series):
    """시계열의 선형 추세 기울기 계산"""
    if len(series) < 2:
        return 0
    x = np.arange(len(series))
    try:
        coeffs = np.polyfit(x, series, 1)
        return coeffs[0]  # 기울기
    except:
        return 0

df_panel['value_trend'] = (
    df_panel.groupby('item_id')['value']
    .transform(lambda x: calculate_trend(x.values))
)
```
- **용도**: 장기 추세 방향 파악 (증가/감소)
- **특성**: `value_trend`

**5. 로그 변환 (Log Transformation)**
```python
df_panel['value_log'] = np.log1p(df_panel['value'])  # log1p: 0 값 처리
df_panel['weight_log'] = np.log1p(df_panel['weight'])
```
- **용도**: 스케일 차이를 줄이고 정규분포에 가깝게 변환
- **특성**: `value_log`, `weight_log`

#### 결과
- **총 추가된 컬럼**: `value_ma3`, `value_ma6`, `value_ma12`, `value_pct_change`, `month_sin`, `month_cos`, `quarter`, `value_trend`, `value_log`, `weight_log`

#### 왜 필요한가?
- **모델 성능 향상**: 시계열 특성을 피처로 사용하여 모델 성능 향상
- **패턴 학습**: 이동평균, 변화율, 계절성 등 패턴을 모델이 학습
- **스케일 정규화**: 로그 변환으로 스케일 차이 해소

---

## EDA 및 분석

### 1. HS4 / item_id 군집성 분석

#### HS4별 평균 거래량
```python
hs4_summary = (
    df.groupby("hs4")[["value", "weight"]]
      .mean()
      .reset_index()
      .sort_values("value", ascending=False)
)
```
- **목적**: 어떤 HS4 품목군의 평균 거래금액과 평균 중량이 높은지 파악
- **용도**: 품목군의 경제적 규모 이해

#### HS4별 quantity 사용률
```python
hs4_qty_rate = (
    df.assign(qty_valid=(df["quantity"] > 0))
      .groupby("hs4")["qty_valid"]
      .mean()
      .reset_index(name="quantity_usage_rate")
      .sort_values("quantity_usage_rate", ascending=False)
)
```
- **목적**: 어떤 HS4는 quantity가 자주 기록되지만, 어떤 HS4는 quantity가 거의 0 또는 누락
- **용도**: 품목군 특성 파악, 모델링 시 quantity를 feature로 사용할지 판단

#### item_id별 거래 규모
```python
item_volume = (
    df.groupby("item_id")["value"]
      .sum()
      .reset_index()
      .sort_values("value", ascending=False)
)
```
- **목적**: 가장 거래규모가 큰 품목과 작은 품목 파악
- **용도**: 예측 우선순위 선정, 군집분석 시 중요

#### item_id별 월 커버리지
```python
item_month_coverage = (
    df.groupby("item_id")["date"]
      .nunique()
      .reset_index(name="month_covered")
      .sort_values("month_covered", ascending=False)
)
```
- **목적**: 모든 item_id가 모든 월에 등장하는지 확인
- **용도**: 모든 월에 등장하는 item_id만 모델링에 활용

---

### 2. item_id Micro-level Deep Dive

#### 샘플 item_id 추출 및 시계열 분석
```python
# 8개 랜덤 sample
sample_items = np.random.choice(df['item_id'].unique(), size=8, replace=False)
sample_df = df[df['item_id'].isin(sample_items)]

monthly_trend = (
    sample_df.groupby(['item_id', 'date'])
             [['value', 'weight', 'quantity']]
             .sum()
             .reset_index()
)
```

#### Zero/Missing 패턴 분석
```python
zero_missing_stats = (
    df[df['item_id'].isin(sample_items)]
    .groupby('item_id')[['value_zero', 'weight_zero', 'quantity_zero',
                         'value_missing', 'weight_missing', 'quantity_missing']]
    .mean()
    .reset_index()
)
```
- **목적**: 각 item_id별로 0값/결측값 비율 파악
- **용도**: 데이터 품질 평가, 모델링 전략 수립

#### 급증/급감 패턴(Anomaly) 탐지
```python
# 전월 대비 변화율
monthly_trend['value_pct_change'] = (
    monthly_trend.groupby('item_id')['value'].pct_change()
)

# 급증 top 5, 급감 bottom 5
top_spikes = monthly_trend.sort_values('value_pct_change', ascending=False).head(5)
bottom_spikes = monthly_trend.sort_values('value_pct_change', ascending=True).head(5)
```
- **목적**: 급증/급감 패턴을 탐지하여 이상치 또는 특이 패턴 파악
- **용도**: 이상치 처리 전략 수립

#### HS 코드 일관성 검증
```python
# HS 코드 cross-check (item_id -> hs4 mapping consistency)
item_hs_check = (
    df.groupby('item_id')['hs4']
      .nunique()
      .reset_index(name='hs4_count')
)

# HS4가 여러 개 존재하는 item만 추출
item_hs_issue = item_hs_check[item_hs_check['hs4_count'] > 1]
```
- **목적**: item_id가 항상 같은 HS4를 가지는지 검증
- **결과**: 모든 item_id는 하나의 HS4만 가짐 (1:1 관계 확인)

---

## 최종 결과물

### 데이터 요약

| 항목 | 값 |
|------|-----|
| **행 수** | 약 4,200행 (100개 item_id × 42개월) |
| **컬럼 수** | 약 30개 이상 |
| **item_id 수** | 100개 |
| **기간** | 2022-01 ~ 2025-07 (42개월) |
| **시계열 길이** | 모든 item_id가 동일한 42개월 시계열 |

### 주요 컬럼 구성

**1. 식별자/키 (7개)**
- `item_id`: 품목 ID
- `hs4`: 4단위 HS 코드
- `hs2`: 2단위 HS 코드
- `hs2_name_kr`: HS2 한국어 이름
- `date`: 날짜 (datetime)
- `year`: 연도
- `month`: 월

**2. 거래량 원본 (3개)**
- `value`: 무역금액
- `weight`: 중량
- `quantity`: 수량

**3. Zero/Missing 플래그 (6개)**
- `value_zero`, `weight_zero`, `quantity_zero`
- `value_missing`, `weight_missing`, `quantity_missing`

**4. 공급망 정보 (4개)**
- `supply_chain_tier`: 공급망 계층 (0, 1, 2, 3)
- `hs4_item_count`: 같은 HS4를 가진 item_id 수
- `hs4_avg_value`: HS4별 평균 거래량
- `value_vs_hs4_avg`: HS4 평균 대비 상대적 거래량

**5. 시계열 특성 (10개)**
- `value_ma3`, `value_ma6`, `value_ma12`: 이동평균
- `value_pct_change`: 변화율
- `month_sin`, `month_cos`, `quarter`: 계절성
- `value_trend`: 추세
- `value_log`, `weight_log`: 로그 변환

### 주요 특징

1. **Panel Completion**: 모든 item_id × 모든 month 조합으로 시계열 길이 일치
2. **Zero/Missing Flag**: 0값과 결측값을 플래그로 표시하여 패턴 학습
3. **공급망 정보**: HS4 기반 공급망 계층 정보 포함
4. **시계열 특성**: 이동평균, 변화율, 계절성, 추세 등 다양한 시계열 특성
5. **EDA 완료**: HS4/item_id 군집성 분석 및 micro-level deep dive 수행

---

## 사용 방법

### 1. 노트북 실행

```bash
cd trade-comovement/analysis
jupyter notebook preprocessing.ipynb
```

### 2. 결과 확인

```python
# df_panel 확인
print(f"Shape: {df_panel.shape}")
print(f"Columns: {df_panel.columns.tolist()}")
print(f"시계열 길이: {df_panel.groupby('item_id')['date'].nunique().describe()}")
```

### 3. 다음 단계

전처리 결과(`df_panel`)는 `train_model_preprocessing.py`에서 사용됩니다:

```python
# train_model_preprocessing.py에서
# df_panel을 사용하여 공행성 쌍 탐색 및 모델 학습
```

---

## 참고 사항

### 데이터 품질 체크리스트

- ✅ 모든 item_id × 모든 month 조합 생성 (Panel Completion)
- ✅ 시계열 길이 일치 (모든 item_id가 42개월)
- ✅ HS4 매핑 완료 (NaN 없음)
- ✅ Zero/Missing Flag 생성 완료
- ✅ 공급망 정보 추가 완료
- ✅ 시계열 특성 추출 완료

### 주의사항

1. **Panel Completion 필수**: 공행성 분석을 위해서는 반드시 Panel Completion 필요
2. **0값 처리**: 0값을 제거하지 않고 플래그로 표시하여 패턴 학습
3. **결측 처리**: 결측값을 보간하지 않고 플래그로 표시
4. **시계열 특성 활용**: 이동평균, 변화율 등 시계열 특성을 피처로 활용 권장

---

## preprocessing.ipynb vs preprocess_ev_battery.ipynb 비교

| 항목 | preprocessing.ipynb | preprocess_ev_battery.ipynb |
|------|---------------------|------------------------------|
| **목적** | 공행성 분석용 Panel 데이터 생성 | EV 배터리 도메인 특화 전처리 |
| **핵심 기능** | Panel Completion | 결측치 보정, 이상치 처리 |
| **출력** | `df_panel` (메모리) | `monthly_features.csv` (파일) |
| **특징** | 시계열 특성, 공급망 정보 | 클러스터 정보, 계층적 보정 |
| **사용 모델** | `train_model_preprocessing.py` | `train_model_ev_battery.py` |

---

**작성일**: 2024-11-18  
**버전**: 1.0  
**관련 파일**: `analysis/preprocessing.ipynb`, `models/train_model_preprocessing.py`

