# 전처리 개선 제안서

## 📋 현재 전처리 과정 요약

### ✅ 잘 하고 있는 부분
1. **Zero/Missing Flag 생성**: 0과 결측치를 플래그로 표시
2. **월별 집계**: item_id + hs4 + date 기준 집계
3. **Panel Completion**: 모든 item_id × 모든 month 조합 완성
4. **HS2 매핑**: 상위 분류 코드 활용

### ⚠️ 개선이 필요한 부분

---

## 🔧 개선 제안

### 1. **HS4 매핑 보완** ⭐ 중요도: 중간

**데이터 구조 이해:**
- ✅ 하나의 `item_id`에는 하나의 `hs4` 코드만 존재 (1:1 관계)
- ✅ 하나의 `hs4` 코드에는 여러 `item_id`가 매칭됨 (1:N 관계)
- ⚠️ Panel 생성 시 거래가 없는 월에 HS4가 NaN으로 남을 수 있음

**개선 방안:**
```python
# HS4 매핑 보완 (Panel에서 NaN 처리)
def fill_hs4_mapping(df_panel):
    """
    item_id별 HS4 매핑을 생성하고 Panel의 NaN을 채움
    - item_id는 항상 같은 HS4를 가지므로 매핑 테이블 생성
    """
    # item_id별 고유 HS4 매핑 (1:1 관계이므로 첫 번째 값만 사용)
    item_hs4_mapping = (
        df_panel[df_panel['hs4'].notna()]
        .groupby('item_id')['hs4']
        .first()  # 모든 값이 같으므로 first() 사용
        .to_dict()
    )
    
    # Panel의 NaN 채우기
    df_panel['hs4'] = df_panel['item_id'].map(item_hs4_mapping).fillna(df_panel['hs4'])
    
    # HS4별 item_id 수 통계 (1:N 관계 확인)
    hs4_item_count = df_panel.groupby('hs4')['item_id'].nunique()
    print(f"HS4별 평균 item_id 수: {hs4_item_count.mean():.2f}")
    print(f"최대 item_id 수를 가진 HS4: {hs4_item_count.idxmax()} ({hs4_item_count.max()}개)")
    
    return df_panel
```

**현재 코드 문제점:**
- `preprocessing.ipynb` Cell 12에서 `value`, `weight`, `quantity`만 fillna(0) 처리
- **HS4는 fillna 처리하지 않아서 Panel에서 NaN으로 남음**
- item_id는 항상 같은 HS4를 가지므로 매핑 테이블로 채울 수 있음

**즉시 적용 가능한 수정:**
```python
# Cell 12 수정 예시
# 거래량 관련 수치 컬럼
value_cols = ['value', 'weight', 'quantity']
df_panel[value_cols] = df_panel[value_cols].fillna(0)

# HS4 매핑 보완 (추가)
item_hs4_mapping = (
    df_panel[df_panel['hs4'].notna()]
    .groupby('item_id')['hs4']
    .first()
    .to_dict()
)
df_panel['hs4'] = df_panel['item_id'].map(item_hs4_mapping).fillna(df_panel['hs4'])

# zero / missing flag 처리
flag_cols = [
    'value_zero', 'weight_zero', 'quantity_zero',
    'value_missing', 'weight_missing', 'quantity_missing'
]
for c in flag_cols:
    if c in df_panel.columns:
        df_panel[c] = df_panel[c].fillna(0).astype(int)
```

---

### 2. **Type 컬럼 활용** ⭐ 중요도: 중간

**현재 문제:**
- `type` 컬럼이 전처리에서 완전히 무시됨
- 무역 유형(수입/수출 등)에 따라 패턴이 다를 수 있음

**개선 방안:**
```python
# Type별 집계 또는 Type을 feature로 활용
def add_type_features(df_panel, df_train):
    """
    Type 정보를 패널에 추가
    - Type별 집계
    - Type 플래그
    """
    # item_id별 주요 type (가장 빈번한 type)
    item_type_mapping = (
        df_train.groupby('item_id')['type']
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        .to_dict()
    )
    
    df_panel['type'] = df_panel['item_id'].map(item_type_mapping)
    df_panel['type'] = df_panel['type'].fillna(1)  # 기본값
    
    # Type별 dummy 변수
    type_dummies = pd.get_dummies(df_panel['type'], prefix='type')
    df_panel = pd.concat([df_panel, type_dummies], axis=1)
    
    return df_panel
```

---

### 3. **이상치(Outlier) 처리** ⭐ 중요도: 높음

**현재 문제:**
- 극단적인 값이 상관계수 계산에 영향을 줄 수 있음
- 급증/급감 패턴이 노이즈일 수 있음

**개선 방안:**
```python
def detect_and_handle_outliers(df_panel, method='iqr', clip=True):
    """
    이상치 탐지 및 처리
    - IQR 방법
    - Z-score 방법
    - 로그 변환 후 처리
    """
    value_cols = ['value', 'weight', 'quantity']
    
    for col in value_cols:
        if method == 'iqr':
            Q1 = df_panel[col].quantile(0.25)
            Q3 = df_panel[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 3 IQR 사용 (더 관대하게)
            upper_bound = Q3 + 3 * IQR
            
        elif method == 'zscore':
            z_scores = np.abs((df_panel[col] - df_panel[col].mean()) / df_panel[col].std())
            lower_bound = df_panel[col][z_scores < 3].min()
            upper_bound = df_panel[col][z_scores < 3].max()
        
        # 이상치 플래그
        df_panel[f'{col}_outlier'] = (
            (df_panel[col] < lower_bound) | (df_panel[col] > upper_bound)
        ).astype(int)
        
        # 이상치 클리핑 또는 제거
        if clip:
            df_panel[col] = df_panel[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_panel
```

---

### 4. **스케일링 및 로그 변환** ⭐ 중요도: 중간

**현재 문제:**
- Value의 스케일이 매우 크게 다름 (수백만 ~ 수억)
- 상관계수 계산 시 큰 값에 편향될 수 있음

**개선 방안:**
```python
def add_scaled_features(df_panel):
    """
    스케일링된 feature 추가
    - 로그 변환 (log1p 사용: 0 값 처리)
    - 정규화
    """
    # 로그 변환 (0 값 처리)
    df_panel['value_log'] = np.log1p(df_panel['value'])
    df_panel['weight_log'] = np.log1p(df_panel['weight'])
    df_panel['quantity_log'] = np.log1p(df_panel['quantity'])
    
    # item_id별 정규화 (Z-score)
    for col in ['value', 'weight', 'quantity']:
        mean = df_panel.groupby('item_id')[col].transform('mean')
        std = df_panel.groupby('item_id')[col].transform('std')
        df_panel[f'{col}_normalized'] = (df_panel[col] - mean) / (std + 1e-8)
    
    return df_panel
```

---

### 5. **시계열 특성 추출** ⭐ 중요도: 높음

**현재 문제:**
- 시계열 패턴(trend, seasonality)을 활용하지 않음
- 이동평균, 변화율 등 유용한 feature 미사용

**개선 방안:**
```python
def add_time_series_features(df_panel):
    """
    시계열 특성 추가
    - 이동평균 (MA)
    - 변화율 (pct_change)
    - 계절성 (month, quarter)
    - 추세 (trend)
    """
    df_panel = df_panel.sort_values(['item_id', 'date'])
    
    # 이동평균
    for window in [3, 6, 12]:  # 3개월, 6개월, 12개월
        df_panel[f'value_ma{window}'] = (
            df_panel.groupby('item_id')['value']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
    
    # 변화율
    df_panel['value_pct_change'] = (
        df_panel.groupby('item_id')['value'].pct_change()
    )
    df_panel['value_pct_change'] = df_panel['value_pct_change'].fillna(0)
    
    # 계절성
    df_panel['month_sin'] = np.sin(2 * np.pi * df_panel['month'] / 12)
    df_panel['month_cos'] = np.cos(2 * np.pi * df_panel['month'] / 12)
    df_panel['quarter'] = df_panel['month'].apply(lambda x: (x-1)//3 + 1)
    
    # 추세 (선형 추세 점수)
    def calculate_trend(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series, 1)
        return coeffs[0]  # 기울기
    
    df_panel['value_trend'] = (
        df_panel.groupby('item_id')['value']
        .transform(lambda x: calculate_trend(x.values))
    )
    
    return df_panel
```

---

### 6. **Quantity 활용 개선** ⭐ 중요도: 중간

**현재 문제:**
- Quantity가 대부분 0이지만, 0이 아닌 경우 유용한 정보일 수 있음
- Value/Weight 대비 Quantity 비율이 의미 있을 수 있음

**개선 방안:**
```python
def add_quantity_features(df_panel):
    """
    Quantity 관련 feature 추가
    - 단가 계산 (value/quantity, value/weight)
    - Quantity 사용률
    """
    # 단가 계산 (0으로 나누기 방지)
    df_panel['unit_price'] = np.where(
        df_panel['quantity'] > 0,
        df_panel['value'] / df_panel['quantity'],
        0
    )
    
    df_panel['value_per_weight'] = np.where(
        df_panel['weight'] > 0,
        df_panel['value'] / df_panel['weight'],
        0
    )
    
    # Quantity 사용률 (item_id별)
    item_qty_usage = (
        df_panel.groupby('item_id')
        .apply(lambda x: (x['quantity'] > 0).sum() / len(x))
        .to_dict()
    )
    df_panel['quantity_usage_rate'] = df_panel['item_id'].map(item_qty_usage)
    
    return df_panel
```

---

### 7. **Panel Completion 개선** ⭐ 중요도: 중간

**현재 문제:**
- 거래가 없는 월을 무조건 0으로 채움
- 실제로는 "거래 없음"과 "0 거래"를 구분할 수 없음

**개선 방안:**
```python
def improved_panel_completion(df_train_monthly, fill_method='zero'):
    """
    Panel completion 개선
    - fill_method: 'zero', 'forward_fill', 'interpolate', 'nan'
    """
    # ... 기존 패널 생성 코드 ...
    
    if fill_method == 'forward_fill':
        # Forward fill (마지막 관측값으로 채우기)
        df_panel = df_panel.sort_values(['item_id', 'date'])
        for col in ['value', 'weight', 'quantity']:
            df_panel[col] = df_panel.groupby('item_id')[col].ffill()
            df_panel[col] = df_panel[col].fillna(0)
    
    elif fill_method == 'interpolate':
        # 선형 보간
        for col in ['value', 'weight', 'quantity']:
            df_panel[col] = df_panel.groupby('item_id')[col].apply(
                lambda x: x.interpolate(method='linear')
            ).fillna(0)
    
    # 거래 존재 여부 플래그 추가
    df_panel['has_transaction'] = (
        (df_panel['value'] > 0) | 
        (df_panel['weight'] > 0) | 
        (df_panel['quantity'] > 0)
    ).astype(int)
    
    return df_panel
```

---

### 8. **HS4 공급망 정보 활용** ⭐ 중요도: 높음

**현재 문제:**
- HS4 코드만 있고 공급망 계층 정보가 없음
- train_model_supply_chain.py에서 사용하지만 전처리 단계에서 미리 추가 가능
- HS4별 item_id 수 정보를 활용하지 않음

**개선 방안:**
```python
def add_supply_chain_features(df_panel):
    """
    공급망 계층 정보 추가
    - Tier 1 (원자재)
    - Tier 2 (중간재)
    - Tier 3 (최종제품)
    - HS4별 item_id 수 (1:N 관계 활용)
    """
    def get_supply_chain_tier(hs4):
        if pd.isna(hs4):
            return 0
        hs4_int = int(hs4)
        if 2800 <= hs4_int <= 3899:  # 화학제품
            return 1  # Tier 1
        elif 7200 <= hs4_int <= 8399:  # 금속제품
            return 2  # Tier 2
        elif 8400 <= hs4_int <= 8599:  # 기계류
            return 3  # Tier 3
        else:
            return 0  # 기타
    
    df_panel['supply_chain_tier'] = df_panel['hs4'].apply(get_supply_chain_tier)
    
    # Tier별 dummy 변수
    tier_dummies = pd.get_dummies(df_panel['supply_chain_tier'], prefix='tier')
    df_panel = pd.concat([df_panel, tier_dummies], axis=1)
    
    # HS4별 item_id 수 (같은 HS4를 가진 item_id가 몇 개인지)
    # → 공급망에서 해당 HS4의 중요도/다양성 지표로 활용 가능
    hs4_item_count = df_panel.groupby('hs4')['item_id'].nunique().to_dict()
    df_panel['hs4_item_count'] = df_panel['hs4'].map(hs4_item_count).fillna(0)
    
    # HS4별 평균 거래량 (같은 HS4 그룹의 평균과 비교)
    hs4_avg_value = df_panel.groupby('hs4')['value'].mean().to_dict()
    df_panel['hs4_avg_value'] = df_panel['hs4'].map(hs4_avg_value).fillna(0)
    df_panel['value_vs_hs4_avg'] = df_panel['value'] / (df_panel['hs4_avg_value'] + 1e-8)
    
    return df_panel
```

---

### 9. **데이터 품질 검증** ⭐ 중요도: 중간

**개선 방안:**
```python
def validate_preprocessing(df_panel):
    """
    전처리 결과 검증
    - 중복 체크
    - 결측치 체크
    - 데이터 타입 체크
    """
    checks = {}
    
    # 중복 체크
    checks['duplicates'] = df_panel.duplicated(['item_id', 'date']).sum()
    
    # 결측치 체크
    checks['missing_values'] = df_panel[['value', 'weight', 'quantity']].isna().sum().to_dict()
    
    # 음수 체크
    checks['negative_values'] = {
        'value': (df_panel['value'] < 0).sum(),
        'weight': (df_panel['weight'] < 0).sum(),
        'quantity': (df_panel['quantity'] < 0).sum()
    }
    
    # 패널 완성도 체크
    expected_rows = len(df_panel['item_id'].unique()) * len(df_panel['date'].unique())
    checks['panel_completeness'] = len(df_panel) / expected_rows
    
    print("=== 전처리 검증 결과 ===")
    for key, value in checks.items():
        print(f"{key}: {value}")
    
    return checks
```

---

## 📝 통합 개선 코드 예시

```python
def improved_preprocessing(df_train):
    """
    개선된 전처리 파이프라인
    """
    # 1. 기본 전처리
    df_train['date'] = pd.to_datetime(
        df_train['year'].astype(str) + '-' + 
        df_train['month'].astype(str).str.zfill(2) + '-01'
    )
    
    # 2. Zero/Missing flags
    for col in ['value', 'weight', 'quantity']:
        df_train[f'{col}_zero'] = (df_train[col] == 0).astype(int)
        df_train[f'{col}_missing'] = df_train[col].isna().astype(int)
    
    # 3. 월별 집계
    df_monthly = (
        df_train.groupby(['item_id', 'hs4', 'date'], as_index=False)
        [['value', 'weight', 'quantity']].sum()
    )
    
    # 4. Panel completion
    df_panel = create_panel(df_monthly)
    
    # 5. HS4 매핑 보완 (Panel에서 NaN 처리)
    df_panel = fill_hs4_mapping(df_panel)
    
    # 6. Type 정보 추가
    df_panel = add_type_features(df_panel, df_train)
    
    # 7. 공급망 정보 추가
    df_panel = add_supply_chain_features(df_panel)
    
    # 8. 이상치 처리
    df_panel = detect_and_handle_outliers(df_panel)
    
    # 9. 스케일링
    df_panel = add_scaled_features(df_panel)
    
    # 10. 시계열 특성
    df_panel = add_time_series_features(df_panel)
    
    # 11. Quantity 특성
    df_panel = add_quantity_features(df_panel)
    
    # 12. HS2 매핑
    df_panel['hs2'] = df_panel['hs4'].astype(str).str[:2]
    df_panel['hs2_name_kr'] = df_panel['hs2'].map(hs2_map)
    
    # 13. 검증
    validate_preprocessing(df_panel)
    
    return df_panel
```

---

## 🎯 우선순위별 적용 권장사항

### Phase 1 (즉시 적용 권장)
1. ✅ HS4 매핑 보완 (Panel NaN 처리)
2. ✅ 공급망 정보 추가
3. ✅ 시계열 특성 추출

### Phase 2 (성능 개선 시)
4. ✅ 이상치 처리
5. ✅ 스케일링 및 로그 변환
6. ✅ Quantity 활용 개선

### Phase 3 (고급 개선)
7. ✅ Type 컬럼 활용
8. ✅ Panel Completion 개선
9. ✅ 데이터 품질 검증

---

## 💡 추가 고려사항

1. **메모리 최적화**: Feature가 많아지면 메모리 사용량 증가 → 필요한 feature만 선택
2. **계산 시간**: 시계열 특성 계산은 시간이 걸릴 수 있음 → 병렬 처리 고려
3. **모델 호환성**: 추가한 feature가 실제 모델에서 사용되는지 확인 필요

