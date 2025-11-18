# preprocess_ev_battery.ipynb 기준 공행성 쌍 예측 모델 상세 가이드

## 📋 개요

이 모델은 `preprocess_ev_battery.ipynb`에서 생성한 전처리 결과를 사용하여 공행성 쌍을 예측합니다.

**핵심 철학**: 데이터 품질 향상 + EV 배터리 도메인 특화 + 클러스터 기반 가중치

---

## 🏗️ 아키텍처

```
train.csv
    ↓
[전처리: preprocess_ev_battery.ipynb]
    ↓
monthly_features.csv
    - 결측치 보정 (value_filled, weight_filled)
    - 이상치 처리 (value_clip, weight_clip)
    - 로그 변환 (log_value, log_weight)
    - 클러스터 라벨링 (battery_materials, ev_machinery_electrical, other_industries)
    ↓
Pivot 테이블 생성 (value_filled 사용)
    ↓
[공행성쌍 탐색]
    - Lag 1~6 상관계수 계산
    - 클러스터 가중치 적용
    ↓
[학습 데이터 생성]
    - 11개 feature
    ↓
[회귀 모델 학습]
    ↓
[예측 및 제출]
```

---

## 📊 데이터 흐름

### 1. 전처리 단계 (preprocess_ev_battery.ipynb)

#### 입력 데이터
```python
train.csv
- item_id, year, month, seq, type, hs4
- weight, quantity, value
```

#### 전처리 과정

1. **기본 로드 & ym 컬럼 생성**
   ```python
   ym = pd.to_datetime(year + "-" + month + "-01")
   ```

2. **item_id-hs4 매핑 확보**
   ```python
   item_hs4_map = train[['item_id', 'hs4']].drop_duplicates()
   ```

3. **월별 집계**
   ```python
   monthly = train.groupby(['item_id', 'year', 'month']).agg({
       'value': 'sum',
       'weight': 'sum',
       'quantity': 'sum'
   })
   ```

4. **EV 배터리 클러스터 라벨링**
   ```python
   def assign_cluster(hs2):
       if hs2 in ["27","28","29","31","38","39","72","79","81"]:
           return "battery_materials"
       elif hs2 in ["84","85","87","90"]:
           return "ev_machinery_electrical"
       else:
           return "other_industries"
   ```

5. **Quantity 구조적 결측 처리**
   ```python
   # HS4별로 quantity > 0인 row가 있는지 확인
   hs4_has_quantity = monthly.groupby('hs4')['quantity_sum'].apply(
       lambda x: (x > 0).any()
   )
   monthly['has_quantity_info_hs4'] = monthly['hs4'].map(hs4_has_quantity)
   monthly['has_quantity_this_row'] = (monthly['quantity_sum'] > 0).astype(int)
   ```

6. **0값 기반 결측 플래그**
   ```python
   weight_zero_flag = (weight_sum == 0)
   value_zero_flag = (value_sum == 0)
   weight_missing_flag = (weight_sum == 0) & (value_sum > 0)
   value_missing_flag = (value_sum == 0) & (weight_sum > 0)
   ```

7. **계층적 결측치 보정**
   ```python
   # weight_filled: weight_missing_flag == 1인 경우
   # 1순위: item_id 중앙값
   # 2순위: hs4 중앙값
   # 3순위: 전체 중앙값
   
   # value_filled: value_missing_flag == 1인 경우
   # 동일한 계층적 보정
   ```

8. **이상치 처리 (99 분위수 클리핑)**
   ```python
   q99_value = value_filled.quantile(0.99)
   value_extreme_flag = (value_filled > q99_value)
   value_clip = np.minimum(value_filled, q99_value)
   ```

9. **로그 변환**
   ```python
   log_value = np.log1p(value_clip)
   log_weight = np.log1p(weight_clip)
   log_quantity = np.log1p(quantity_clip)
   ```

### 2. Pivot 테이블 생성

```python
pivot = monthly_features.pivot_table(
    index='item_id',
    columns='ym',
    values='value_filled',  # 보정된 값 사용
    aggfunc='sum',
    fill_value=0.0
)
```

**특징**: `value_filled` 사용 (결측치 보정된 값, 이상치 처리 전)

### 3. 공행성쌍 탐색

#### 클러스터 가중치 함수

```python
def get_cluster_boost(leader_cluster, follower_cluster):
    # 같은 클러스터 내
    if leader_cluster == follower_cluster:
        if leader_cluster == 'battery_materials':
            return 2.0  # 배터리 소재 간 높은 연관성
        elif leader_cluster == 'ev_machinery_electrical':
            return 1.8  # EV 기기 간 연관성
        else:
            return 1.2  # 기타 산업 간 연관성
    
    # 배터리 소재 → EV 기기 (공급망 관계)
    if leader_cluster == 'battery_materials' and \
       follower_cluster == 'ev_machinery_electrical':
        return 1.5
    
    return 1.0  # 기본값
```

#### 가중치 적용 알고리즘

```python
# 기본 상관계수 계산
corr = safe_corr(leader[:-lag], follower[lag:])

# 클러스터 가중치 적용
cluster_boost = get_cluster_boost(leader_cluster, follower_cluster)
effective_corr = abs(corr) * cluster_boost
effective_threshold = corr_threshold / cluster_boost

# 가중치가 높을수록 낮은 상관계수도 채택 가능
if effective_corr >= corr_threshold:
    # 공행성쌍으로 채택
```

#### 클러스터별 가중치 요약

| 상황 | 가중치 | 효과 |
|------|--------|------|
| battery_materials 내부 | 2.0 | 상관계수 0.2 이상도 채택 가능 |
| ev_machinery_electrical 내부 | 1.8 | 상관계수 0.22 이상도 채택 가능 |
| battery_materials → ev_machinery_electrical | 1.5 | 공급망 관계 반영 |
| other_industries 내부 | 1.2 | 약간의 가중치 |
| 기타 | 1.0 | 기본값 |

### 4. 학습 데이터 생성

#### Feature 구성 (11개)

1. **기본 시계열 (3개)**
   - `b_t`: follower의 현재 시점 값 (value_filled)
   - `b_t_1`: follower의 이전 시점 값
   - `a_t_lag`: leader의 lag 시점 값

2. **공행성 정보 (2개)**
   - `max_corr`: 최대 상관계수
   - `best_lag`: 최적 lag 값

3. **클러스터 정보 (3개)**
   - `cluster_boost`: 클러스터 가중치
   - `same_cluster`: 같은 클러스터 여부 (0/1)
   - `same_hs4`: 같은 HS4 여부 (0/1)

4. **전처리 특성 (3개)**
   - `follower_log_value`: follower의 로그 변환 값
   - `follower_value_clip`: follower의 클리핑된 값
   - `follower_has_quantity`: follower의 quantity 존재 여부

#### Target
- `target`: `b_t_plus_1` (follower의 다음 시점 값)

### 5. 모델 학습

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(train_X, train_y)
```

### 6. 예측

```python
# 마지막 시점에서 예측
b_t = b_series[t_last]
b_t_1 = b_series[t_prev]
a_t_lag = a_series[t_last - lag]

# 전처리 특성
follower_log_value = monthly_features에서 추출
follower_value_clip = monthly_features에서 추출
follower_has_quantity = monthly_features에서 추출

# 예측
y_pred = reg.predict([b_t, b_t_1, a_t_lag, corr, lag,
                     cluster_boost, same_cluster, same_hs4,
                     follower_log_value, follower_value_clip,
                     follower_has_quantity])
```

---

## 🎯 핵심 특징

### 1. 데이터 품질 향상

#### 결측치 보정
- **계층적 보정**: item_id → hs4 → 전체 중앙값
- **원본 보존**: `value_sum`은 유지, `value_filled` 별도 생성
- **도메인 지식 반영**: 같은 품목/그룹의 중앙값 사용

#### 이상치 처리
- **99 분위수 기준**: 극단값만 클리핑
- **플래그 생성**: `value_extreme_flag`로 이상치 추적
- **원본 보존**: `value_filled` 유지, `value_clip` 별도 생성

#### 로그 변환
- **log1p 사용**: 0값 처리
- **스케일 안정화**: 수백만 ~ 수억 범위를 안정적으로 변환

### 2. EV 배터리 도메인 특화

#### 클러스터 라벨링
- **battery_materials**: 배터리 소재·화학·금속 (HS2: 27,28,29,31,38,39,72,79,81)
- **ev_machinery_electrical**: EV·설비·전기기기 (HS2: 84,85,87,90)
- **other_industries**: 기타 산업

#### 클러스터 가중치
- 같은 클러스터 내에서 높은 가중치
- 배터리 소재 → EV 기기 공급망 관계 반영

### 3. Quantity 구조적 결측 처리

- **HS4별 quantity 사용 여부**: 어떤 HS4는 quantity가 항상 0
- **플래그 생성**: `has_quantity_info_hs4`, `has_quantity_this_row`
- **Feature로 활용**: quantity 정보의 의미 여부를 모델에 제공

---

## 📈 성능 최적화 팁

### 1. 클러스터 가중치 조정

```python
def get_cluster_boost(leader_cluster, follower_cluster):
    # 가중치를 조정하여 성능 개선 가능
    if leader_cluster == follower_cluster:
        if leader_cluster == 'battery_materials':
            return 2.0  # 조정 가능: 1.5 ~ 2.5
        elif leader_cluster == 'ev_machinery_electrical':
            return 1.8  # 조정 가능: 1.3 ~ 2.0
```

**조정 가이드**:
- 가중치 높이기 → 더 많은 쌍 탐색 (노이즈 증가 가능)
- 가중치 낮추기 → 더 엄격한 쌍만 선택 (쌍 수 감소)

### 2. 이상치 처리 임계값 조정

```python
# 99 분위수 대신 95 분위수 사용
q95_value = value_filled.quantile(0.95)
value_clip = np.minimum(value_filled, q95_value)
```

**효과**: 더 많은 값을 클리핑하여 극단값 영향 감소

### 3. Feature 선택

필요시 feature selection으로 중요 feature만 사용:

```python
# Feature 중요도 확인
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': reg.coef_,
    'abs_coefficient': np.abs(reg.coef_)
}).sort_values('abs_coefficient', ascending=False)

# 상위 8개 feature만 선택
top_features = feature_importance.head(8)['feature'].tolist()
```

---

## 🔍 결과 분석

### 클러스터별 공행성쌍 분포

```python
# 클러스터별 쌍 수
cluster_pairs = pairs.groupby(['leader_cluster', 'follower_cluster']).size()
print(cluster_pairs)

# 예상 결과:
# battery_materials × battery_materials: 가장 많음 (가중치 2.0)
# ev_machinery_electrical × ev_machinery_electrical: 중간
# battery_materials × ev_machinery_electrical: 공급망 관계
```

### 가중치 효과 분석

```python
# 가중치별 쌍 수
print(pairs['cluster_boost'].value_counts().sort_index())

# 가중치별 평균 상관계수
print(pairs.groupby('cluster_boost')['max_corr'].abs().mean())

# 가중치가 높을수록 평균 상관계수가 낮을 수 있음
# (낮은 상관계수도 채택되기 때문)
```

### 전처리 효과 분석

```python
# 보정 전후 비교
print(f"보정된 행 수: {(monthly_features['value_filled'] != monthly_features['value_sum']).sum()}")

# 이상치 비율
print(f"이상치 비율: {monthly_features['value_extreme_flag'].mean():.2%}")

# 클리핑 효과
print(f"클리핑된 행 수: {(monthly_features['value_clip'] != monthly_features['value_filled']).sum()}")
```

---

## ⚠️ 주의사항

1. **전처리 파일 의존성**
   - `monthly_features.csv` 파일이 반드시 필요
   - 전처리를 먼저 실행해야 함

2. **Panel Completion 없음**
   - 거래 없는 월의 정보 손실
   - 시계열 길이가 item_id마다 다를 수 있음

3. **클러스터 가중치의 주관성**
   - 가중치 값이 도메인 지식에 의존
   - 데이터에 맞게 조정 필요

---

## 🔄 개선 방향

1. **Panel Completion 추가**
   - 거래 없는 월도 포함하여 완전한 시계열 구조 확보

2. **시계열 특성 추가**
   - 이동평균, 변화율 등 추가
   - 계절성 특성 추가

3. **앙상블 모델**
   - 여러 모델의 예측 결합
   - 가중 평균 또는 스태킹

---

## 📚 참고 자료

- `preprocess_ev_battery.ipynb`: 전처리 상세 과정
- `preprocess_ev_battery.py`: 전처리 Python 스크립트
- `MODEL_COMPARISON.md`: 두 모델 비교
- `MODEL_PREPROCESSING.md`: preprocessing 모델 상세 가이드

