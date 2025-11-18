# preprocessing.ipynb 기준 공행성 쌍 예측 모델 상세 가이드

## 📋 개요

이 모델은 `preprocessing.ipynb`에서 생성한 전처리 결과를 사용하여 공행성 쌍을 예측합니다.

**핵심 철학**: Panel completion을 통한 완전한 시계열 구조 + 시계열 특성 + 공급망 정보

---

## 🏗️ 아키텍처

```
train.csv
    ↓
[전처리: preprocessing.ipynb 방식]
    ↓
df_panel (Panel completion)
    ↓
[시계열 특성 추가]
    - value_ma6 (6개월 이동평균)
    - value_pct_change (변화율)
    ↓
[공급망 정보 추가]
    - supply_chain_tier (Tier 1/2/3)
    ↓
Pivot 테이블 생성
    ↓
[공행성쌍 탐색]
    - Lag 1~6 상관계수 계산
    - 공급망 방향성 점수 적용
    ↓
[학습 데이터 생성]
    - 9개 feature
    ↓
[회귀 모델 학습]
    ↓
[예측 및 제출]
```

---

## 📊 데이터 흐름

### 1. 전처리 단계

#### 입력 데이터
```python
train.csv
- item_id, year, month, seq, type, hs4
- weight, quantity, value
```

#### 전처리 과정
1. **date 컬럼 생성**
   ```python
   date = pd.to_datetime(year + "-" + month + "-01")
   ```

2. **월별 집계**
   ```python
   df_train_monthly = train.groupby(['item_id', 'hs4', 'date'])[
       'value', 'weight', 'quantity'
   ].sum()
   ```

3. **Panel Completion**
   ```python
   # 모든 item_id × 모든 month 조합 생성
   panel = pd.MultiIndex.from_product([all_items, full_months])
   df_panel = panel.merge(df_train_monthly, how='left')
   # 거래 없는 월은 0으로 채움
   ```

4. **HS4 매핑 보완**
   ```python
   # item_id는 항상 같은 HS4를 가짐 (1:1 관계)
   item_hs4_mapping = df_panel.groupby('item_id')['hs4'].first().to_dict()
   df_panel['hs4'] = df_panel['item_id'].map(item_hs4_mapping)
   ```

5. **공급망 정보 추가**
   ```python
   def get_supply_chain_tier(hs4):
       if 2800 <= hs4 <= 3899: return 1  # Tier 1 (원자재)
       elif 7200 <= hs4 <= 8399: return 2  # Tier 2 (중간재)
       elif 8400 <= hs4 <= 8599: return 3  # Tier 3 (최종제품)
       else: return 0
   ```

6. **시계열 특성 추가**
   ```python
   # 6개월 이동평균
   df_panel['value_ma6'] = df_panel.groupby('item_id')['value'].transform(
       lambda x: x.rolling(6, min_periods=1).mean()
   )
   
   # 변화율
   df_panel['value_pct_change'] = df_panel.groupby('item_id')['value'].pct_change()
   ```

### 2. Pivot 테이블 생성

```python
pivot = df_panel.pivot_table(
    index='item_id',
    columns='date',
    values='value',
    aggfunc='sum',
    fill_value=0.0
)
```

**결과**: `(item_id 수) × (월 수)` 매트릭스

### 3. 공행성쌍 탐색

#### 알고리즘
```python
for leader in items:
    for follower in items:
        if leader == follower: continue
        
        best_corr = 0.0
        best_lag = None
        
        # Lag 1~6 탐색
        for lag in range(1, 7):
            corr = safe_corr(leader[:-lag], follower[lag:])
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        # 공급망 방향성 점수 계산
        supply_chain_score = get_supply_chain_direction(
            leader_hs4, follower_hs4
        )
        
        # 임계값 이상이면 채택
        if abs(best_corr) >= 0.4:
            pairs.append({
                'leading_item_id': leader,
                'following_item_id': follower,
                'best_lag': best_lag,
                'max_corr': best_corr,
                'supply_chain_score': supply_chain_score,
                'same_hs4': 1 if leader_hs4 == follower_hs4 else 0
            })
```

#### 공급망 방향성 점수
| 상황 | 점수 | 설명 |
|------|------|------|
| 같은 HS4 | 1.5 | 같은 품목 그룹 |
| Tier 1 → Tier 3 | 2.0 | 정상 공급망 방향 |
| 같은 Tier | 1.0 | 같은 계층 |
| 역방향 | 0.5 | 비정상 방향 |

### 4. 학습 데이터 생성

#### Feature 구성 (9개)

1. **기본 시계열 (3개)**
   - `b_t`: follower의 현재 시점 값
   - `b_t_1`: follower의 이전 시점 값
   - `a_t_lag`: leader의 lag 시점 값

2. **공행성 정보 (2개)**
   - `max_corr`: 최대 상관계수
   - `best_lag`: 최적 lag 값

3. **공급망 정보 (2개)**
   - `supply_chain_score`: 공급망 방향성 점수
   - `same_hs4`: 같은 HS4 여부 (0/1)

4. **시계열 특성 (2개)**
   - `follower_ma6`: follower의 6개월 이동평균
   - `follower_pct_change`: follower의 변화율

#### Target
- `target`: `b_t_plus_1` (follower의 다음 시점 값)

### 5. 모델 학습

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(train_X, train_y)
```

**모델 타입**: 선형 회귀 (Linear Regression)

### 6. 예측

```python
# 마지막 시점 (t_last)에서 예측
b_t = b_series[t_last]
b_t_1 = b_series[t_prev]
a_t_lag = a_series[t_last - lag]

# 시계열 특성
follower_ma6 = df_panel에서 추출
follower_pct_change = df_panel에서 추출

# 예측
y_pred = reg.predict([b_t, b_t_1, a_t_lag, corr, lag,
                     supply_chain_score, same_hs4,
                     follower_ma6, follower_pct_change])
```

---

## 🎯 핵심 특징

### 1. Panel Completion의 장점

- **완전한 시계열 구조**: 모든 item_id가 모든 month에 존재
- **일관된 분석**: 시계열 길이가 동일하여 비교 용이
- **0값의 의미**: 거래 없음 vs 정보 미기입 구분 가능

### 2. 시계열 특성의 활용

- **이동평균 (MA6)**: 단기 추세 파악
- **변화율 (pct_change)**: 급증/급감 패턴 탐지

### 3. 공급망 정보의 활용

- **방향성 점수**: 공급망 흐름 반영
- **같은 HS4**: 같은 품목 그룹 내 연관성

---

## 📈 성능 최적화 팁

### 1. 하이퍼파라미터 조정

```python
# 공행성쌍 탐색 파라미터
max_lag = 6  # 최대 lag (1~6)
min_nonzero = 12  # 최소 거래 개월 수
corr_threshold = 0.4  # 상관계수 임계값
```

**조정 가이드**:
- `corr_threshold` 낮추기 → 더 많은 쌍 탐색 (노이즈 증가 가능)
- `min_nonzero` 높이기 → 더 안정적인 쌍만 선택 (쌍 수 감소)

### 2. Feature 선택

현재 9개 feature 모두 사용하지만, 필요시 feature selection 가능:

```python
# Feature 중요도 확인
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': reg.coef_,
    'abs_coefficient': np.abs(reg.coef_)
}).sort_values('abs_coefficient', ascending=False)
```

### 3. 모델 개선

- **다항 회귀**: 비선형 관계 모델링
- **Ridge/Lasso**: 정규화로 과적합 방지
- **Random Forest**: 비선형 관계 및 상호작용 학습

---

## 🔍 결과 분석

### 공행성쌍 통계

```python
# 공행성쌍 수
print(f"총 공행성쌍 수: {len(pairs)}")

# Lag 분포
print(pairs['best_lag'].value_counts().sort_index())

# 상관계수 분포
print(f"평균 상관계수: {pairs['max_corr'].abs().mean():.3f}")
print(f"최대 상관계수: {pairs['max_corr'].abs().max():.3f}")

# 공급망 점수 분포
print(pairs['supply_chain_score'].value_counts().sort_index())
```

### 예측값 통계

```python
# 예측값 분포
print(f"평균 예측값: {submission['value'].mean():,.0f}")
print(f"중앙값: {submission['value'].median():,.0f}")
print(f"최대값: {submission['value'].max():,.0f}")
print(f"최소값: {submission['value'].min():,.0f}")

# 0값 비율
zero_ratio = (submission['value'] == 0).sum() / len(submission)
print(f"0값 비율: {zero_ratio:.2%}")
```

---

## ⚠️ 주의사항

1. **Panel Completion의 메모리 사용**
   - 모든 item_id × 모든 month 조합으로 데이터 크기 증가
   - 대용량 데이터의 경우 메모리 부족 가능

2. **0값 처리**
   - 거래 없음과 정보 미기입을 구분하지 않음
   - 0값이 많은 경우 모델 성능 저하 가능

3. **이상치 영향**
   - 이상치 처리가 없어 극단값이 모델에 영향
   - 로그 변환 없어 스케일 차이 문제 가능

---

## 🔄 개선 방향

1. **이상치 처리 추가**
   - 99 분위수 기준 클리핑
   - 극단값 플래그 추가

2. **로그 변환 추가**
   - 스케일 안정화
   - 0값 처리 (log1p)

3. **결측치 보정**
   - 계층적 보정 (item_id → hs4 → 전체)
   - 보정 버전 별도 컬럼 생성

---

## 📚 참고 자료

- `preprocessing.ipynb`: 전처리 상세 과정
- `preprocessing_improvements.md`: 전처리 개선 제안
- `MODEL_COMPARISON.md`: 두 모델 비교

