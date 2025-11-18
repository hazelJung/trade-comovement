# 공행성 쌍 예측 모델 비교 가이드

이 문서는 두 가지 전처리 방식에 기반한 공행성 쌍 예측 모델을 비교하고 설명합니다.

---

## 📋 목차

1. [모델 개요](#모델-개요)
2. [preprocessing.ipynb 기준 모델](#1-preprocessingipynb-기준-모델)
3. [preprocess_ev_battery.ipynb 기준 모델](#2-preprocess_ev_batteryipynb-기준-모델)
4. [모델 비교](#모델-비교)
5. [실행 방법](#실행-방법)
6. [결과 해석](#결과-해석)

---

## 모델 개요

### 두 가지 접근 방식

1. **preprocessing.ipynb 기준 모델**
   - 기존 전처리 파이프라인 활용
   - Panel completion 방식
   - 시계열 특성 및 공급망 정보 활용

2. **preprocess_ev_battery.ipynb 기준 모델**
   - EV 배터리 도메인 특화 전처리
   - 결측치 보정 및 이상치 처리
   - 클러스터 기반 가중치 적용

---

## 1. preprocessing.ipynb 기준 모델

### 📁 파일 위치
- **Python 스크립트**: `models/train_model_preprocessing.py`
- **Jupyter 노트북**: `models/train_model_preprocessing.ipynb`
- **제출 파일**: `results/preprocessing_submit.csv`

### 🔧 전처리 특징

#### 데이터 구조
- **입력**: `train.csv` (원본 데이터)
- **전처리 결과**: `df_panel` (Panel completion)
  - 모든 `item_id × 모든 month` 조합 완성
  - 거래 없는 월은 0으로 채움
  - HS4 매핑 보완

#### 주요 전처리 단계
1. **월별 집계**: `(item_id, hs4, date)` 기준으로 `value`, `weight`, `quantity` 합산
2. **Panel Completion**: 모든 item_id와 모든 month의 조합 생성
3. **HS4 매핑 보완**: item_id별 고유 HS4 매핑 (1:1 관계)
4. **공급망 정보 추가**: Tier 1/2/3 분류
5. **시계열 특성 추가**:
   - `value_ma6`: 6개월 이동평균
   - `value_pct_change`: 전월 대비 변화율

### 🎯 공행성쌍 탐색 방식

#### 기본 알고리즘
```python
# Lag 1~6 탐색
for lag in range(1, max_lag + 1):
    corr = safe_corr(leader[:-lag], follower[lag:])
    # 최고 상관계수 선택
```

#### 공급망 정보 활용
- **공급망 방향성 점수**:
  - 같은 HS4: 1.5점
  - 정상 방향 (Tier 1 → Tier 3): 2.0점
  - 같은 계층: 1.0점
  - 역방향: 0.5점

#### 임계값
- `corr_threshold = 0.4`
- `min_nonzero = 12` (최소 12개월 이상 거래 존재)

### 📊 학습 데이터 Feature (9개)

| Feature | 설명 |
|---------|------|
| `b_t` | follower의 현재 시점 값 |
| `b_t_1` | follower의 이전 시점 값 |
| `a_t_lag` | leader의 lag 시점 값 |
| `max_corr` | 최대 상관계수 |
| `best_lag` | 최적 lag 값 |
| `supply_chain_score` | 공급망 방향성 점수 |
| `same_hs4` | 같은 HS4 여부 (0/1) |
| `follower_ma6` | follower의 6개월 이동평균 |
| `follower_pct_change` | follower의 변화율 |

### 💡 장점
- ✅ 시계열 특성으로 추세 파악 가능
- ✅ 공급망 정보로 도메인 지식 반영
- ✅ Panel completion으로 완전한 시계열 구조 확보

### ⚠️ 한계
- ⚠️ 0값을 그대로 사용 (결측치 보정 없음)
- ⚠️ 이상치 처리 없음
- ⚠️ 스케일 차이 문제 (로그 변환 없음)

---

## 2. preprocess_ev_battery.ipynb 기준 모델

### 📁 파일 위치
- **Python 스크립트**: `models/train_model_ev_battery.py`
- **Jupyter 노트북**: `models/train_model_ev_battery.ipynb`
- **제출 파일**: `results/ev_battery_submit.csv`

### 🔧 전처리 특징

#### 데이터 구조
- **입력**: `train.csv` (원본 데이터)
- **전처리 결과**: `monthly_features.csv`
  - `(item_id, year, month)` 단위 집계
  - 결측치 보정 및 이상치 처리 완료
  - 로그 변환 및 클러스터 라벨링 완료

#### 주요 전처리 단계
1. **월별 집계**: `(item_id, year, month)` 기준 집계
2. **EV 배터리 클러스터 라벨링**:
   - `battery_materials`: 배터리 소재·화학·금속
   - `ev_machinery_electrical`: EV·설비·전기기기
   - `other_industries`: 기타 산업
3. **Quantity 구조적 결측 처리**: HS4별 quantity 사용 여부 플래그
4. **0값 기반 결측 플래그**: 사실상 결측치 탐지
5. **계층적 결측치 보정**:
   - item_id 중앙값 → hs4 중앙값 → 전체 중앙값
6. **이상치 처리**: 99 분위수 기준 클리핑
7. **로그 변환**: log1p 변환으로 스케일 안정화

### 🎯 공행성쌍 탐색 방식

#### 클러스터 가중치 적용
```python
def get_cluster_boost(leader_cluster, follower_cluster):
    # 같은 클러스터 내
    if leader_cluster == follower_cluster:
        if 'battery_materials': return 2.0
        elif 'ev_machinery_electrical': return 1.8
        else: return 1.2
    
    # 배터리 소재 → EV 기기
    if leader == 'battery_materials' and follower == 'ev_machinery_electrical':
        return 1.5
    
    return 1.0
```

#### 가중치 적용 방식
- **effective_corr = abs(corr) × cluster_boost**
- **effective_threshold = corr_threshold / cluster_boost**
- 클러스터 가중치가 높을수록 낮은 상관계수도 채택 가능

#### 임계값
- 기본 `corr_threshold = 0.4`
- 클러스터 가중치에 따라 동적 조정

### 📊 학습 데이터 Feature (11개)

| Feature | 설명 |
|---------|------|
| `b_t` | follower의 현재 시점 값 (value_filled) |
| `b_t_1` | follower의 이전 시점 값 |
| `a_t_lag` | leader의 lag 시점 값 |
| `max_corr` | 최대 상관계수 |
| `best_lag` | 최적 lag 값 |
| `cluster_boost` | 클러스터 가중치 |
| `same_cluster` | 같은 클러스터 여부 (0/1) |
| `same_hs4` | 같은 HS4 여부 (0/1) |
| `follower_log_value` | follower의 로그 변환 값 |
| `follower_value_clip` | follower의 클리핑된 값 |
| `follower_has_quantity` | follower의 quantity 존재 여부 |

### 💡 장점
- ✅ 결측치 보정으로 데이터 품질 향상
- ✅ 이상치 처리로 노이즈 제거
- ✅ 로그 변환으로 스케일 안정화
- ✅ EV 배터리 도메인 특화 클러스터링
- ✅ Quantity 구조적 결측 정보 활용

### ⚠️ 한계
- ⚠️ Panel completion 없음 (거래 없는 월 정보 손실)
- ⚠️ 시계열 특성(이동평균 등) 제한적

---

## 모델 비교

### 📊 전처리 방식 비교

| 항목 | preprocessing.ipynb | preprocess_ev_battery.ipynb |
|------|---------------------|------------------------------|
| **집계 단위** | (item_id, hs4, date) | (item_id, year, month) |
| **Panel Completion** | ✅ 있음 | ❌ 없음 |
| **결측치 보정** | ❌ 없음 | ✅ 계층적 보정 |
| **이상치 처리** | ❌ 없음 | ✅ 99 분위수 클리핑 |
| **로그 변환** | ❌ 없음 | ✅ log1p 변환 |
| **클러스터링** | 공급망 Tier | EV 배터리 클러스터 |
| **시계열 특성** | ✅ 이동평균, 변화율 | ❌ 제한적 |

### 🎯 공행성쌍 탐색 비교

| 항목 | preprocessing 모델 | ev_battery 모델 |
|------|-------------------|-----------------|
| **기본 알고리즘** | Lag 1~6 상관계수 | Lag 1~6 상관계수 |
| **가중치 적용** | 공급망 방향성 점수 | 클러스터 가중치 |
| **임계값** | 고정 0.4 | 동적 조정 (클러스터별) |
| **도메인 지식** | 공급망 계층 | EV 배터리 클러스터 |

### 📈 Feature 비교

| Feature | preprocessing | ev_battery |
|---------|--------------|------------|
| 기본 시계열 | ✅ b_t, b_t_1, a_t_lag | ✅ b_t, b_t_1, a_t_lag |
| 공행성 정보 | ✅ max_corr, best_lag | ✅ max_corr, best_lag |
| 도메인 정보 | ✅ supply_chain_score, same_hs4 | ✅ cluster_boost, same_cluster, same_hs4 |
| 시계열 특성 | ✅ follower_ma6, follower_pct_change | ❌ |
| 전처리 특성 | ❌ | ✅ follower_log_value, follower_value_clip, follower_has_quantity |
| **총 Feature 수** | **9개** | **11개** |

### 🔍 예상 성능 차이

#### preprocessing 모델이 유리한 경우
- 시계열 패턴이 중요한 경우
- 추세 및 계절성이 중요한 경우
- Panel completion이 필요한 경우

#### ev_battery 모델이 유리한 경우
- 데이터 품질 문제(결측치, 이상치)가 많은 경우
- 스케일 차이가 큰 경우
- EV 배터리 도메인 특화 예측이 필요한 경우
- 클러스터 내 연관성이 높은 경우

---

## 실행 방법

### 1. preprocessing.ipynb 기준 모델

#### 사전 준비
```bash
# preprocessing.ipynb 실행 (선택사항)
# 또는 train_model_preprocessing.py가 자체적으로 전처리 수행
```

#### 실행
```bash
cd trade-comovement/models

# Python 스크립트 실행
python train_model_preprocessing.py

# 또는 Jupyter 노트북 실행
jupyter notebook train_model_preprocessing.ipynb
```

#### 결과 파일
- `results/preprocessing_submit.csv`

### 2. preprocess_ev_battery.ipynb 기준 모델

#### 사전 준비 (필수)
```bash
cd trade-comovement/analysis

# 전처리 실행
python preprocess_ev_battery.py
# 또는
jupyter notebook preprocess_ev_battery.ipynb

# monthly_features.csv 생성 확인
ls monthly_features.csv
```

#### 실행
```bash
cd ../models

# Python 스크립트 실행
python train_model_ev_battery.py

# 또는 Jupyter 노트북 실행
jupyter notebook train_model_ev_battery.ipynb
```

#### 결과 파일
- `results/ev_battery_submit.csv`

---

## 결과 해석

### 제출 파일 형식

두 모델 모두 동일한 형식의 제출 파일을 생성합니다:

```csv
leading_item_id,following_item_id,value
ITEM001,ITEM002,12345
ITEM003,ITEM004,67890
...
```

### 성능 비교 방법

1. **공행성쌍 수 비교**
   - 각 모델이 탐색한 공행성쌍의 개수
   - 더 많은 쌍 = 더 넓은 커버리지

2. **클러스터/공급망 분포**
   - 어떤 클러스터/계층에서 더 많은 쌍을 찾았는지
   - 도메인 특화 성능 지표

3. **예측값 분포**
   - 예측값의 통계적 특성
   - 극단값 여부

### 모델 선택 가이드

#### preprocessing 모델을 선택하는 경우
- ✅ 시계열 패턴 분석이 중요한 경우
- ✅ 공급망 계층 구조가 명확한 경우
- ✅ Panel completion이 필요한 경우
- ✅ 이동평균, 변화율 등 시계열 특성이 유용한 경우

#### ev_battery 모델을 선택하는 경우
- ✅ 데이터 품질 문제가 있는 경우
- ✅ EV 배터리 도메인 특화 예측이 필요한 경우
- ✅ 클러스터 내 연관성이 높은 경우
- ✅ 스케일 차이가 큰 경우
- ✅ 결측치/이상치 처리가 중요한 경우

### 앙상블 가능성

두 모델의 결과를 앙상블하여 성능을 향상시킬 수 있습니다:

```python
# 가중 평균 앙상블 예시
submission_preprocessing = pd.read_csv('results/preprocessing_submit.csv')
submission_ev_battery = pd.read_csv('results/ev_battery_submit.csv')

# 공통 쌍에 대해 가중 평균
merged = submission_preprocessing.merge(
    submission_ev_battery,
    on=['leading_item_id', 'following_item_id'],
    suffixes=('_preprocessing', '_ev_battery')
)

merged['value'] = (
    0.5 * merged['value_preprocessing'] + 
    0.5 * merged['value_ev_battery']
).round().astype(int)
```

---

## 📝 참고사항

### 데이터 의존성

- **preprocessing 모델**: 원본 `train.csv`만 있으면 실행 가능
- **ev_battery 모델**: `monthly_features.csv` 필요 (전처리 선행 필수)

### 실행 시간

- **preprocessing 모델**: 중간 (Panel completion 포함)
- **ev_battery 모델**: 빠름 (전처리 결과 사용)

### 메모리 사용량

- **preprocessing 모델**: 높음 (Panel completion으로 데이터 증가)
- **ev_battery 모델**: 낮음 (집계된 데이터만 사용)

---

## 🔄 업데이트 이력

- 2024-11-16: 초기 버전 작성
  - preprocessing.ipynb 기준 모델
  - preprocess_ev_battery.ipynb 기준 모델

---

## 📚 관련 문서

- `preprocessing.ipynb`: 기존 전처리 파이프라인
- `preprocess_ev_battery.ipynb`: EV 배터리 전처리 파이프라인
- `preprocessing_improvements.md`: 전처리 개선 제안서

