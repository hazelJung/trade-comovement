# 평가 지표 고려 개선 체크리스트

## ✅ 완료된 개선

### 1. max_pairs 튜닝으로 Precision 최적화 ✅

**구현 위치**:
- `train_model_preprocessing.py`: `MAX_PAIRS` 변수 추가
- `train_model_preprocessing.ipynb`: Cell 7 수정
- `train_model_ev_battery.py`: 주석 추가 (선택사항)

**사용 방법**:
```python
MAX_PAIRS = 1500  # 종합 점수 상위 1500개만 선택
pairs = find_comovement_pairs(..., max_pairs=MAX_PAIRS)
```

**효과**:
- 종합 점수로 정렬 후 상위 N개만 선택
- FP 감소 → Precision 향상
- F1 Score (60%) 개선 기여

### 2. NMAE 손실 함수 고려 ✅

**구현 위치**:
- `train_model_preprocessing.py`: `USE_QUANTILE` 옵션 추가
- `train_model_preprocessing.ipynb`: Cell 11 수정
- `train_model_ev_battery.py`: `USE_QUANTILE` 옵션 추가
- `train_model_ev_battery.ipynb`: Cell 11 수정

**사용 방법**:
```python
USE_QUANTILE = True  # Quantile Regression 사용
RIDGE_ALPHA = 1.0    # 정규화 강도

if USE_QUANTILE:
    reg = QuantileRegressor(quantile=0.5, alpha=RIDGE_ALPHA, solver='highs')
else:
    reg = Ridge(alpha=RIDGE_ALPHA)
```

**효과**:
- Quantile Regression은 중앙값 예측 → 상대 오차에 덜 민감
- NMAE (40%) 개선 기여

**참고**:
- Quantile Regression은 sklearn 1.0+ 필요
- solver='highs'는 scipy 최신 버전 필요

### 3. Cross-Validation으로 하이퍼파라미터 튜닝 ✅

**구현 위치**:
- `hyperparameter_tuning.py`: 새로 생성

**사용 방법**:
```python
from hyperparameter_tuning import time_series_cv_tuning

best_params, results = time_series_cv_tuning(pivot, df_panel, item_hs4)
```

**튜닝 대상**:
- `max_pairs`: None, 1200, 1500, 1800
- `min_abs_corr`: 0.25, 0.30, 0.35
- `ridge_alpha`: 0.1, 1.0, 10.0

**효과**:
- Time Series Split으로 객관적 평가
- 최적 하이퍼파라미터 자동 탐색
- F1 Score + NMAE 종합 최적화

## 📝 하이퍼파라미터 튜닝 가이드

### 빠른 튜닝 (수동)

```python
# preprocessing 모델
MAX_PAIRS = 1500  # 1200, 1500, 1800 중 선택
MIN_ABS_CORR = 0.30  # 0.25, 0.30, 0.35 중 선택
RIDGE_ALPHA = 1.0  # 0.1, 1.0, 10.0 중 선택

# ev_battery 모델
MIN_ABS_CORR = 0.25  # 0.25, 0.30 중 선택
RIDGE_ALPHA = 1.0  # 0.1, 1.0, 10.0 중 선택
```

### 자동 튜닝 (Cross-Validation)

```python
# hyperparameter_tuning.py 실행
python hyperparameter_tuning.py

# 결과 확인
# ../results/hyperparameter_tuning_results.csv
```

## 🎯 평가 지표 최적화 현황

| 개선 사항 | 상태 | 평가 지표 반영 |
|----------|------|----------------|
| 1. max_pairs 튜닝 | ✅ 완료 | F1 Score (Precision) |
| 2. NMAE 손실 함수 | ✅ 완료 | NMAE (예측 정확도) |
| 3. Cross-Validation | ✅ 완료 | 종합 최적화 |

## 💡 사용 권장사항

### Precision이 중요한 경우
```python
MAX_PAIRS = 1200  # 적은 쌍 → 높은 Precision
MIN_ABS_CORR = 0.35  # 엄격한 필터링
```

### Recall이 중요한 경우
```python
MAX_PAIRS = None  # 모든 쌍 사용
MIN_ABS_CORR = 0.25  # 완화된 필터링
```

### 균형잡힌 경우
```python
MAX_PAIRS = 1500  # 중간값
MIN_ABS_CORR = 0.30  # 중간값
```

## 📊 다음 단계

1. **실제 평가**: 정답 파일과 비교하여 최적 하이퍼파라미터 확인
2. **그리드 서치**: 더 많은 조합 테스트
3. **앙상블**: 여러 하이퍼파라미터 조합의 예측 결합

---

**업데이트**: 2024-11-16
**상태**: 모든 개선 사항 구현 완료 ✅

