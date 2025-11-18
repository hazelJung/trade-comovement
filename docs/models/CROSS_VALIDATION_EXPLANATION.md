# Cross-Validation (교차 검증) 설명

## 📚 Cross-Validation이란?

**Cross-Validation (교차 검증)**은 모델의 성능을 평가하고 하이퍼파라미터를 튜닝하는 방법입니다.

### 기본 개념

데이터를 여러 개의 "fold"로 나누고, 각 fold를 번갈아가며 검증 데이터로 사용하는 방법입니다.

```
전체 데이터: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

K-Fold Cross-Validation (K=5):
Fold 1: 학습 [2,3,4,5,6,7,8,9,10] → 검증 [1]
Fold 2: 학습 [1,3,4,5,6,7,8,9,10] → 검증 [2]
Fold 3: 학습 [1,2,4,5,6,7,8,9,10] → 검증 [3]
Fold 4: 학습 [1,2,3,5,6,7,8,9,10] → 검증 [4]
Fold 5: 학습 [1,2,3,4,6,7,8,9,10] → 검증 [5]
```

### 왜 사용하나요?

1. **과적합 방지**: 한 번의 train/test split만 사용하면 운이 좋은/나쁜 split에 의존할 수 있음
2. **신뢰성 향상**: 여러 번 평가하여 평균 성능을 계산
3. **하이퍼파라미터 튜닝**: 어떤 하이퍼파라미터가 가장 좋은지 객관적으로 비교

### 우리 프로젝트에서의 활용

```python
# 예시: max_pairs를 1000, 1500, 2000으로 테스트
for max_pairs in [1000, 1500, 2000]:
    scores = []
    for fold in range(5):  # 5-Fold CV
        # 학습 데이터로 모델 학습
        # 검증 데이터로 F1 Score + NMAE 계산
        score = calculate_score(...)
        scores.append(score)
    
    avg_score = np.mean(scores)
    print(f"max_pairs={max_pairs}: 평균 점수={avg_score}")
```

### Time Series Cross-Validation

우리 프로젝트는 시계열 데이터이므로, **Time Series Split**을 사용해야 합니다:

```python
# 일반 K-Fold (X) - 미래 데이터로 과거를 예측하는 것은 안됨
# Time Series Split (O) - 과거 데이터로 미래를 예측

Train: [2022-01 ~ 2023-12] → Test: [2024-01]
Train: [2022-01 ~ 2024-01] → Test: [2024-02]
Train: [2022-01 ~ 2024-02] → Test: [2024-03]
...
```

## 🎯 우리 프로젝트 적용 예시

### 하이퍼파라미터 튜닝

```python
# 1. max_pairs 튜닝
for max_pairs in [1200, 1500, 1800, 2000]:
    pairs = find_comovement_pairs(..., max_pairs=max_pairs)
    # Time Series CV로 평가
    score = evaluate_with_cv(pairs, ...)
    print(f"max_pairs={max_pairs}: score={score}")

# 2. min_abs_corr 튜닝
for min_abs_corr in [0.25, 0.30, 0.35]:
    pairs = find_comovement_pairs(..., min_abs_corr=min_abs_corr)
    score = evaluate_with_cv(pairs, ...)
    print(f"min_abs_corr={min_abs_corr}: score={score}")

# 3. Ridge alpha 튜닝
for alpha in [0.1, 1.0, 10.0]:
    reg = Ridge(alpha=alpha)
    # 학습 및 평가
    score = evaluate_with_cv(...)
    print(f"alpha={alpha}: score={score}")
```

### 평가 지표 계산

```python
def evaluate_with_cv(pairs, pivot, df_panel, ...):
    """Time Series Cross-Validation으로 평가"""
    months = pivot.columns.to_list()
    n_months = len(months)
    
    scores = []
    
    # 마지막 12개월을 검증용으로 사용 (예시)
    for val_start in range(n_months - 12, n_months - 1):
        train_months = months[:val_start]
        val_month = months[val_start]
        
        # 학습 데이터 생성
        train_data = build_training_data(...)
        
        # 모델 학습
        reg.fit(...)
        
        # 검증 데이터로 예측
        predictions = predict(...)
        
        # 실제 값과 비교하여 F1 + NMAE 계산
        # (정답 파일이 있다면)
        score = calculate_f1_nmae(predictions, ground_truth)
        scores.append(score)
    
    return np.mean(scores)
```

## ⚠️ 주의사항

1. **시계열 특성**: 일반 K-Fold가 아닌 Time Series Split 사용
2. **정답 파일 필요**: CV를 하려면 검증 기간의 정답이 필요
3. **계산 비용**: 여러 fold를 돌리므로 시간이 오래 걸림

## 💡 실전 활용

정답 파일이 없으면:
- **대신 사용**: 학습 데이터의 마지막 몇 개월을 검증용으로 분리
- **또는**: 여러 하이퍼파라미터로 제출 파일 생성 후 실제 평가로 비교

---

**요약**: Cross-Validation은 데이터를 여러 번 나누어 모델을 평가하는 방법입니다. 하이퍼파라미터 튜닝에 유용하지만, 시계열 데이터에서는 Time Series Split을 사용해야 합니다.

