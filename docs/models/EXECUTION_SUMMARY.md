# 개선사항 적용 모델 실행 결과 요약

## 실행 일시
2024-11-16

## 실행 결과

### 1. preprocessing 모델 (`train_model_preprocessing.py`)

**공행성쌍 탐색**:
- 탐색된 공행성쌍 수: **1,425개**
- 종합 점수 기반 선택 (max_pairs=None, 모든 쌍 사용)

**학습 데이터**:
- 생성된 학습 데이터: **54,743행**
- Feature 수: 9개
  - `b_t`, `b_t_1`, `a_t_lag`, `max_corr`, `best_lag`
  - `supply_chain_score`, `same_hs4`
  - `follower_ma6`, `follower_pct_change`

**모델 학습**:
- 모델: **Ridge Regression** (alpha=1.0)
- Feature Scaling: **StandardScaler** 적용
- Quantile Regression: 사용 안 함 (USE_QUANTILE=False)

**예측 결과**:
- 예측된 공행성쌍 수: **1,425개**
- 제출 파일: `../results/preprocessing_submit.csv`

---

### 2. ev_battery 모델 (`train_model_ev_battery.py`)

**공행성쌍 탐색**:
- 탐색된 공행성쌍 수: **2,604개**
- 클러스터 가중치 적용
- 클러스터별 분포:
  - battery_materials → battery_materials: 1,111개
  - battery_materials → ev_machinery_electrical: 312개
  - battery_materials → other_industries: 232개
  - ev_machinery_electrical → battery_materials: 124개
  - ev_machinery_electrical → ev_machinery_electrical: 99개
  - ev_machinery_electrical → other_industries: 61개
  - other_industries → battery_materials: 278개
  - other_industries → ev_machinery_electrical: 58개
  - other_industries → other_industries: 329개

**학습 데이터**:
- 생성된 학습 데이터: **100,125행**
- Feature 수: 11개
  - `b_t`, `b_t_1`, `a_t_lag`, `max_corr`, `best_lag`
  - `cluster_boost`, `same_cluster`, `same_hs4`
  - `follower_log_value`, `follower_value_clip`, `follower_has_quantity`

**모델 학습**:
- 모델: **Ridge Regression** (alpha=1.0)
- Feature Scaling: **StandardScaler** 적용
- Quantile Regression: 사용 안 함 (USE_QUANTILE=False)

**예측 결과**:
- 예측된 공행성쌍 수: **2,604개**
- 제출 파일: `../results/ev_battery_submit.csv`

---

## 개선사항 적용 확인

### ✅ 적용된 개선사항

1. **max_pairs 튜닝** (preprocessing 모델)
   - 종합 점수로 정렬 후 상위 N개 선택 가능
   - 현재: max_pairs=None (모든 쌍 사용)

2. **NMAE 손실 함수 고려**
   - Quantile Regression 옵션 추가
   - 현재: USE_QUANTILE=False (Ridge 사용)

3. **Ridge 정규화 + Feature Scaling**
   - 두 모델 모두 적용 완료
   - Ridge alpha=1.0
   - StandardScaler로 feature 정규화

4. **이중 필터링**
   - preprocessing: min_abs_corr=0.3 + corr_threshold=0.4
   - ev_battery: min_abs_corr=0.25 + effective_corr >= 0.4

5. **안정성 점수** (preprocessing 모델)
   - 시계열 변동계수(CV) 기반 안정성 계산
   - 종합 점수에 반영 (50% 상관계수 + 30% 안정성 + 20% 공급망)

---

## 모델 비교

| 항목 | preprocessing 모델 | ev_battery 모델 |
|------|-------------------|-----------------|
| 공행성쌍 수 | 1,425 | 2,604 |
| 학습 데이터 행 수 | 54,743 | 100,125 |
| Feature 수 | 9 | 11 |
| 클러스터 정보 | ❌ | ✅ |
| 안정성 점수 | ✅ | ❌ |
| 종합 점수 | ✅ | ❌ |
| 클러스터 가중치 | ❌ | ✅ |

---

## 다음 단계

1. **하이퍼파라미터 튜닝**
   - `hyperparameter_tuning.py` 실행하여 최적 하이퍼파라미터 탐색
   - max_pairs, min_abs_corr, ridge_alpha 튜닝

2. **Quantile Regression 테스트**
   - USE_QUANTILE=True로 설정하여 NMAE 개선 효과 확인

3. **성능 평가**
   - 실제 정답 파일과 비교하여 F1 Score + NMAE 계산
   - 두 모델의 성능 비교

---

**실행 완료**: 모든 개선사항이 적용된 모델이 성공적으로 실행되었습니다! ✅

