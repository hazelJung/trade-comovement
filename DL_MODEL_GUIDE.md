# 딥러닝 모델 사용 가이드

## 📊 예상 개선 효과

### 현재 상황 분석
- **데이터 규모**: 100개 item, 약 43개월 시계열
- **학습 데이터**: 공행성 쌍 기반 (수백~수천 개)
- **현재 모델**: RandomForest + GBM + ExtraTrees 앙상블

### 딥러닝 모델 예상 개선

| 항목 | 개선 효과 | 설명 |
|------|----------|------|
| **시계열 패턴 학습** | +5-10% | LSTM이 장기 의존성 포착 |
| **장거리 의존성** | +3-7% | 12개월 이상의 lag 관계 학습 |
| **비선형 관계** | +2-5% | 복잡한 상호작용 패턴 발견 |
| **총 예상 개선** | **+10-22%** | 데이터 크기에 따라 다름 |

### ⚠️ 주의사항

1. **오버피팅 위험**
   - 작은 데이터셋에서는 전통적 모델이 더 안정적일 수 있음
   - Early Stopping, Dropout 필수
   - Cross-validation으로 검증

2. **학습 시간**
   - CPU: 5-10분
   - GPU: 1-2분 (선택사항)

3. **하이퍼파라미터 튜닝**
   - Learning rate, Batch size, Epochs 조정 필요
   - 모델 구조 최적화 필요

## 🚀 구현 방법

### 1. 설치

```bash
pip install tensorflow
# 또는 GPU 버전
pip install tensorflow-gpu
```

### 2. 모델 선택

#### 옵션 1: LSTM (추천)
- **장점**: 시계열 특화, 구현 간단
- **단점**: 학습 시간 다소 길음
- **적용**: 시계열 시퀀스 직접 입력

#### 옵션 2: CNN-LSTM
- **장점**: 로컬 패턴 + 장기 의존성
- **단점**: 파라미터 많음
- **적용**: 복잡한 패턴이 있는 경우

#### 옵션 3: Transformer (고급)
- **장점**: Attention 메커니즘으로 중요한 시점 포착
- **단점**: 구현 복잡, 데이터 많아야 효과적
- **적용**: 대규모 데이터셋

### 3. 통합 방법

`improved_model_v2.py`에 다음 함수 추가:

```python
# 1. 딥러닝 모델 정의
def build_lstm_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(32, return_sequences=False, dropout=0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 2. 시퀀스 데이터 생성
def create_sequences(train_df, feature_cols, seq_len=6):
    # Pair별로 시계열 시퀀스 생성
    sequences = []
    targets = []
    # ... 구현
    return np.array(sequences), np.array(targets)

# 3. 학습 및 예측
X_seq, y_seq = create_sequences(train_df, feature_cols)
model = build_lstm_model((seq_len, len(feature_cols)))
model.fit(X_seq, y_seq, epochs=50, validation_split=0.2)
```

### 4. 앙상블 결합

```python
# 전통적 모델 예측
y_pred_traditional = ensemble_predict(X_test)

# 딥러닝 모델 예측
y_pred_dl = dl_model.predict(X_seq_test)

# 가중 평균 (성능 기반)
y_pred_final = 0.7 * y_pred_traditional + 0.3 * y_pred_dl
```

## 📈 실제 개선 효과 측정

### 비교 실험

1. **Baseline**: improved_model_v2.py (현재)
2. **DL Only**: 딥러닝 모델만 사용
3. **Ensemble**: 전통적 + 딥러닝 앙상블

### 평가 지표

- **F1 Score**: 공행성 쌍 정확도 (60% 가중치)
- **NMAE**: 예측 오차 (40% 가중치)
- **Total Score**: 0.6 * F1 + 0.4 * (1 - NMAE)

## 💡 권장 사항

### 데이터가 작은 경우 (< 1000 samples)
- ✅ 전통적 앙상블 모델 우선
- ⚠️ 딥러닝은 간단한 구조만 (LSTM 1-2층)
- ✅ 강한 정규화 (Dropout 0.3-0.5)

### 데이터가 충분한 경우 (> 1000 samples)
- ✅ 딥러닝 모델 적극 활용
- ✅ 복잡한 구조 시도 (CNN-LSTM, Attention)
- ✅ 앙상블로 결합

### 현재 프로젝트 (추정 500-2000 samples)
- ✅ **LSTM 모델 추가** (추천)
- ✅ 전통적 모델과 앙상블 (70:30 또는 60:40)
- ⚠️ 간단한 구조로 시작 (오버피팅 방지)

## 🔧 빠른 시작

1. `improved_model_dl.py` 실행하여 딥러닝 모델 테스트
2. 성능 확인 후 `improved_model_v2.py`에 통합
3. 앙상블 가중치 조정하여 최적 성능 찾기

## 📝 결론

**예상 개선**: +10-22% (현재 점수 0.18 → 0.20-0.22)

**추천**: 
- 먼저 전통적 모델 최적화 (feature engineering, 하이퍼파라미터)
- 그 다음 딥러닝 모델 추가
- 최종적으로 앙상블

**주의**: 데이터가 작으면 딥러닝의 이점이 제한적일 수 있음. 실제 실험으로 검증 필요!


