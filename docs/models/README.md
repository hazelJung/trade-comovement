# Models 디렉토리

## 📁 구조

```
models/
├── train_model_preprocessing.py          # preprocessing.ipynb 기준 모델 (Python)
├── train_model_preprocessing.ipynb      # preprocessing.ipynb 기준 모델 (Notebook)
├── train_model_ev_battery.py            # preprocess_ev_battery.ipynb 기준 모델 (Python)
├── train_model_ev_battery.ipynb        # preprocess_ev_battery.ipynb 기준 모델 (Notebook)
├── MODEL_COMPARISON.md                  # 두 모델 비교 가이드
├── MODEL_PREPROCESSING.md               # preprocessing 모델 상세 가이드
├── MODEL_EV_BATTERY.md                  # ev_battery 모델 상세 가이드
├── PAIR_COUNT_ANALYSIS.md               # 공행성 쌍 개수 차이 분석
├── IMPROVEMENTS_SUMMARY.md              # 개선 사항 요약
├── PERFORMANCE_REPORT.md                # 성능 비교 리포트
└── archive/                              # 오래된 모델 파일들
    ├── train_model.py
    ├── train_model.ipynb
    ├── train_model_supply_chain.py
    └── train_model_supply_chain.ipynb
```

## 🚀 빠른 시작

### preprocessing 모델 실행

```bash
cd trade-comovement/models
python train_model_preprocessing.py
# 또는
jupyter notebook train_model_preprocessing.ipynb
```

**결과**: `../results/preprocessing_submit.csv`

### ev_battery 모델 실행

```bash
# 먼저 전처리 실행
cd trade-comovement/analysis
python preprocess_ev_battery.py

# 그 다음 모델 실행
cd ../models
python train_model_ev_battery.py
# 또는
jupyter notebook train_model_ev_battery.ipynb
```

**결과**: `../results/ev_battery_submit.csv`

## 📊 모델 비교

자세한 비교는 `MODEL_COMPARISON.md`를 참고하세요.

### 주요 차이점

| 항목 | preprocessing | ev_battery |
|------|--------------|------------|
| **쌍 개수** | 1,425개 | 2,604개 |
| **전처리** | Panel completion | 결측치 보정 + 이상치 처리 |
| **특징** | 시계열 특성 | 클러스터 정보 |
| **모델** | Ridge (alpha=1.0) | Ridge (alpha=1.0) |

## 📈 최근 성능

자세한 성능 비교는 `PERFORMANCE_REPORT.md`를 참고하세요.

- **공통 쌍**: 1,425개
- **예측값 상관계수**: 0.9951 (매우 높음)
- **평균 예측값 차이**: 472,944

## 📚 문서

- `MODEL_COMPARISON.md`: 두 모델의 상세 비교
- `MODEL_PREPROCESSING.md`: preprocessing 모델 가이드
- `MODEL_EV_BATTERY.md`: ev_battery 모델 가이드
- `PAIR_COUNT_ANALYSIS.md`: 공행성 쌍 개수 차이 분석
- `IMPROVEMENTS_SUMMARY.md`: 개선 사항 요약
- `PERFORMANCE_REPORT.md`: 성능 비교 리포트

