# Models 디렉토리

이 디렉토리는 공행성 쌍 예측 모델을 포함합니다.

## 📁 구조

```
models/
├── archive/                 # 아카이브된 모델
├── *.ipynb                 # 모델 학습 노트북
├── *.py                    # 모델 학습 스크립트
└── hyperparameter_tuning.py
```

**문서**: 모든 모델 관련 문서는 `docs/models/` 디렉토리에 있습니다.

## 📝 주요 파일

### 모델 코드
- `train_model_preprocessing.py` / `.ipynb`: preprocessing 기반 모델
- `train_model_ev_battery.py` / `.ipynb`: EV 배터리 전처리 기반 모델
- `hyperparameter_tuning.py`: 하이퍼파라미터 튜닝 스크립트

### 문서
모든 모델 관련 문서는 `docs/models/` 디렉토리에 있습니다:
- `docs/models/MODEL_PREPROCESSING.md`: preprocessing 모델 설명
- `docs/models/MODEL_EV_BATTERY.md`: EV 배터리 모델 설명
- `docs/models/MODEL_COMPARISON.md`: 두 모델 비교
- `docs/models/IMPROVEMENTS_*.md`: 개선사항 문서
- `docs/models/PERFORMANCE_REPORT.md`: 성능 비교 리포트
- `docs/models/EXECUTION_SUMMARY.md`: 실행 결과 요약

## 🚀 사용법

1. 전처리 결과 준비
   - `analysis/data/monthly_features.csv` (EV 배터리 모델)
   - 또는 `preprocessing.ipynb` 실행 (preprocessing 모델)

2. 모델 실행
   ```bash
   python train_model_preprocessing.py
   python train_model_ev_battery.py
   ```

3. 결과 확인
   - 제출 파일: `results/submissions/`

