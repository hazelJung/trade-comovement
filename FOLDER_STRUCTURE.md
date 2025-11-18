# 프로젝트 폴더 구조

## 📁 전체 구조

```
trade-comovement/
├── analysis/              # 데이터 분석 및 전처리
│   ├── data/             # 전처리 결과 데이터
│   │   └── monthly_features.csv
│   ├── archive/          # 아카이브된 파일
│   ├── *.ipynb           # 분석 노트북
│   └── *.py              # 분석 스크립트
│
├── models/               # 모델 학습 및 예측
│   ├── archive/          # 아카이브된 모델
│   ├── *.ipynb           # 모델 학습 노트북
│   ├── *.py              # 모델 학습 스크립트
│   └── hyperparameter_tuning.py
│
├── docs/                  # 통합 문서 디렉토리
│   ├── analysis/         # 분석 관련 문서
│   │   ├── hs4_analysis_report.md
│   │   ├── hs4_item_analysis_summary.md
│   │   ├── preprocessing_improvements.md
│   │   └── README_ANALYSIS.md
│   ├── models/          # 모델 기술 문서
│   │   ├── MODEL_COMPARISON.md
│   │   ├── MODEL_PREPROCESSING.md
│   │   ├── MODEL_EV_BATTERY.md
│   │   ├── IMPROVEMENTS_*.md
│   │   ├── PERFORMANCE_REPORT.md
│   │   ├── EXECUTION_SUMMARY.md
│   │   └── README.md
│   ├── project/         # 프로젝트 전체 문서
│   │   ├── supply_chain_model_summary.md
│   │   ├── tier_explanation.md
│   │   ├── notion_prompt_final.md
│   │   └── notion_organization_prompt.md
│   ├── README.md         # 문서 디렉토리 가이드
│   └── INTEGRATION_SUMMARY.md
│
├── results/              # 결과 파일
│   ├── submissions/      # 제출 파일
│   │   ├── baseline_submit.csv
│   │   ├── ev_battery_submit.csv
│   │   ├── preprocessing_submit.csv
│   │   └── supply_chain_submit.csv
│   └── analysis/         # 분석 결과
│       ├── hs4_anomaly_detail_info.csv
│       ├── hs4_clustering_results.csv
│       ├── hs4_monthly_anomalies.csv
│       ├── hs4_monthly_anomalies_detail_table.csv
│       ├── hs4_supply_chain_ratio.csv
│       ├── hs4_supply_chain_tier_summary.csv
│       └── item_hs4_mapping.csv
│
├── data/                 # 원본 데이터
│   ├── train.csv
│   └── sample_submission.csv
│
├── images/               # 시각화 이미지
│   └── *.png
│
└── config/               # 설정 파일
    ├── requirements.txt
    └── setup_env.md
```

## 📂 폴더별 설명

### analysis/
- **목적**: 데이터 분석 및 전처리
- **data/**: 전처리 결과 데이터 파일
- **파일 종류**: 
  - `.ipynb`: 분석 노트북
  - `.py`: 분석 스크립트

### models/
- **목적**: 모델 학습 및 예측
- **파일 종류**:
  - `.ipynb`: 모델 학습 노트북
  - `.py`: 모델 학습 스크립트
  - `hyperparameter_tuning.py`: 하이퍼파라미터 튜닝 스크립트

### docs/
- **목적**: 통합 문서 디렉토리
- **analysis/**: 분석 관련 문서 (HS4 분석, 전처리 개선사항 등)
- **models/**: 모델 기술 문서 (모델 비교, 성능 리포트, 개선사항 등)
- **project/**: 프로젝트 전체 문서 (공급망 모델 요약, Notion 프롬프트 등)
- **파일 종류**: `.md`

### results/
- **목적**: 모델 및 분석 결과 저장
- **submissions/**: 제출용 예측 결과 파일
- **analysis/**: 분석 결과 CSV 파일
- **파일 종류**: `.csv`

### data/
- **목적**: 원본 데이터 저장
- **파일**: `train.csv`, `sample_submission.csv`

### images/
- **목적**: 시각화 결과 저장
- **파일 종류**: `.png`

### config/
- **목적**: 프로젝트 설정
- **파일**: `requirements.txt`, `setup_env.md`

## 🔍 파일 찾기 가이드

### 모델 문서 찾기
→ `docs/models/`

### 제출 파일 찾기
→ `results/submissions/`

### 전처리 결과 찾기
→ `analysis/data/monthly_features.csv`

### 분석 결과 찾기
→ `results/analysis/`

### 모델 코드 찾기
→ `models/*.py`, `models/*.ipynb`

### 분석 코드 찾기
→ `analysis/*.py`, `analysis/*.ipynb`

