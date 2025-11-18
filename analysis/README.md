# Analysis 디렉토리

이 디렉토리는 데이터 분석 및 전처리 스크립트를 포함합니다.

## 📁 구조

```
analysis/
├── data/                   # 전처리 결과 데이터
│   └── monthly_features.csv
├── archive/                # 아카이브된 파일
├── *.ipynb                 # 분석 노트북
└── *.py                    # 분석 스크립트
```

**문서**: 모든 분석 관련 문서는 `docs/analysis/` 디렉토리에 있습니다.

## 📝 주요 파일

### 전처리
- `preprocessing.ipynb`: 기본 전처리
- `preprocess_ev_battery.ipynb` / `.py`: EV 배터리 전처리
- `data/monthly_features.csv`: 전처리 결과

### 분석
- `analyze_hs4.ipynb` / `.py`: HS4 분석
- `analyze_hs4_item_clustering.ipynb` / `.py`: 클러스터링 분석
- `complete_analysis.ipynb`: 종합 분석

### 문서
모든 분석 관련 문서는 `docs/analysis/` 디렉토리에 있습니다:
- `docs/analysis/preprocessing_improvements.md`: 전처리 개선사항
- `docs/analysis/hs4_analysis_report.md`: HS4 코드 분석 리포트
- `docs/analysis/hs4_item_analysis_summary.md`: HS4-Item 분석 요약

## 🚀 사용법

1. 전처리 실행
   ```bash
   # Jupyter Notebook
   jupyter notebook preprocess_ev_battery.ipynb
   
   # Python 스크립트
   python preprocess_ev_battery.py
   ```

2. 결과 확인
   - 전처리 결과: `data/monthly_features.csv`
   - 분석 결과: `results/analysis/`

