# 무역량 예측 AI 모델 프로젝트

## 📁 프로젝트 구조

```
openlaboratory/
├── data/                    # 데이터 파일
│   ├── train.csv           # 학습 데이터
│   └── sample_submission.csv # 제출 샘플
│
├── models/                  # 모델 코드
│   ├── train_model.py      # Baseline 모델
│   ├── train_model_supply_chain.py  # 공급망 기반 모델
│   └── train_model.ipynb   # Jupyter Notebook
│
├── analysis/                # 분석 스크립트
│   ├── analyze_hs4.py      # HS4 코드 분석
│   ├── analyze_hs4_item_clustering.py  # 군집 분석
│   ├── hs4_supply_chain_ratio.py  # 공급망 비중 분석
│   └── show_tier_info.py   # Tier 정보 표시
│
├── results/                 # 결과 파일
│   ├── baseline_submit.csv  # Baseline 예측 결과
│   ├── supply_chain_submit.csv  # 공급망 모델 예측 결과
│   ├── hs4_supply_chain_ratio.csv  # HS4 공급망 비중
│   ├── hs4_supply_chain_tier_summary.csv  # 계층별 요약
│   └── hs4_clustering_results.csv  # 클러스터링 결과
│
├── reports/                 # 분석 보고서
│   ├── hs4_analysis_report.md  # HS4 분석 보고서
│   ├── hs4_item_analysis_summary.md  # 품목 분석 요약
│   ├── supply_chain_model_summary.md  # 공급망 모델 요약
│   ├── tier_explanation.md  # Tier 설명
│   ├── notion_prompt_final.md  # 노션 정리 프롬프트
│   └── notion_organization_prompt.md  # 노션 정리 프롬프트 (상세)
│
├── images/                  # 그래프 및 시각화
│   ├── cluster_distribution.png
│   ├── hs4_clustering.png
│   ├── hs4_quantity_usage.png
│   ├── hs4_value_weight_analysis.png
│   ├── item_id_coverage_analysis.png
│   └── item_id_trade_volume_analysis.png
│
├── config/                  # 설정 파일
│   ├── requirements.txt    # 패키지 목록
│   └── setup_env.md        # 가상환경 설정 가이드
│
└── venv/                    # 가상환경 (생성됨)
```

## 🚀 빠른 시작

### 1. 가상환경 설정
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r config/requirements.txt
```

### 2. 모델 실행

#### Baseline 모델
```bash
python models/train_model.py
```

#### 공급망 기반 모델
```bash
python models/train_model_supply_chain.py
```

### 3. 분석 실행
```bash
# HS4 코드 분석
python analysis/analyze_hs4.py

# 군집 분석
python analysis/analyze_hs4_item_clustering.py

# 공급망 비중 분석
python analysis/hs4_supply_chain_ratio.py
```

## 📊 주요 결과

### 모델 성과
- **Baseline 모델**: 1,425개 공행성쌍 발견
- **공급망 모델**: 3,085개 공행성쌍 발견 (116% 증가)

### 데이터 분석
- **기계산업 관련**: 전체 무역량의 85.8%
- **Tier 1 (원자재)**: 67.43% 비중
- **Tier 3 (최종제품)**: 14.89% 비중

## 📝 주요 파일 설명

### 모델 파일
- `train_model.py`: 기본 공행성 분석 모델
- `train_model_supply_chain.py`: 공급망 구조를 고려한 개선 모델

### 분석 파일
- `analyze_hs4.py`: HS4 코드 기계산업 공급망 연관성 분석
- `analyze_hs4_item_clustering.py`: HS4/item_id 군집 분석 및 K-means 클러스터링
- `hs4_supply_chain_ratio.py`: HS4 코드별 공급망 비중 분석

### 보고서
- `hs4_analysis_report.md`: HS4 코드 분석 상세 보고서
- `supply_chain_model_summary.md`: 공급망 모델 요약
- `tier_explanation.md`: 공급망 계층 구조 설명

## 🔍 공급망 계층 구조

- **Tier 1 (원자재)**: 화학제품 (HS4: 2800-3899) - 67.43%
- **Tier 2 (중간재)**: 금속제품 (HS4: 7200-8399) - 3.37%
- **Tier 3 (최종제품)**: 기계류 (HS4: 8400-9299) - 14.89%

## 📈 생성된 결과물

### 예측 결과
- `results/baseline_submit.csv`: Baseline 모델 예측
- `results/supply_chain_submit.csv`: 공급망 모델 예측

### 분석 결과
- `results/hs4_supply_chain_ratio.csv`: HS4별 상세 통계
- `results/hs4_clustering_results.csv`: K-means 클러스터링 결과

### 시각화
- `images/`: 모든 분석 그래프 (6개 PNG 파일)

## 📚 참고 문서

- `config/setup_env.md`: 가상환경 설정 가이드
- `reports/notion_prompt_final.md`: 노션 정리용 프롬프트

## 🛠️ 개발 환경

- Python 3.12
- 주요 라이브러리: pandas, numpy, scikit-learn, matplotlib, seaborn
- Jupyter Notebook 지원

## 📌 주요 발견사항

1. **도메인 지식 활용의 중요성**: 공급망 구조를 고려하여 공행성쌍 116% 증가
2. **기계산업 중심 데이터**: 전체의 85.8%가 기계산업 공급망 관련
3. **명확한 계층 구조**: 원자재 → 중간재 → 최종제품의 3단계 구조

