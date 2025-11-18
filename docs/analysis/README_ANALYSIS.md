# Analysis 디렉토리

## 📁 구조

```
analysis/
├── preprocessing.ipynb                   # 기존 전처리 파이프라인
├── preprocess_ev_battery.py            # EV 배터리 전처리 (Python)
├── preprocess_ev_battery.ipynb         # EV 배터리 전처리 (Notebook)
├── monthly_features.csv                 # 전처리 결과 (ev_battery 모델용)
├── preprocessing_improvements.md       # 전처리 개선 제안서
├── analyze_hs4.py                      # HS4 분석 스크립트
├── analyze_hs4.ipynb                   # HS4 분석 노트북
├── analyze_hs4_item_clustering.py     # HS4-Item 클러스터링 분석
├── analyze_hs4_item_clustering.ipynb  # HS4-Item 클러스터링 분석 노트북
├── complete_analysis.ipynb             # 전체 분석 노트북
├── hs4_supply_chain_ratio.py           # HS4 공급망 비율 분석
├── hs4_supply_chain_ratio.ipynb        # HS4 공급망 비율 분석 노트북
├── show_tier_info.py                   # Tier 정보 표시
├── show_tier_info.ipynb                # Tier 정보 표시 노트북
└── archive/                             # 오래된 분석 파일들
    ├── organize_files.py
    └── organize_files.ipynb
```

## 🚀 주요 파일

### 전처리

- **preprocessing.ipynb**: 기존 전처리 파이프라인 (Panel completion)
- **preprocess_ev_battery.py/ipynb**: EV 배터리 특화 전처리
  - 결측치 보정
  - 이상치 처리
  - 클러스터 라벨링
  - 결과: `monthly_features.csv`

### 분석

- **analyze_hs4.py/ipynb**: HS4 코드 분석
- **analyze_hs4_item_clustering.py/ipynb**: HS4-Item 클러스터링 분석
- **complete_analysis.ipynb**: 전체 분석 통합

## 📊 전처리 결과

### monthly_features.csv

EV 배터리 모델에서 사용하는 전처리 결과 파일입니다.

**주요 컬럼**:
- `item_id`, `hs4`, `hs2`, `year`, `month`, `ym`
- `cluster`: battery_materials, ev_machinery_electrical, other_industries
- `value_sum`, `weight_sum`, `quantity_sum`: 원본 합계
- `value_filled`, `weight_filled`: 보정된 값
- `value_clip`, `weight_clip`: 클리핑된 값
- `log_value`, `log_weight`, `log_quantity`: 로그 변환 값
- 기타 플래그 컬럼들

## 📝 사용 방법

### EV 배터리 전처리 실행

```bash
cd trade-comovement/analysis
python preprocess_ev_battery.py
# 또는
jupyter notebook preprocess_ev_battery.ipynb
```

**결과**: `monthly_features.csv` 생성

### 분석 실행

```bash
cd trade-comovement/analysis
python analyze_hs4.py
# 또는
jupyter notebook analyze_hs4.ipynb
```

