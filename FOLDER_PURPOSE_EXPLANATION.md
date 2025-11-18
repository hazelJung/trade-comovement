# 폴더 구조 목적 설명

## 📁 reports/ vs docs/ 차이점

### reports/ (프로젝트 루트)
**목적**: 프로젝트 전체 분석 결과 및 외부 공유용 문서

**내용**:
- `hs4_analysis_report.md`: HS4 코드 기계산업 공급망 연관성 분석 보고서
- `hs4_item_analysis_summary.md`: HS4-Item 분석 요약
- `supply_chain_model_summary.md`: 공급망 모델 요약 (모델 성능 개선 사항)
- `tier_explanation.md`: 공급망 계층(Tier) 설명
- `notion_prompt_final.md`: Notion 정리용 프롬프트
- `notion_organization_prompt.md`: Notion 조직화 프롬프트

**특징**:
- 📊 **분석 결과 중심**: 데이터 분석 결과와 인사이트
- 📝 **외부 공유용**: Notion 정리, 프레젠테이션용
- 🎯 **프로젝트 전체 관점**: 특정 모듈이 아닌 전체 프로젝트 분석

---

### docs/ (각 모듈 내부)
**목적**: 모듈별 기술 문서 및 개발 가이드

#### models/docs/
**내용**:
- `MODEL_COMPARISON.md`: 두 모델 비교 가이드
- `MODEL_PREPROCESSING.md`: preprocessing 모델 상세 가이드
- `MODEL_EV_BATTERY.md`: ev_battery 모델 상세 가이드
- `IMPROVEMENTS_CHECKLIST.md`: 개선사항 체크리스트
- `IMPROVEMENTS_SUMMARY.md`: 개선사항 요약
- `EXECUTION_SUMMARY.md`: 실행 요약
- `PERFORMANCE_REPORT.md`: 성능 비교 리포트
- `EVALUATION_METRIC_ANALYSIS.md`: 평가 지표 분석
- `CROSS_VALIDATION_EXPLANATION.md`: Cross-Validation 설명

**특징**:
- 🔧 **기술 문서**: 모델 개발, 개선, 실행 방법
- 📚 **개발 가이드**: 코드 사용법, 모델 설명
- 🎓 **학습 자료**: 개념 설명, 평가 지표 설명

#### analysis/docs/
**내용**:
- `preprocessing_improvements.md`: 전처리 개선사항
- `README.md`: 분석 디렉토리 가이드

**특징**:
- 🔧 **전처리 기술 문서**: 전처리 방법론, 개선사항

---

## 🤔 왜 분리했는가?

### 현재 구조의 장점
1. **명확한 역할 분리**
   - `reports/`: 비즈니스/분석 관점 (What, Why)
   - `docs/`: 기술/개발 관점 (How)

2. **접근성**
   - 분석 결과는 프로젝트 루트에서 바로 접근
   - 기술 문서는 해당 모듈 내에서 관리

3. **유지보수**
   - 모듈별 문서는 해당 모듈과 함께 관리
   - 프로젝트 전체 문서는 별도 관리

### 현재 구조의 단점
1. **중복 가능성**
   - `reports/supply_chain_model_summary.md`와 `models/docs/PERFORMANCE_REPORT.md`가 유사한 내용 포함 가능

2. **찾기 어려움**
   - 문서가 여러 곳에 분산되어 있어 찾기 어려울 수 있음

3. **일관성 부족**
   - 폴더 구조가 일관되지 않음 (reports는 루트, docs는 모듈 내부)

---

## 💡 개선 제안

### 옵션 1: 통합 (권장)
```
docs/
├── analysis/          # 분석 결과 및 전처리 문서
│   ├── hs4_analysis_report.md
│   ├── preprocessing_improvements.md
│   └── ...
├── models/            # 모델 기술 문서
│   ├── MODEL_COMPARISON.md
│   ├── PERFORMANCE_REPORT.md
│   └── ...
└── project/           # 프로젝트 전체 문서
    ├── notion_prompt_final.md
    └── tier_explanation.md
```

### 옵션 2: 현재 구조 유지 + README 추가
- 각 폴더에 README.md 추가하여 목적 명시
- 문서 간 링크 추가

### 옵션 3: reports를 docs로 통합
```
docs/
├── reports/          # 분석 결과
├── models/           # 모델 문서
└── analysis/         # 전처리 문서
```

---

## 📋 현재 파일 분류

### reports/ → 분석 결과 및 외부 공유용
- ✅ `hs4_analysis_report.md` - 분석 결과
- ✅ `hs4_item_analysis_summary.md` - 분석 요약
- ✅ `supply_chain_model_summary.md` - 모델 요약 (분석 관점)
- ✅ `tier_explanation.md` - 개념 설명
- ✅ `notion_prompt_final.md` - 외부 공유용
- ✅ `notion_organization_prompt.md` - 외부 공유용

### models/docs/ → 모델 기술 문서
- ✅ `MODEL_COMPARISON.md` - 기술 가이드
- ✅ `MODEL_PREPROCESSING.md` - 기술 가이드
- ✅ `MODEL_EV_BATTERY.md` - 기술 가이드
- ✅ `IMPROVEMENTS_*.md` - 개발 문서
- ✅ `PERFORMANCE_REPORT.md` - 성능 리포트 (기술 관점)
- ✅ `EXECUTION_SUMMARY.md` - 실행 문서

### analysis/docs/ → 전처리 기술 문서
- ✅ `preprocessing_improvements.md` - 기술 문서

---

## 🎯 결론

**현재 구조는 역할 분리가 명확하지만, 통합하는 것이 더 나을 수 있습니다.**

**추천**: 옵션 1 (docs/로 통합) 또는 현재 구조 유지 + 각 폴더에 README 추가

