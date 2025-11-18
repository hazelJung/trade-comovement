import os, shutil, re
import sys
try:
    import nbformat as nbf
except Exception:
    print('nbformat not found, please install via pip install nbformat', file=sys.stderr)
    raise

NB_PATH = 'preprocessing.ipynb'
BACKUP_PATH = 'preprocessing.ipynb.bak'

if not os.path.exists(NB_PATH):
    raise SystemExit(f'Notebook not found: {NB_PATH}')

# Backup
if not os.path.exists(BACKUP_PATH):
    shutil.copyfile(NB_PATH, BACKUP_PATH)

nb = nbf.read(NB_PATH, as_version=4)

KEYWORDS = [
    (r"^\s*%?matplotlib|plt\.", "시각화/플롯 생성"),
    (r"^\s*import |^\s*from ", "라이브러리 임포트"),
    (r"read_csv\(|read_excel\(|read_parquet\(", "데이터 로드"),
    (r"to_csv\(|to_parquet\(|to_excel\(", "데이터 저장"),
    (r"isnull\(|isna\(|fillna\(|dropna\(", "결측치 확인/처리"),
    (r"duplicated\(|drop_duplicates\(", "중복 데이터 처리"),
    (r"astype\(|to_datetime\(|strptime", "타입 변환/날짜 처리"),
    (r"groupby\(|pivot_table\(|agg\(", "집계/그룹 연산"),
    (r"merge\(|join\(|concat\(", "데이터 결합/병합"),
    (r"rename\(|assign\(", "컬럼 이름 변경/추가"),
    (r"sort_values\(|sort_index\(", "정렬"),
    (r"query\(|loc\[|iloc\[", "행/열 선택 및 필터링"),
    (r"StandardScaler\(|MinMaxScaler\(", "스케일링/정규화"),
    (r"train_test_split\(|KFold\(|StratifiedKFold\(", "데이터 분할/검증 설정"),
    (r"fit\(|predict\(|score\(", "모델 학습/예측"),
    (r"roc_auc|precision|recall|f1|confusion_matrix|classification_report", "모델 평가"),
    (r"seaborn|sns\.", "시각화(Seaborn)"),
    (r"np\.|numpy|pd\.|pandas", "수치/데이터프레임 처리"),
]

GENERIC_TITLE = "코드 셀 동작 설명"

def summarize_code(code: str) -> str:
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    head = ' '.join(lines[:3])[:200]

    matched = []
    for pattern, label in KEYWORDS:
        if re.search(pattern, code, flags=re.MULTILINE):
            matched.append(label)
    matched = list(dict.fromkeys(matched))  # dedupe, preserve order

    if not matched:
        matched_desc = "일반 전처리/로직 수행"
    else:
        matched_desc = ", ".join(matched)

    bullets = []
    if '데이터 로드' in matched:
        bullets.append("- 데이터 파일을 읽어 DataFrame으로 로드")
    if '결측치 확인/처리' in matched:
        bullets.append("- 결측치 상태를 확인하고 적절히 제거/대체")
    if '중복 데이터 처리' in matched:
        bullets.append("- 중복 행을 탐지하여 제거")
    if '타입 변환/날짜 처리' in matched:
        bullets.append("- 컬럼 타입 변환 또는 날짜 파싱 수행")
    if '집계/그룹 연산' in matched:
        bullets.append("- 그룹 기준으로 집계/통계 계산")
    if '데이터 결합/병합' in matched:
        bullets.append("- 키를 기준으로 데이터 병합 또는 이어붙이기")
    if '스케일링/정규화' in matched:
        bullets.append("- 학습 전 특징 스케일링/정규화")
    if '데이터 분할/검증 설정' in matched:
        bullets.append("- 학습/검증 데이터 분할 및 교차검증 설정")
    if '모델 학습/예측' in matched:
        bullets.append("- 모델 학습을 수행하고 예측/점수 산출")
    if '모델 평가' in matched:
        bullets.append("- 다양한 평가지표로 성능 평가")
    if '시각화/플롯 생성' in matched or '시각화(Seaborn)' in matched:
        bullets.append("- 그래프/차트로 결과 시각화")

    bullet_text = "\n".join(bullets) if bullets else "- 주요 전처리/계산/출력 수행"

    preview = head if head else code.strip()[:120]
    if preview:
        preview = preview.replace('\n', ' ')[:200]

    md = f"""**{GENERIC_TITLE}**\n\n- **주요 범주**: {matched_desc}\n{bullet_text}\n\n> 코드 미리보기: `{preview}`"""
    return md

new_cells = []
for cell in nb.cells:
    if cell.get('cell_type') == 'code':
        desc = summarize_code(cell.get('source', ''))
        md_cell = nbf.v4.new_markdown_cell(desc)
        new_cells.append(md_cell)
    new_cells.append(cell)

nb.cells = new_cells

nbf.write(nb, NB_PATH)
print('Annotated notebook saved. Backup at', BACKUP_PATH)
