import os
import shutil
from pathlib import Path

print("="*70)
print("파일 정리 스크립트")
print("="*70)

# 현재 디렉토리
base_dir = Path('.')

# 정리할 폴더 구조
folders = {
    'data': ['train.csv', 'sample_submission.csv'],
    'models': ['train_model.py', 'train_model_supply_chain.py', 'train_model.ipynb'],
    'analysis': ['analyze_hs4.py', 'analyze_hs4_item_clustering.py', 'hs4_supply_chain_ratio.py', 'show_tier_info.py'],
    'results': ['baseline_submit.csv', 'supply_chain_submit.csv', 'hs4_supply_chain_ratio.csv', 
                'hs4_supply_chain_tier_summary.csv', 'hs4_clustering_results.csv'],
    'reports': ['hs4_analysis_report.md', 'hs4_item_analysis_summary.md', 'supply_chain_model_summary.md', 
                'tier_explanation.md', 'notion_prompt_final.md', 'notion_organization_prompt.md'],
    'images': ['*.png'],
    'config': ['requirements.txt', 'setup_env.md']
}

# 폴더 생성
for folder_name in folders.keys():
    folder_path = base_dir / folder_name
    folder_path.mkdir(exist_ok=True)
    print(f"폴더 생성/확인: {folder_name}/")

# 파일 이동
moved_files = []
skipped_files = []

for folder_name, file_patterns in folders.items():
    folder_path = base_dir / folder_name
    
    for pattern in file_patterns:
        if '*' in pattern:
            # 패턴 매칭 (예: *.png)
            import glob
            files = glob.glob(pattern)
            for file in files:
                src = Path(file)
                if src.exists() and src.is_file():
                    dst = folder_path / src.name
                    if not dst.exists():
                        shutil.move(str(src), str(dst))
                        moved_files.append(f"{src.name} -> {folder_name}/")
                        print(f"이동: {src.name} -> {folder_name}/")
                    else:
                        skipped_files.append(f"{src.name} (이미 존재)")
        else:
            # 단일 파일
            src = base_dir / pattern
            if src.exists() and src.is_file():
                dst = folder_path / src.name
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                    moved_files.append(f"{src.name} -> {folder_name}/")
                    print(f"이동: {src.name} -> {folder_name}/")
                else:
                    skipped_files.append(f"{src.name} (이미 존재)")

print("\n" + "="*70)
print("정리 완료!")
print("="*70)
print(f"\n이동된 파일: {len(moved_files)}개")
print(f"건너뛴 파일: {len(skipped_files)}개")

if moved_files:
    print("\n이동된 파일 목록:")
    for file in moved_files:
        print(f"  - {file}")

if skipped_files:
    print("\n건너뛴 파일 목록:")
    for file in skipped_files:
        print(f"  - {file}")

# 남은 파일 확인
print("\n" + "="*70)
print("남은 파일 확인")
print("="*70)
remaining_files = [f for f in base_dir.iterdir() 
                  if f.is_file() and f.name not in ['organize_files.py', '.gitignore'] 
                  and f.suffix not in ['.pyc', '.pyo']]
if remaining_files:
    print("\n정리되지 않은 파일:")
    for file in remaining_files:
        print(f"  - {file.name}")
else:
    print("\n모든 파일이 정리되었습니다!")

