import json
import pathlib
import shutil

path = pathlib.Path('preprocessing.ipynb')
backup = path.with_name(path.stem + '_cleanup_backup.ipynb')
shutil.copy2(path, backup)
with path.open('r', encoding='utf-8') as f:
    data = json.load(f)
orig_count = len(data.get('cells', []))
filtered = []
removed_hs2 = 0
removed_baseline = 0
for cell in data.get('cells', []):
    src = ''.join(cell.get('source', []))
    if 'df_panel["hs2"] = df_panel["hs4"].astype("string").str[:2]' in src and 'fillna("UN")' not in src:
        removed_hs2 += 1
        continue
    if 'SimpleItemMeanModel' in src:
        removed_baseline += 1
        continue
    filtered.append(cell)

data['cells'] = filtered
with path.open('w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=1)

print(f"Removed hs2 cells: {removed_hs2}")
print(f"Removed baseline cells: {removed_baseline}")
print(f"Total cells before: {orig_count}, after: {len(filtered)}")
