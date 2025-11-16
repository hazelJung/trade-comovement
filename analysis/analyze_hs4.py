import pandas as pd

df = pd.read_csv('train.csv')

print("="*70)
print("HS4 코드 기계산업 공급망 연관성 분석")
print("="*70)

# HS4 코드 분류
hs4_list = sorted(df['hs4'].unique())

# 1. 기계류 및 전기기기 (84-85장)
machine_codes = [c for c in hs4_list if 8400 <= c < 8600]
print(f"\n[1. 기계류 및 전기기기 (84-85장)] - 총 {len(machine_codes)}개")
print(f"코드: {machine_codes}")
for hs4 in machine_codes:
    items = df[df['hs4']==hs4]['item_id'].unique()
    total_value = df[df['hs4']==hs4]['value'].sum()
    print(f"  HS4 {hs4}: 품목 {len(items)}개, 총 무역량 {total_value:,.0f}")

# 2. 금속 및 금속제품 (72-83장) - 기계 부품 원자재
metal_codes = [c for c in hs4_list if 7200 <= c < 8400]
print(f"\n[2. 금속 및 금속제품 (72-83장)] - 총 {len(metal_codes)}개")
print(f"코드: {metal_codes}")
for hs4 in metal_codes:
    items = df[df['hs4']==hs4]['item_id'].unique()
    total_value = df[df['hs4']==hs4]['value'].sum()
    print(f"  HS4 {hs4}: 품목 {len(items)}개, 총 무역량 {total_value:,.0f}")

# 3. 화학공업 제품 (28-38장) - 기계 산업용 화학재료
chem_codes = [c for c in hs4_list if 2800 <= c < 3900]
print(f"\n[3. 화학공업 제품 (28-38장)] - 총 {len(chem_codes)}개")
print(f"코드: {chem_codes}")
for hs4 in chem_codes:
    items = df[df['hs4']==hs4]['item_id'].unique()
    total_value = df[df['hs4']==hs4]['value'].sum()
    print(f"  HS4 {hs4}: 품목 {len(items)}개, 총 무역량 {total_value:,.0f}")

# 4. 운송기기 (86-89장)
transport_codes = [c for c in hs4_list if 8600 <= c < 9000]
print(f"\n[4. 운송기기 (86-89장)] - 총 {len(transport_codes)}개")
print(f"코드: {transport_codes}")
for hs4 in transport_codes:
    items = df[df['hs4']==hs4]['item_id'].unique()
    total_value = df[df['hs4']==hs4]['value'].sum()
    print(f"  HS4 {hs4}: 품목 {len(items)}개, 총 무역량 {total_value:,.0f}")

# 5. 정밀기기 (90-92장)
precision_codes = [c for c in hs4_list if 9000 <= c < 9300]
print(f"\n[5. 정밀기기 (90-92장)] - 총 {len(precision_codes)}개")
print(f"코드: {precision_codes}")
for hs4 in precision_codes:
    items = df[df['hs4']==hs4]['item_id'].unique()
    total_value = df[df['hs4']==hs4]['value'].sum()
    print(f"  HS4 {hs4}: 품목 {len(items)}개, 총 무역량 {total_value:,.0f}")

# 기계산업 관련성 요약
print("\n" + "="*70)
print("기계산업 공급망 관련성 요약")
print("="*70)

related_codes = machine_codes + metal_codes + chem_codes + transport_codes + precision_codes
related_items = df[df['hs4'].isin(related_codes)]['item_id'].nunique()
total_items = df['item_id'].nunique()
related_value = df[df['hs4'].isin(related_codes)]['value'].sum()
total_value = df['value'].sum()

print(f"\n기계산업 관련 HS4 코드: {len(related_codes)}개 / 전체 {len(hs4_list)}개 ({len(related_codes)/len(hs4_list)*100:.1f}%)")
print(f"기계산업 관련 품목: {related_items}개 / 전체 {total_items}개 ({related_items/total_items*100:.1f}%)")
print(f"기계산업 관련 무역량: {related_value:,.0f} / 전체 {total_value:,.0f} ({related_value/total_value*100:.1f}%)")

print("\n" + "="*70)
print("HS4 코드별 상세 정보 (기계산업 관련)")
print("="*70)

# 주요 기계산업 관련 코드 상세
key_codes = {
    8461: "절삭가공용 공작기계",
    8467: "기타 공작기계",
    8479: "기타 기계류",
    8501: "전기모터 및 발전기",
    8505: "전자석, 영구자석 및 그 부품",
    8527: "무선통신용 기기",
    8102: "니켈",
    8105: "코발트",
    7202: "철합금",
    7207: "철강재",
    2805: "니켈",
    2811: "기타 무기화합물",
    3824: "기타 화학제품"
}

for hs4, desc in key_codes.items():
    if hs4 in hs4_list:
        items = df[df['hs4']==hs4]['item_id'].unique()
        total_value = df[df['hs4']==hs4]['value'].sum()
        count = df[df['hs4']==hs4].shape[0]
        print(f"\nHS4 {hs4}: {desc}")
        print(f"  거래 건수: {count:,}건")
        print(f"  품목 수: {len(items)}개")
        print(f"  총 무역량: {total_value:,.0f}")
        print(f"  품목 ID 샘플: {list(items)[:3]}")

