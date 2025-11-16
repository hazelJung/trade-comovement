import pandas as pd

df = pd.read_csv('hs4_supply_chain_ratio.csv', encoding='utf-8-sig')

print("="*80)
print("Tier 1 (원자재 - 화학) 상세 설명")
print("="*80)
print("\n[정의]")
print("  - HS4 범위: 2800-3899 (화학공업 제품)")
print("  - 역할: 기계산업의 가장 기본이 되는 원자재")
print("  - 무역량 비중: 67.43% (가장 큼)")
print("  - 무게 비중: 87.36% (가장 큼)")

print("\n[주요 품목 - 상위 10개]")
tier1 = df[df['공급망계층'] == 'Tier 1 (원자재 - 화학)'].head(10)
for idx, row in tier1.iterrows():
    print(f"  - HS4 {int(row['HS4'])}: 무역량 {row['무역량']:,.0f} ({row['무역량비중(%)']}%)")

print("\n[예시]")
print("  - 암모니아(2814) → 비료 제조 → 농업 기계 부품")
print("  - 니켈(2805) → 배터리 제조 → 전기차 부품")
print("  - 화학제품(3824) → 플라스틱/고무 제조 → 기계 부품")

print("\n" + "="*80)
print("Tier 2 (중간재 - 금속) 상세 설명")
print("="*80)
print("\n[정의]")
print("  - HS4 범위: 7200-8399 (금속 및 금속제품)")
print("  - 역할: 원자재를 가공하여 만든 중간 제품 (부품 소재)")
print("  - 무역량 비중: 3.37%")
print("  - 무게 비중: 6.47%")

print("\n[주요 품목]")
tier2 = df[df['공급망계층'] == 'Tier 2 (중간재 - 금속)']
for idx, row in tier2.iterrows():
    print(f"  - HS4 {int(row['HS4'])}: 무역량 {row['무역량']:,.0f} ({row['무역량비중(%)']}%)")

print("\n[예시]")
print("  - 철강재(7207) → 기계 부품 제조 → 최종 기계 제품")
print("  - 니켈(8102) → 특수강 제조 → 항공기 부품")
print("  - 코발트(8105) → 배터리 제조 → 전기차 부품")

print("\n" + "="*80)
print("공급망 흐름 예시")
print("="*80)
print("\n예시 1: 전기차 배터리")
print("  Tier 1 (원자재)")
print("    ↓ 니켈(2805) + 코발트(8105)")
print("  Tier 2 (중간재)")
print("    ↓ 니켈(8102) + 코발트(8105) → 배터리 셀")
print("  Tier 3 (최종제품)")
print("    ↓ 전자석(8505) + 전기모터(8501) → 전기차")

print("\n예시 2: 공작기계")
print("  Tier 1 (원자재)")
print("    ↓ 화학제품(3824) → 플라스틱/고무")
print("  Tier 2 (중간재)")
print("    ↓ 철강재(7207) → 기계 부품")
print("  Tier 3 (최종제품)")
print("    ↓ 공작기계(8461) → 제조업 장비")

print("\n" + "="*80)
print("핵심 포인트")
print("="*80)
print("\n1. Tier 1 (원자재):")
print("   - 가장 큰 비중 (67.43%)")
print("   - 화학제품이 기계산업의 기초")
print("   - 예: 암모니아, 비료, 화학제품")
print("\n2. Tier 2 (중간재):")
print("   - 원자재를 가공한 부품 소재")
print("   - 예: 철강재, 니켈, 코발트")
print("\n3. 공급망 방향성:")
print("   - Tier 1 → Tier 2 → Tier 3 순서로 흐름")
print("   - 하위 계층의 변화가 상위 계층에 영향")

