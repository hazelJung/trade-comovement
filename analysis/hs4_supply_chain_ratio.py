import pandas as pd
import numpy as np

print("="*70)
print("HS4 코드 기계산업 공급망 비중 분석")
print("="*70)

# 데이터 로드
df = pd.read_csv('./train.csv')

# 공급망 계층 정의 함수
def get_supply_chain_tier(hs4):
    """HS4 코드를 기반으로 공급망 계층 반환"""
    if 8400 <= hs4 < 8600:  # 기계류 및 전기기기
        return "Tier 3 (최종제품 - 기계류)"
    elif 8600 <= hs4 < 9000:  # 운송기기
        return "Tier 3 (최종제품 - 운송기기)"
    elif 9000 <= hs4 < 9300:  # 정밀기기
        return "Tier 3 (최종제품 - 정밀기기)"
    elif 7200 <= hs4 < 8400:  # 금속제품
        return "Tier 2 (중간재 - 금속)"
    elif 2800 <= hs4 < 3900:  # 화학제품
        return "Tier 1 (원자재 - 화학)"
    else:
        return "기타"

def get_supply_chain_category(hs4):
    """HS4 코드를 기반으로 공급망 카테고리 반환"""
    if 8400 <= hs4 < 8600:
        return "기계산업 관련"
    elif 8600 <= hs4 < 9000:
        return "기계산업 관련"
    elif 9000 <= hs4 < 9300:
        return "기계산업 관련"
    elif 7200 <= hs4 < 8400:
        return "기계산업 관련"
    elif 2800 <= hs4 < 3900:
        return "기계산업 관련"
    else:
        return "기타"

# HS4별 통계 계산
hs4_analysis = df.groupby('hs4').agg({
    'value': ['sum', 'mean', 'count'],
    'weight': ['sum', 'mean'],
    'item_id': 'nunique'
}).reset_index()

hs4_analysis.columns = ['hs4', 'value_sum', 'value_mean', 'transaction_count', 
                        'weight_sum', 'weight_mean', 'item_count']

# 공급망 정보 추가
hs4_analysis['supply_chain_tier'] = hs4_analysis['hs4'].apply(get_supply_chain_tier)
hs4_analysis['supply_chain_category'] = hs4_analysis['hs4'].apply(get_supply_chain_category)

# 전체 통계
total_value = df['value'].sum()
total_weight = df['weight'].sum()
total_transactions = len(df)

# 비중 계산
hs4_analysis['value_ratio'] = (hs4_analysis['value_sum'] / total_value * 100).round(2)
hs4_analysis['weight_ratio'] = (hs4_analysis['weight_sum'] / total_weight * 100).round(2)
hs4_analysis['transaction_ratio'] = (hs4_analysis['transaction_count'] / total_transactions * 100).round(2)

# 기계산업 관련 여부
hs4_analysis['is_machinery_related'] = hs4_analysis['supply_chain_category'] == '기계산업 관련'

# 정렬 (value_sum 기준)
hs4_analysis = hs4_analysis.sort_values('value_sum', ascending=False)

# 기계산업 관련 통계
machinery_related = hs4_analysis[hs4_analysis['is_machinery_related'] == True]
machinery_value = machinery_related['value_sum'].sum()
machinery_weight = machinery_related['weight_sum'].sum()
machinery_transactions = machinery_related['transaction_count'].sum()
machinery_hs4_count = len(machinery_related)

print("\n[전체 통계]")
print(f"총 무역량 (value): {total_value:,.0f}")
print(f"총 무게 (weight): {total_weight:,.0f}")
print(f"총 거래 건수: {total_transactions:,}건")
print(f"총 HS4 코드 수: {df['hs4'].nunique()}개")

print("\n[기계산업 공급망 통계]")
print(f"기계산업 관련 HS4 코드: {machinery_hs4_count}개 / 전체 {df['hs4'].nunique()}개 ({machinery_hs4_count/df['hs4'].nunique()*100:.1f}%)")
print(f"기계산업 관련 무역량: {machinery_value:,.0f} / 전체 {total_value:,.0f} ({machinery_value/total_value*100:.1f}%)")
print(f"기계산업 관련 무게: {machinery_weight:,.0f} / 전체 {total_weight:,.0f} ({machinery_weight/total_weight*100:.1f}%)")
print(f"기계산업 관련 거래 건수: {machinery_transactions:,}건 / 전체 {total_transactions:,}건 ({machinery_transactions/total_transactions*100:.1f}%)")

# 공급망 계층별 통계
print("\n[공급망 계층별 통계]")
tier_stats = hs4_analysis.groupby('supply_chain_tier').agg({
    'value_sum': 'sum',
    'weight_sum': 'sum',
    'transaction_count': 'sum',
    'hs4': 'count',
    'item_count': 'sum'  # item_count의 합
}).reset_index()

tier_stats.columns = ['supply_chain_tier', 'value_sum', 'weight_sum', 'transaction_count', 'hs4_count', 'item_count']
tier_stats['value_ratio'] = (tier_stats['value_sum'] / total_value * 100).round(2)
tier_stats['weight_ratio'] = (tier_stats['weight_sum'] / total_weight * 100).round(2)
tier_stats['transaction_ratio'] = (tier_stats['transaction_count'] / total_transactions * 100).round(2)

tier_stats = tier_stats.sort_values('value_sum', ascending=False)
print(tier_stats.to_string(index=False))

# 기계산업 관련 HS4 상세 정보
print("\n" + "="*70)
print("[기계산업 관련 HS4 코드 상세 정보]")
print("="*70)

machinery_detail = machinery_related[['hs4', 'supply_chain_tier', 'value_sum', 'value_ratio', 
                                      'weight_sum', 'weight_ratio', 'transaction_count', 
                                      'transaction_ratio', 'item_count']].copy()

machinery_detail.columns = ['HS4', '공급망 계층', '무역량(Value)', '무역량 비중(%)', 
                            '무게(Weight)', '무게 비중(%)', '거래 건수', '거래 비중(%)', '품목 수']

# 천 단위 구분 표시
machinery_detail['무역량(Value)'] = machinery_detail['무역량(Value)'].apply(lambda x: f"{x:,.0f}")
machinery_detail['무게(Weight)'] = machinery_detail['무게(Weight)'].apply(lambda x: f"{x:,.0f}")

print("\n기계산업 관련 HS4 코드 (무역량 순):")
print(machinery_detail.to_string(index=False))

# 계층별 상세 정보
print("\n" + "="*70)
print("[공급망 계층별 상세 정보]")
print("="*70)

for tier in ['Tier 1 (원자재 - 화학)', 'Tier 2 (중간재 - 금속)', 
             'Tier 3 (최종제품 - 기계류)', 'Tier 3 (최종제품 - 운송기기)', 
             'Tier 3 (최종제품 - 정밀기기)']:
    tier_data = machinery_related[machinery_related['supply_chain_tier'] == tier]
    if len(tier_data) > 0:
        print(f"\n[{tier}]")
        tier_detail = tier_data[['hs4', 'value_sum', 'value_ratio', 'item_count']].head(10).copy()
        tier_detail.columns = ['HS4', '무역량', '비중(%)', '품목수']
        tier_detail['무역량'] = tier_detail['무역량'].apply(lambda x: f"{x:,.0f}")
        print(tier_detail.to_string(index=False))
        
        tier_total_value = tier_data['value_sum'].sum()
        tier_total_ratio = (tier_total_value / total_value * 100)
        print(f"\n계층 총계: {tier_total_value:,.0f} ({tier_total_ratio:.2f}%)")

# CSV로 저장
print("\n" + "="*70)
print("[데이터프레임 저장]")
print("="*70)

# 전체 HS4 정보 저장
hs4_analysis_export = hs4_analysis[['hs4', 'supply_chain_tier', 'supply_chain_category', 
                                    'value_sum', 'value_ratio', 'weight_sum', 'weight_ratio',
                                    'transaction_count', 'transaction_ratio', 'item_count']].copy()
hs4_analysis_export.columns = ['HS4', '공급망계층', '카테고리', '무역량', '무역량비중(%)', 
                               '무게', '무게비중(%)', '거래건수', '거래비중(%)', '품목수']
hs4_analysis_export.to_csv('hs4_supply_chain_ratio.csv', index=False, encoding='utf-8-sig')
print("저장 완료: hs4_supply_chain_ratio.csv")

# 계층별 요약 저장
tier_stats_export = tier_stats.copy()
tier_stats_export.columns = ['공급망계층', '무역량', '무게', '거래건수', 'HS4수', '품목수', 
                            '무역량비중(%)', '무게비중(%)', '거래비중(%)']
tier_stats_export.to_csv('hs4_supply_chain_tier_summary.csv', index=False, encoding='utf-8-sig')
print("저장 완료: hs4_supply_chain_tier_summary.csv")

# 데이터프레임 출력 (상위 30개)
print("\n" + "="*70)
print("[상위 30개 HS4 코드 (무역량 기준)]")
print("="*70)

display_df = hs4_analysis.head(30)[['hs4', 'supply_chain_tier', 'value_sum', 'value_ratio', 
                                   'item_count', 'transaction_count']].copy()
display_df.columns = ['HS4', '공급망 계층', '무역량', '비중(%)', '품목수', '거래건수']
display_df['무역량'] = display_df['무역량'].apply(lambda x: f"{x:,.0f}")

print(display_df.to_string(index=False))

# 기계산업 비중 요약표
print("\n" + "="*70)
print("[기계산업 공급망 비중 요약]")
print("="*70)

summary_data = {
    '구분': ['전체', '기계산업 관련', '기타'],
    'HS4 코드 수': [
        df['hs4'].nunique(),
        machinery_hs4_count,
        df['hs4'].nunique() - machinery_hs4_count
    ],
    'HS4 비중(%)': [
        100.0,
        round(machinery_hs4_count/df['hs4'].nunique()*100, 1),
        round((df['hs4'].nunique() - machinery_hs4_count)/df['hs4'].nunique()*100, 1)
    ],
    '무역량': [
        f"{total_value:,.0f}",
        f"{machinery_value:,.0f}",
        f"{total_value - machinery_value:,.0f}"
    ],
    '무역량 비중(%)': [
        100.0,
        round(machinery_value/total_value*100, 1),
        round((total_value - machinery_value)/total_value*100, 1)
    ],
    '거래 건수': [
        f"{total_transactions:,}",
        f"{machinery_transactions:,}",
        f"{total_transactions - machinery_transactions:,}"
    ],
    '거래 비중(%)': [
        100.0,
        round(machinery_transactions/total_transactions*100, 1),
        round((total_transactions - machinery_transactions)/total_transactions*100, 1)
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("분석 완료!")
print("="*70)

