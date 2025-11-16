import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import matplotlib.dates as mdates


# 시각화 플래그 (1번 담당자는 시각화 중요)
SHOW_PLOTS = True
plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
plt.rcParams['axes.unicode_minus'] = False      # 마이너스(-) 깨짐 방지

plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
plt.rcParams['axes.unicode_minus'] = False      # 마이너스(-) 깨짐 방지

df = pd.read_csv("train.csv")


# 1. Quantity가 0인 레코드만 필터링
zero_quantity_df = df[df['quantity'] == 0]
# 2. 전체 Quantity=0 레코드 수 확인
total_zero_count = len(zero_quantity_df)
# 3. HS4 코드별 빈도수 계산
hs4_counts = zero_quantity_df['hs4'].value_counts()
# 4. 상위 10개 HS4 코드 및 비율 계산
top_10_hs4_zero_quantity = hs4_counts.head(10)
top_10_percentage = (top_10_hs4_zero_quantity.sum() / total_zero_count) * 100

print(f"--- Records where Quantity = 0: {total_zero_count}개 ---")
print("--- Quantity가 0인 경우의 상위 10개 HS4 코드 빈도 ---")
print(top_10_hs4_zero_quantity)
print(f"\n상위 10개 HS4 코드가 차지하는 비율: {top_10_percentage:.2f}%")


#Quantity가 0인 상위 10개 HS4 코드의 Value, Weight

# 5. Quantity가 0이었던 상위 10개 HS4 코드 리스트
target_hs4_codes = [2805, 3824, 2811, 8505, 8102, 2807, 3815, 2846, 3102, 2529]

# 6. 해당 HS4 코드에 해당하는 모든 레코드 필터링
filtered_df = df[df['hs4'].isin(target_hs4_codes)]

# 7. HS4 코드별로 총 value와 총 weight 합산
hs4_magnitude_summary = filtered_df.groupby('hs4')[['value', 'weight']].sum().reset_index()

# 8. value 기준으로 내림차순 정렬
hs4_magnitude_summary = hs4_magnitude_summary.sort_values(by='value', ascending=False)

# 9. 출력 포맷 지정 및 컬럼 정리
hs4_magnitude_summary['total_value'] = hs4_magnitude_summary['value'].map('{:,.0f}'.format)
hs4_magnitude_summary['total_weight'] = hs4_magnitude_summary['weight'].map('{:,.0f}'.format)

final_summary = hs4_magnitude_summary[['hs4', 'total_value', 'total_weight']]
final_summary.columns = ['HS4 코드', '총 무역량 (Value)', '총 중량 (Weight)']

print("--- Quantity가 0인 빈도가 높았던 상위 10개 HS4 코드의 총 무역량 및 중량 ---")
# 오류 발생 라인을 to_string으로 수정하여 tabulate 종속성 제거
print(final_summary.to_string(index=False))



#HS4코드와 Item ID의 관계분석

hs4_count_per_item = df.groupby('item_id')['hs4'].nunique()
# 10. HS4 코드 당 Item ID 개수 확인
# HS4 코드를 기준으로 그룹화하여 고유 Item ID가 1개를 초과하는지 확인
item_count_per_hs4 = df.groupby('hs4')['item_id'].nunique()

# 11. 관계 유형 확인
is_item_one_to_one = (hs4_count_per_item > 1).any() # Item ID 하나가 여러 HS4에 대응하는가
is_hs4_one_to_many = (item_count_per_hs4 > 1).any() # HS4 하나가 여러 Item ID에 대응하는가

print(f"고유 Item ID 개수: {df['item_id'].nunique()}개")
print(f"고유 HS4 코드 개수: {df['hs4'].nunique()}개")
print(f"Item ID 당 HS4 코드 개수가 1개를 초과하는가 (다대일 관계 여부): {is_item_one_to_one}")
print(f"HS4 코드 당 Item ID 개수가 1개를 초과하는가 (일대다 관계 여부): {is_hs4_one_to_many}")

# 12. 예시 출력 (HS4 -> Item ID)
if is_hs4_one_to_many:
    multi_item_hs4 = item_count_per_hs4[item_count_per_hs4 > 1].sort_values(ascending=False).head(1)
    if not multi_item_hs4.empty:
        example_hs4 = multi_item_hs4.index[0]
        related_items = df[df['hs4'] == example_hs4]['item_id'].unique()
        print(f"\n예시 HS4 코드 {example_hs4}: {len(related_items)}개의 Item ID ({related_items[:3]}...)")




# 13. Item ID별 메타데이터 (월 커버리지) 계산
item_coverage = df.groupby('item_id')[['year', 'month']].apply(lambda x: x.drop_duplicates().shape[0]).reset_index()
item_coverage.columns = ['item_id', 'month_coverage']

# 14. 총 월 수 (43개월) 대비 비율 계산
total_months = df[['year', 'month']].drop_duplicates().shape[0]
item_coverage['coverage_rate'] = (item_coverage['month_coverage'] / total_months) * 100

# 15. Item ID 월 커버리지 정렬
item_coverage_sorted = item_coverage.sort_values(by='coverage_rate', ascending=False)

# 16. 출력 형식 수정 (Traceback 해결 코드)
# to_string()에서 복잡한 float_format 딕셔너리 사용 시 발생하는 TypeError를 회피하기 위해, 
# 출력 전 컬럼을 원하는 형식의 문자열로 명시적으로 변환합니다.
item_coverage_sorted['month_coverage'] = item_coverage_sorted['month_coverage'].astype(int).astype(str)
item_coverage_sorted['coverage_rate_str'] = item_coverage_sorted['coverage_rate'].map('{:,.2f}%'.format)

# 최종 출력 DataFrame 구성
final_output = item_coverage_sorted[['item_id', 'month_coverage', 'coverage_rate_str']]
final_output.columns = ['item_id', '월 커버리지 (개월)', '커버리지 비율']

print("\n--- 3. Item ID별 월 커버리지 (상위 5개 / 하위 5개) ---")

print("[상위 5개 (가장 꾸준한 Item)]")
print(final_output.head(5).to_string(index=False))

print("\n[하위 5개 (가장 불규칙한 Item)]")
print(final_output.tail(5).to_string(index=False))


target_hs4 = 4202
filtered_df = df[df['hs4'] == target_hs4]

# SettingWithCopyWarning 방지 및 월별 집계 준비
filtered_df.loc[:, 'date'] = pd.to_datetime(filtered_df['year'].astype(str) + '-' + filtered_df['month'].astype(str) + '-01')
monthly_aggregated = filtered_df.groupby('date')[['value', 'weight', 'quantity']].sum().reset_index()

# 2. 누락된 월 채우기 및 시계열 인덱스 설정
full_date_range = pd.date_range(start='2022-01-01', end='2025-07-01', freq='MS')
full_df = pd.DataFrame({'date': full_date_range})
ts_df_raw = pd.merge(full_df, monthly_aggregated, on='date', how='left').fillna(0)
ts_df_raw = ts_df_raw.set_index('date')

# 3. Quantity 최댓값 지점 찾기
max_qty_value = ts_df_raw['quantity'].max()
max_qty_dates = ts_df_raw[ts_df_raw['quantity'] == max_qty_value].index

# 4. 시각화 (3개 변수의 원본 시계열 플롯)
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
plt.suptitle(f'HS4 코드 {target_hs4} (여행용품/핸드백)의 월별 규모 트렌드 (원본 데이터)', fontsize=16)
plot_vars = ['value', 'weight', 'quantity']
titles = ['Trade Volume (Value)', 'Weight (중량)', 'Quantity (수량)']
colors = ['darkblue', 'darkgreen', 'darkorange']

for i, var in enumerate(plot_vars):
    axes[i].plot(ts_df_raw.index, ts_df_raw[var], label=var, color=colors[i], linewidth=2)
    axes[i].set_title(f'Raw {titles[i]}', fontsize=12, loc='left')
    axes[i].set_ylabel(f'Total {var}', fontsize=10)
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # 가로/세로 월별 점선 추가
    axes[i].grid(axis='y', which='both', linestyle='--', alpha=0.7) 
    axes[i].grid(axis='x', which='minor', linestyle=':', alpha=0.5) 
    axes[i].grid(axis='x', which='major', linestyle='--', alpha=0.7) 

    # Quantity 최댓값 마킹 (Request 3 - Quantity 플롯에만 적용)
    if var == 'quantity':
        axes[i].scatter(max_qty_dates, ts_df_raw.loc[max_qty_dates, 'quantity'], 
                        color='red', zorder=5, label=f'Max Quantity: {max_qty_value:,.0f}', s=80)
        axes[i].legend(loc='upper right', fontsize=10)


# X축 포맷 설정 (연/월 표시)
axes[-1].set_xlabel('Date (연-월)', fontsize=12)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6)) 
axes[-1].xaxis.set_minor_locator(mdates.MonthLocator()) 
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'hs4_{target_hs4}_timeseries_max_qty.png')
