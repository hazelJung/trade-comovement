import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("HS4 / item_id 군집성 & 품목 그룹 분석")
print("="*70)

# 데이터 로드
print("\n[1단계] 데이터 로드...")
df = pd.read_csv('./train.csv')
print(f"전체 데이터 shape: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")

# 기본 통계
print("\n[2단계] 기본 통계...")
print(f"HS4 코드 수: {df['hs4'].nunique()}개")
print(f"item_id 수: {df['item_id'].nunique()}개")
print(f"기간: {df['year'].min()}년 {df['month'].min()}월 ~ {df['year'].max()}년 {df['month'].max()}월")

# ============================================================================
# 1. HS4별 평균 value/weight
# ============================================================================
print("\n" + "="*70)
print("[분석 1] HS4별 평균 value/weight")
print("="*70)

hs4_stats = df.groupby('hs4').agg({
    'value': ['mean', 'sum', 'std', 'count'],
    'weight': ['mean', 'sum', 'std'],
    'quantity': ['mean', 'sum', 'std'],
    'item_id': 'nunique'
}).round(2)

hs4_stats.columns = ['_'.join(col).strip() for col in hs4_stats.columns.values]
hs4_stats = hs4_stats.reset_index()

# value/weight 비율 계산 (단가)
hs4_stats['avg_price_per_weight'] = (hs4_stats['value_mean'] / hs4_stats['weight_mean']).replace([np.inf, -np.inf], np.nan)
hs4_stats = hs4_stats.sort_values('value_sum', ascending=False)

print("\nHS4별 통계 (상위 20개):")
print(hs4_stats.head(20).to_string())

# 그래프 1: HS4별 평균 value/weight
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1-1. HS4별 평균 value (상위 20개)
top_hs4_value = hs4_stats.head(20)
axes[0, 0].barh(range(len(top_hs4_value)), top_hs4_value['value_mean'], color='steelblue')
axes[0, 0].set_yticks(range(len(top_hs4_value)))
axes[0, 0].set_yticklabels(top_hs4_value['hs4'].astype(str))
axes[0, 0].set_xlabel('평균 Value')
axes[0, 0].set_title('HS4별 평균 Value (상위 20개)')
axes[0, 0].invert_yaxis()

# 1-2. HS4별 평균 weight (상위 20개)
top_hs4_weight = hs4_stats.nlargest(20, 'weight_mean')
axes[0, 1].barh(range(len(top_hs4_weight)), top_hs4_weight['weight_mean'], color='coral')
axes[0, 1].set_yticks(range(len(top_hs4_weight)))
axes[0, 1].set_yticklabels(top_hs4_weight['hs4'].astype(str))
axes[0, 1].set_xlabel('평균 Weight')
axes[0, 1].set_title('HS4별 평균 Weight (상위 20개)')
axes[0, 1].invert_yaxis()

# 1-3. HS4별 총 Value (상위 20개)
axes[1, 0].barh(range(len(top_hs4_value)), top_hs4_value['value_sum'], color='green')
axes[1, 0].set_yticks(range(len(top_hs4_value)))
axes[1, 0].set_yticklabels(top_hs4_value['hs4'].astype(str))
axes[1, 0].set_xlabel('총 Value')
axes[1, 0].set_title('HS4별 총 Value (상위 20개)')
axes[1, 0].invert_yaxis()

# 1-4. HS4별 단가 (value/weight) (상위 20개)
top_hs4_price = hs4_stats.nlargest(20, 'avg_price_per_weight')
axes[1, 1].barh(range(len(top_hs4_price)), top_hs4_price['avg_price_per_weight'], color='purple')
axes[1, 1].set_yticks(range(len(top_hs4_price)))
axes[1, 1].set_yticklabels(top_hs4_price['hs4'].astype(str))
axes[1, 1].set_xlabel('단가 (Value/Weight)')
axes[1, 1].set_title('HS4별 단가 (상위 20개)')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('hs4_value_weight_analysis.png', dpi=300, bbox_inches='tight')
print("\n그래프 저장: hs4_value_weight_analysis.png")
plt.close()

# ============================================================================
# 2. HS4별 quantity 사용률
# ============================================================================
print("\n" + "="*70)
print("[분석 2] HS4별 quantity 사용률")
print("="*70)

hs4_quantity_stats = df.groupby('hs4').agg({
    'quantity': lambda x: (x > 0).sum(),  # quantity가 0보다 큰 행 수
    'value': 'count'  # 전체 행 수
}).reset_index()
hs4_quantity_stats.columns = ['hs4', 'quantity_used_count', 'total_count']
hs4_quantity_stats['quantity_usage_rate'] = (hs4_quantity_stats['quantity_used_count'] / 
                                             hs4_quantity_stats['total_count'] * 100).round(2)

hs4_quantity_stats = hs4_quantity_stats.sort_values('quantity_usage_rate', ascending=False)

print("\nHS4별 quantity 사용률 (상위 20개):")
print(hs4_quantity_stats.head(20).to_string())

# 그래프 2: HS4별 quantity 사용률
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 2-1. Quantity 사용률 (상위 20개)
top_quantity_usage = hs4_quantity_stats.head(20)
axes[0].barh(range(len(top_quantity_usage)), top_quantity_usage['quantity_usage_rate'], color='teal')
axes[0].set_yticks(range(len(top_quantity_usage)))
axes[0].set_yticklabels(top_quantity_usage['hs4'].astype(str))
axes[0].set_xlabel('Quantity 사용률 (%)')
axes[0].set_title('HS4별 Quantity 사용률 (상위 20개)')
axes[0].invert_yaxis()

# 2-2. Quantity 사용률 분포
axes[1].hist(hs4_quantity_stats['quantity_usage_rate'], bins=30, color='orange', edgecolor='black')
axes[1].set_xlabel('Quantity 사용률 (%)')
axes[1].set_ylabel('HS4 코드 수')
axes[1].set_title('HS4별 Quantity 사용률 분포')
axes[1].axvline(hs4_quantity_stats['quantity_usage_rate'].mean(), 
                color='red', linestyle='--', label=f'평균: {hs4_quantity_stats["quantity_usage_rate"].mean():.2f}%')
axes[1].legend()

plt.tight_layout()
plt.savefig('hs4_quantity_usage.png', dpi=300, bbox_inches='tight')
print("그래프 저장: hs4_quantity_usage.png")
plt.close()

# ============================================================================
# 3. item_id별 거래량 상위/하위 분석
# ============================================================================
print("\n" + "="*70)
print("[분석 3] item_id별 거래량 상위/하위 분석")
print("="*70)

item_stats = df.groupby('item_id').agg({
    'value': ['sum', 'mean', 'count'],
    'weight': ['sum', 'mean'],
    'quantity': ['sum', 'mean'],
    'year': ['min', 'max'],
    'month': ['min', 'max']
}).round(2)

item_stats.columns = ['_'.join(col).strip() for col in item_stats.columns.values]
item_stats = item_stats.reset_index()
item_stats = item_stats.sort_values('value_sum', ascending=False)

print(f"\n전체 item_id 수: {len(item_stats)}개")
print(f"\n상위 10개 item_id (거래량 기준):")
print(item_stats.head(10)[['item_id', 'value_sum', 'value_mean', 'value_count']].to_string())

print(f"\n하위 10개 item_id (거래량 기준):")
print(item_stats.tail(10)[['item_id', 'value_sum', 'value_mean', 'value_count']].to_string())

# 그래프 3: item_id별 거래량 분포
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3-1. 상위 20개 item_id 거래량
top_items = item_stats.head(20)
axes[0, 0].barh(range(len(top_items)), top_items['value_sum'], color='darkblue')
axes[0, 0].set_yticks(range(len(top_items)))
axes[0, 0].set_yticklabels(top_items['item_id'], fontsize=8)
axes[0, 0].set_xlabel('총 Value')
axes[0, 0].set_title('상위 20개 item_id 거래량')
axes[0, 0].invert_yaxis()

# 3-2. 하위 20개 item_id 거래량
bottom_items = item_stats.tail(20)
axes[0, 1].barh(range(len(bottom_items)), bottom_items['value_sum'], color='lightcoral')
axes[0, 1].set_yticks(range(len(bottom_items)))
axes[0, 1].set_yticklabels(bottom_items['item_id'], fontsize=8)
axes[0, 1].set_xlabel('총 Value')
axes[0, 1].set_title('하위 20개 item_id 거래량')
axes[0, 1].invert_yaxis()

# 3-3. 거래량 분포 (로그 스케일)
axes[1, 0].hist(np.log1p(item_stats['value_sum']), bins=50, color='green', edgecolor='black')
axes[1, 0].set_xlabel('Log(총 Value + 1)')
axes[1, 0].set_ylabel('item_id 수')
axes[1, 0].set_title('item_id별 거래량 분포 (로그 스케일)')

# 3-4. 거래량 상위/하위 비교
top_10_sum = item_stats.head(10)['value_sum'].sum()
bottom_10_sum = item_stats.tail(10)['value_sum'].sum()
total_sum = item_stats['value_sum'].sum()

axes[1, 1].bar(['상위 10개', '하위 10개', '나머지'], 
               [top_10_sum, bottom_10_sum, total_sum - top_10_sum - bottom_10_sum],
               color=['darkblue', 'lightcoral', 'lightgray'])
axes[1, 1].set_ylabel('총 Value')
axes[1, 1].set_title('거래량 상위/하위 비교')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('item_id_trade_volume_analysis.png', dpi=300, bbox_inches='tight')
print("그래프 저장: item_id_trade_volume_analysis.png")
plt.close()

# ============================================================================
# 4. item_id 개수, item당 row 길이·월 커버리지
# ============================================================================
print("\n" + "="*70)
print("[분석 4] item_id 개수, item당 row 길이·월 커버리지")
print("="*70)

item_coverage = df.groupby('item_id').agg({
    'year': lambda x: f"{x.min()}-{x.max()}",
    'month': ['min', 'max', 'nunique'],
    'value': 'count'
}).reset_index()

item_coverage.columns = ['item_id', 'year_range', 'month_min', 'month_max', 'month_nunique', 'row_count']

# 전체 기간 계산
total_months = (df['year'].max() - df['year'].min()) * 12 + (df['month'].max() - df['month'].min() + 1)
item_coverage['month_coverage_rate'] = (item_coverage['month_nunique'] / total_months * 100).round(2)

item_coverage = item_coverage.sort_values('row_count', ascending=False)

print(f"\n전체 기간: {df['year'].min()}년 {df['month'].min()}월 ~ {df['year'].max()}년 {df['month'].max()}월")
print(f"총 월 수: {total_months}개월")
print(f"\nitem_id 통계:")
print(f"  - 총 item_id 수: {len(item_coverage)}개")
print(f"  - 평균 row 수: {item_coverage['row_count'].mean():.1f}개")
print(f"  - 평균 월 커버리지: {item_coverage['month_nunique'].mean():.1f}개월")
print(f"  - 평균 월 커버리지율: {item_coverage['month_coverage_rate'].mean():.2f}%")

print(f"\n상위 10개 item_id (row 수 기준):")
print(item_coverage.head(10).to_string())

# 그래프 4: item_id별 row 길이 및 월 커버리지
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4-1. item_id별 row 수 분포
axes[0, 0].hist(item_coverage['row_count'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Row 수')
axes[0, 0].set_ylabel('item_id 수')
axes[0, 0].set_title('item_id별 Row 수 분포')
axes[0, 0].axvline(item_coverage['row_count'].mean(), 
                   color='red', linestyle='--', 
                   label=f'평균: {item_coverage["row_count"].mean():.1f}')
axes[0, 0].legend()

# 4-2. item_id별 월 커버리지 분포
axes[0, 1].hist(item_coverage['month_nunique'], bins=50, color='lightgreen', edgecolor='black')
axes[0, 1].set_xlabel('월 커버리지 (개월)')
axes[0, 1].set_ylabel('item_id 수')
axes[0, 1].set_title('item_id별 월 커버리지 분포')
axes[0, 1].axvline(item_coverage['month_nunique'].mean(), 
                   color='red', linestyle='--', 
                   label=f'평균: {item_coverage["month_nunique"].mean():.1f}')
axes[0, 1].legend()

# 4-3. Row 수 vs 월 커버리지 산점도
axes[1, 0].scatter(item_coverage['row_count'], item_coverage['month_nunique'], 
                  alpha=0.5, color='purple')
axes[1, 0].set_xlabel('Row 수')
axes[1, 0].set_ylabel('월 커버리지 (개월)')
axes[1, 0].set_title('Row 수 vs 월 커버리지')
axes[1, 0].grid(True, alpha=0.3)

# 4-4. 월 커버리지율 분포
axes[1, 1].hist(item_coverage['month_coverage_rate'], bins=50, color='orange', edgecolor='black')
axes[1, 1].set_xlabel('월 커버리지율 (%)')
axes[1, 1].set_ylabel('item_id 수')
axes[1, 1].set_title('item_id별 월 커버리지율 분포')
axes[1, 1].axvline(item_coverage['month_coverage_rate'].mean(), 
                   color='red', linestyle='--', 
                   label=f'평균: {item_coverage["month_coverage_rate"].mean():.2f}%')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('item_id_coverage_analysis.png', dpi=300, bbox_inches='tight')
print("그래프 저장: item_id_coverage_analysis.png")
plt.close()

# ============================================================================
# 5. HS4 or item_id 간 군집 분포 그래프
# ============================================================================
print("\n" + "="*70)
print("[분석 5] HS4 or item_id 간 군집 분포 그래프")
print("="*70)

# 5-1. HS4별 item_id 수 분포
hs4_item_count = df.groupby('hs4')['item_id'].nunique().sort_values(ascending=False)

print(f"\nHS4별 item_id 수:")
print(f"  - 평균: {hs4_item_count.mean():.2f}개")
print(f"  - 최대: {hs4_item_count.max()}개")
print(f"  - 최소: {hs4_item_count.min()}개")

# 5-2. item_id별 HS4 수 (대부분 1개일 것으로 예상)
item_hs4_count = df.groupby('item_id')['hs4'].nunique()

print(f"\nitem_id별 HS4 수:")
print(f"  - 평균: {item_hs4_count.mean():.2f}개")
print(f"  - 최대: {item_hs4_count.max()}개")
print(f"  - 최소: {item_hs4_count.min()}개")

# 그래프 5: 군집 분포
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5-1. HS4별 item_id 수 (상위 30개)
top_hs4_items = hs4_item_count.head(30)
axes[0, 0].barh(range(len(top_hs4_items)), top_hs4_items.values, color='steelblue')
axes[0, 0].set_yticks(range(len(top_hs4_items)))
axes[0, 0].set_yticklabels(top_hs4_items.index.astype(str), fontsize=8)
axes[0, 0].set_xlabel('item_id 수')
axes[0, 0].set_title('HS4별 item_id 수 (상위 30개)')
axes[0, 0].invert_yaxis()

# 5-2. HS4별 item_id 수 분포
axes[0, 1].hist(hs4_item_count.values, bins=30, color='coral', edgecolor='black')
axes[0, 1].set_xlabel('item_id 수')
axes[0, 1].set_ylabel('HS4 코드 수')
axes[0, 1].set_title('HS4별 item_id 수 분포')
axes[0, 1].axvline(hs4_item_count.mean(), color='red', linestyle='--', 
                   label=f'평균: {hs4_item_count.mean():.2f}')
axes[0, 1].legend()

# 5-3. item_id별 거래 빈도 (상위 30개)
top_item_freq = item_coverage.head(30)
axes[1, 0].barh(range(len(top_item_freq)), top_item_freq['row_count'], color='green')
axes[1, 0].set_yticks(range(len(top_item_freq)))
axes[1, 0].set_yticklabels(top_item_freq['item_id'], fontsize=8)
axes[1, 0].set_xlabel('Row 수')
axes[1, 0].set_title('item_id별 거래 빈도 (상위 30개)')
axes[1, 0].invert_yaxis()

# 5-4. HS4별 거래 건수 분포
hs4_transaction_count = df.groupby('hs4').size().sort_values(ascending=False)
top_hs4_trans = hs4_transaction_count.head(30)
axes[1, 1].barh(range(len(top_hs4_trans)), top_hs4_trans.values, color='purple')
axes[1, 1].set_yticks(range(len(top_hs4_trans)))
axes[1, 1].set_yticklabels(top_hs4_trans.index.astype(str), fontsize=8)
axes[1, 1].set_xlabel('거래 건수')
axes[1, 1].set_title('HS4별 거래 건수 (상위 30개)')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
print("그래프 저장: cluster_distribution.png")
plt.close()

# ============================================================================
# 6. K-means Clustering (HS4 기반)
# ============================================================================
print("\n" + "="*70)
print("[분석 6] K-means Clustering (HS4 기반)")
print("="*70)

# HS4별 특성 추출
hs4_features = df.groupby('hs4').agg({
    'value': ['mean', 'std', 'sum'],
    'weight': ['mean', 'sum'],
    'quantity': ['mean', 'sum'],
    'item_id': 'nunique',
    'year': 'nunique',
    'month': 'nunique'
}).reset_index()

hs4_features.columns = ['hs4', 'value_mean', 'value_std', 'value_sum', 
                        'weight_mean', 'weight_sum', 'quantity_mean', 'quantity_sum',
                        'item_count', 'year_count', 'month_count']

# NaN 처리 및 로그 변환
hs4_features = hs4_features.fillna(0)
numeric_cols = ['value_mean', 'value_std', 'value_sum', 'weight_mean', 'weight_sum',
               'quantity_mean', 'quantity_sum', 'item_count', 'year_count', 'month_count']

# 로그 변환 (0인 경우 처리)
for col in numeric_cols:
    hs4_features[f'{col}_log'] = np.log1p(hs4_features[col])

# Clustering에 사용할 feature 선택
cluster_features = ['value_mean_log', 'value_sum_log', 'weight_mean_log', 
                    'item_count', 'month_count']

X = hs4_features[cluster_features].values

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 최적 클러스터 수 찾기 (Elbow Method)
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Elbow 그래프
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('클러스터 수 (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

# 최적 K 선택 (간단히 4로 설정)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
hs4_features['cluster'] = kmeans.fit_predict(X_scaled)

# 클러스터별 통계
print(f"\n클러스터 수: {optimal_k}개")
print("\n클러스터별 통계:")
cluster_stats = hs4_features.groupby('cluster')[cluster_features].mean()
print(cluster_stats.round(2))

# 클러스터별 HS4 수
cluster_counts = hs4_features['cluster'].value_counts().sort_index()
print(f"\n클러스터별 HS4 수:")
print(cluster_counts)

# 클러스터 시각화 (PCA로 2D 축소)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hs4_features['cluster'], 
               cmap='viridis', s=100, alpha=0.6)
axes[1].set_xlabel(f'PC1 (설명력: {pca.explained_variance_ratio_[0]:.2%})')
axes[1].set_ylabel(f'PC2 (설명력: {pca.explained_variance_ratio_[1]:.2%})')
axes[1].set_title('HS4 Clustering (PCA 2D)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hs4_clustering.png', dpi=300, bbox_inches='tight')
print("그래프 저장: hs4_clustering.png")
plt.close()

# 클러스터별 상세 정보
print("\n클러스터별 상세 정보:")
for cluster_id in sorted(hs4_features['cluster'].unique()):
    cluster_hs4 = hs4_features[hs4_features['cluster'] == cluster_id]
    print(f"\n[클러스터 {cluster_id}] - {len(cluster_hs4)}개 HS4")
    print(f"  평균 value_sum: {cluster_hs4['value_sum'].mean():,.0f}")
    print(f"  평균 item_count: {cluster_hs4['item_count'].mean():.1f}")
    print(f"  대표 HS4: {cluster_hs4.nlargest(5, 'value_sum')['hs4'].tolist()}")

# 결과 저장
hs4_features.to_csv('hs4_clustering_results.csv', index=False)
print("\n클러스터링 결과 저장: hs4_clustering_results.csv")

# ============================================================================
# 7. 종합 요약
# ============================================================================
print("\n" + "="*70)
print("[종합 요약]")
print("="*70)

print(f"""
1. HS4 분석:
   - 총 HS4 코드: {df['hs4'].nunique()}개
   - 평균 value/weight: {hs4_stats['avg_price_per_weight'].mean():.2f}
   - Quantity 사용률 평균: {hs4_quantity_stats['quantity_usage_rate'].mean():.2f}%

2. item_id 분석:
   - 총 item_id: {df['item_id'].nunique()}개
   - 평균 거래량: {item_stats['value_sum'].mean():,.0f}
   - 평균 월 커버리지: {item_coverage['month_nunique'].mean():.1f}개월

3. 군집 분석:
   - HS4별 평균 item_id 수: {hs4_item_count.mean():.2f}개
   - K-means 클러스터: {optimal_k}개

4. 생성된 파일:
   - hs4_value_weight_analysis.png
   - hs4_quantity_usage.png
   - item_id_trade_volume_analysis.png
   - item_id_coverage_analysis.png
   - cluster_distribution.png
   - hs4_clustering.png
   - hs4_clustering_results.csv
""")

print("="*70)
print("분석 완료!")
print("="*70)

