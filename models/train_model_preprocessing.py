"""
preprocessing.ipynb 기준 공행성 쌍 예측 모델
- preprocessing.ipynb에서 생성한 df_panel을 사용
- 시계열 특성 및 공급망 정보 활용
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'submissions'

# analysis 폴더의 전처리 결과를 사용할 수 있도록 경로 설정
sys.path.append(str(PROJECT_ROOT / 'analysis'))

print("="*70)
print("preprocessing.ipynb 기준 공행성 쌍 예측 모델")
print("="*70)

# 1. 전처리된 데이터 로드 (preprocessing.ipynb 결과 사용)
print("\n[1단계] 전처리된 데이터 로드...")
print("주의: preprocessing.ipynb를 먼저 실행하여 df_panel을 준비해야 합니다.")

# preprocessing.ipynb에서 생성된 df_panel을 사용한다고 가정
# 실제로는 전처리 결과를 저장했다가 로드하거나, 전처리 코드를 포함해야 함
# 여기서는 train.csv를 다시 로드하고 전처리하는 방식으로 구현

train = pd.read_csv(str(DATA_DIR / 'train.csv'))

# preprocessing.ipynb 방식으로 전처리
# 1. date 컬럼 생성
train['date'] = pd.to_datetime(
    train['year'].astype(str) + "-" + train['month'].astype(str).str.zfill(2) + "-01"
)

# 2. 월별 집계
agg_cols = ["value", "weight", "quantity"]
df_train_monthly = (
    train.groupby(["item_id", "hs4", "date"], as_index=False)[agg_cols].sum()
)

# 3. Panel 생성
min_month = df_train_monthly["date"].min()
max_month = df_train_monthly["date"].max()
full_months = pd.date_range(min_month, max_month, freq="MS")
all_items = df_train_monthly["item_id"].unique()

panel = pd.MultiIndex.from_product(
    [all_items, full_months],
    names=["item_id", "date"]
).to_frame(index=False)

df_panel = (
    panel.merge(df_train_monthly, on=["item_id", "date"], how="left")
         .sort_values(["item_id", "date"])
         .reset_index(drop=True)
)

# NaN을 0으로 채우기
for c in agg_cols:
    df_panel[c] = df_panel[c].fillna(0)

df_panel["year"] = df_panel["date"].dt.year
df_panel["month"] = df_panel["date"].dt.month

# 4. HS4 매핑 보완
item_hs4_mapping = (
    df_panel[df_panel['hs4'].notna()]
    .groupby('item_id')['hs4']
    .first()
    .to_dict()
)
df_panel['hs4'] = df_panel['item_id'].map(item_hs4_mapping).fillna(df_panel['hs4'])

# 5. 공급망 정보 추가
def get_supply_chain_tier(hs4):
    if pd.isna(hs4):
        return 0
    hs4_int = int(hs4)
    if 2800 <= hs4_int <= 3899:
        return 1  # Tier 1
    elif 7200 <= hs4_int <= 8399:
        return 2  # Tier 2
    elif 8400 <= hs4_int <= 8599:
        return 3  # Tier 3
    else:
        return 0

df_panel['supply_chain_tier'] = df_panel['hs4'].apply(get_supply_chain_tier)

# 6. 시계열 특성 추가 (간단 버전)
df_panel = df_panel.sort_values(['item_id', 'date']).reset_index(drop=True)
df_panel['value_ma6'] = df_panel.groupby('item_id')['value'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)
# 변화율 계산 (inf, -inf 처리)
df_panel['value_pct_change'] = df_panel.groupby('item_id')['value'].pct_change().fillna(0)
# inf, -inf 값을 0으로 대체 (0에서 나누기로 인한 inf 발생 방지)
df_panel['value_pct_change'] = df_panel['value_pct_change'].replace([np.inf, -np.inf], 0)

print(f"전처리 완료: {len(df_panel):,}행")

# 2. Pivot 테이블 생성 (value 사용)
print("\n[2단계] Pivot 테이블 생성...")
pivot = (
    df_panel.pivot_table(
        index='item_id',
        columns='date',
        values='value',
        aggfunc='sum',
        fill_value=0.0
    )
)

print(f"Pivot table shape: {pivot.shape}")

# item_id-hs4 매핑
item_hs4 = df_panel[['item_id', 'hs4']].drop_duplicates().set_index('item_id')['hs4'].to_dict()

# 3. 공행성쌍 탐색
def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def calculate_stability_score(x, y, lag):
    """시계열 안정성 점수 계산 (변동성 낮을수록 높은 점수)"""
    if len(x) <= lag or len(y) <= lag:
        return 0.0
    
    x_aligned = x[:-lag]
    y_aligned = y[lag:]
    
    # 0이 아닌 값만 사용
    x_nonzero = x_aligned[x_aligned != 0]
    y_nonzero = y_aligned[y_aligned != 0]
    
    if len(x_nonzero) < 3 or len(y_nonzero) < 3:
        return 0.0
    
    # 변동계수 (CV) 계산: 표준편차 / 평균
    cv_x = np.std(x_nonzero) / (np.mean(x_nonzero) + 1e-8)
    cv_y = np.std(y_nonzero) / (np.mean(y_nonzero) + 1e-8)
    
    # 변동계수가 낮을수록 안정적 (점수 높음)
    stability = 1.0 / (1.0 + (cv_x + cv_y) / 2.0)
    return min(stability, 1.0)


def find_comovement_pairs(pivot, item_hs4, max_lag=6, min_nonzero=12, 
                          corr_threshold=0.4, min_abs_corr=0.3, max_pairs=None):
    """
    공행성쌍 탐색 (개선 버전)
    - 안정성 점수 추가
    - 최소 상관계수 조건 추가
    - 종합 점수로 정렬 후 상위 N개 선택
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items), desc="Finding comovement pairs"):
        leader_hs4 = item_hs4.get(leader, 0)
        x = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            follower_hs4 = item_hs4.get(follower, 0)
            y = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0
            best_stability = 0.0

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                corr = safe_corr(x[:-lag], y[lag:])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
                    best_stability = calculate_stability_score(x, y, lag)

            # 이중 필터링: 최소 상관계수 + 임계값
            if best_lag is not None and abs(best_corr) >= min_abs_corr and abs(best_corr) >= corr_threshold:
                # 공급망 정보 추가
                leader_tier = get_supply_chain_tier(leader_hs4)
                follower_tier = get_supply_chain_tier(follower_hs4)
                same_hs4 = 1 if leader_hs4 == follower_hs4 else 0
                
                # 공급망 방향성 점수
                if leader_hs4 == follower_hs4:
                    supply_chain_score = 1.5
                elif leader_tier < follower_tier:
                    supply_chain_score = 2.0
                elif leader_tier == follower_tier:
                    supply_chain_score = 1.0
                else:
                    supply_chain_score = 0.5

                # 종합 점수: 상관계수 + 안정성 + 공급망 점수
                composite_score = abs(best_corr) * 0.5 + best_stability * 0.3 + (supply_chain_score / 2.0) * 0.2

                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                    "stability_score": best_stability,
                    "composite_score": composite_score,
                    "supply_chain_score": supply_chain_score,
                    "same_hs4": same_hs4,
                })

    pairs = pd.DataFrame(results)
    
    # 종합 점수로 정렬 후 상위 N개 선택 (선택사항)
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs.nlargest(max_pairs, 'composite_score')
    
    return pairs


print("\n[3단계] 공행성쌍 탐색...")
# 개선 1: max_pairs 튜닝으로 Precision 최적화
# 종합 점수로 정렬 후 상위 N개만 선택 (Precision 향상)
# None이면 모든 쌍 사용, 숫자를 지정하면 상위 N개만 선택
MAX_PAIRS = None  # 튜닝 가능: 1200, 1500, 1800 등
pairs = find_comovement_pairs(pivot, item_hs4, max_pairs=MAX_PAIRS)
print(f"탐색된 공행성쌍 수: {len(pairs)}")
if MAX_PAIRS is not None:
    print(f"  (종합 점수 상위 {MAX_PAIRS}개 선택)")
print(f"Pairs head:\n{pairs.head()}")

# 4. 학습 데이터 생성
def build_training_data(pivot, pairs, df_panel):
    """preprocessing.ipynb의 시계열 특성을 활용한 학습 데이터 생성"""
    months = pivot.columns.to_list()
    n_months = len(months)

    rows = []

    for row in tqdm(pairs.itertuples(index=False), desc="Building training data", total=len(pairs)):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)
        supply_chain_score = float(row.supply_chain_score)
        same_hs4 = int(row.same_hs4)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            # 시계열 특성 추가 (해당 시점의 특성)
            current_date = months[t]
            try:
                leader_features = df_panel[
                    (df_panel['item_id'] == leader) & (df_panel['date'] == current_date)
                ]
                follower_features = df_panel[
                    (df_panel['item_id'] == follower) & (df_panel['date'] == current_date)
                ]

                leader_ma6 = leader_features['value_ma6'].values[0] if len(leader_features) > 0 and 'value_ma6' in leader_features.columns else 0
                follower_ma6 = follower_features['value_ma6'].values[0] if len(follower_features) > 0 and 'value_ma6' in follower_features.columns else 0
                follower_pct_change = follower_features['value_pct_change'].values[0] if len(follower_features) > 0 and 'value_pct_change' in follower_features.columns else 0
            except (KeyError, IndexError):
                # 컬럼이 없거나 데이터가 없는 경우 기본값 사용
                leader_ma6 = 0
                follower_ma6 = 0
                follower_pct_change = 0

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "supply_chain_score": supply_chain_score,
                "same_hs4": float(same_hs4),
                "follower_ma6": follower_ma6,
                "follower_pct_change": follower_pct_change,
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train


print("\n[4단계] 학습 데이터 생성...")
df_train_model = build_training_data(pivot, pairs, df_panel)
print(f'생성된 학습 데이터의 shape: {df_train_model.shape}')
print(f"Training data head:\n{df_train_model.head()}")

# 5. 회귀 모델 학습 (개선 버전)
print("\n[5단계] 회귀 모델 학습...")
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler

feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                'supply_chain_score', 'same_hs4', 'follower_ma6', 'follower_pct_change']

train_X = df_train_model[feature_cols].values
train_y = df_train_model["target"].values

# inf, -inf, NaN 값 처리
train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)

# Feature scaling (NMAE 개선을 위해)
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# 개선 2: NMAE 손실 함수 고려
# NMAE는 상대 오차를 사용하므로, Quantile Regression 또는 가중치 적용 고려
# 현재는 Ridge 사용 (alpha 튜닝 가능)
USE_QUANTILE = False  # True로 설정하면 Quantile Regression 사용 (중앙값 예측)
RIDGE_ALPHA = 1.0  # 튜닝 가능: 0.1, 1.0, 10.0 등

if USE_QUANTILE:
    # Quantile Regression (중앙값 = 0.5 quantile)
    # NMAE와 유사한 특성: 상대 오차에 덜 민감
    reg = QuantileRegressor(quantile=0.5, alpha=RIDGE_ALPHA, solver='highs')
    print("  - Quantile Regression 사용 (중앙값 예측)")
else:
    # Ridge 회귀 (정규화로 과적합 방지)
    reg = Ridge(alpha=RIDGE_ALPHA)
    print(f"  - Ridge Regression 사용 (alpha={RIDGE_ALPHA})")

reg.fit(train_X_scaled, train_y)

print("Model training completed!")
print(f"Feature importance (coefficients):")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {reg.coef_[i]:.6f}")

# 6. 예측 및 제출 파일 생성
def predict(pivot, pairs, reg, df_panel):
    """예측 수행"""
    months = pivot.columns.to_list()
    n_months = len(months)

    t_last = n_months - 1
    t_prev = n_months - 2

    preds = []

    for row in tqdm(pairs.itertuples(index=False), desc="Making predictions", total=len(pairs)):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)
        supply_chain_score = float(row.supply_chain_score)
        same_hs4 = int(row.same_hs4)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        # 시계열 특성
        current_date = months[t_last]
        try:
            follower_features = df_panel[
                (df_panel['item_id'] == follower) & (df_panel['date'] == current_date)
            ]
            follower_ma6 = follower_features['value_ma6'].values[0] if len(follower_features) > 0 and 'value_ma6' in follower_features.columns else 0
            follower_pct_change = follower_features['value_pct_change'].values[0] if len(follower_features) > 0 and 'value_pct_change' in follower_features.columns else 0
        except (KeyError, IndexError):
            # 컬럼이 없거나 데이터가 없는 경우 기본값 사용
            follower_ma6 = 0
            follower_pct_change = 0

        X_test = np.array([[b_t, b_t_1, a_t_lag, corr, float(lag),
                           supply_chain_score, float(same_hs4), follower_ma6, follower_pct_change]])
        # Feature scaling 적용
        X_test_scaled = scaler.transform(X_test)
        y_pred = reg.predict(X_test_scaled)[0]

        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred


print("\n[6단계] 예측 수행...")
submission = predict(pivot, pairs, reg, df_panel)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_DIR / 'preprocessing_submit.csv'
submission.to_csv(str(output_path), index=False)
print(f"제출 파일 생성 완료: {output_path}")
print(f"예측된 공행성쌍 수: {len(submission)}")

