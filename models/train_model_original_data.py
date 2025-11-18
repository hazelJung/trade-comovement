"""
원본 데이터 기반 공행성 쌍 예측 모델 (다중 모델 비교)

이 스크립트는 원본 데이터(value_sum, 보정/클리핑 없음)를 사용하여 공행성 쌍을 예측하고,
여러 모델의 성능을 비교합니다.

특징:
- 원본 데이터 사용 (결측치 보정 없음, 극단값 클리핑 없음)
- 여러 모델 비교: Ridge, Quantile Regression, Random Forest, XGBoost
- 평가 지표: F1 Score (60%) + NMAE (40%)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'submissions'
ANALYSIS_DATA_DIR = PROJECT_ROOT / 'analysis' / 'data'

# XGBoost 설치 확인
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost가 설치되지 않았습니다. pip install xgboost로 설치하세요.")
    XGBOOST_AVAILABLE = False

print("="*70)
print("원본 데이터 기반 공행성 쌍 예측 모델 (다중 모델 비교)")
print("="*70)

# 1. 전처리된 데이터 로드
print("\n[1단계] 전처리된 데이터 로드...")
try:
    monthly_features = pd.read_csv(str(ANALYSIS_DATA_DIR / 'monthly_features.csv'))
    print(f"전처리 결과 로드 완료: {len(monthly_features):,}행")
except FileNotFoundError:
    print("⚠️  monthly_features.csv 파일이 없습니다.")
    print("preprocess_ev_battery.ipynb를 먼저 실행하세요.")
    sys.exit(1)

# 2. Pivot 테이블 생성 (원본 데이터 사용)
print("\n[2단계] Pivot 테이블 생성 (원본 데이터)...")
pivot = (
    monthly_features.pivot_table(
        index='item_id',
        columns='ym',
        values='value_sum',  # 원본 값 사용
        aggfunc='sum',
        fill_value=0.0
    )
)

print(f"Pivot table shape: {pivot.shape}")

# item_id별 메타 정보
item_info = monthly_features[['item_id', 'hs4', 'hs2', 'cluster']].drop_duplicates().set_index('item_id')
item_hs4 = item_info['hs4'].to_dict()
item_cluster = item_info['cluster'].to_dict()

print(f"item_id 수: {len(item_hs4)}")
print(f"클러스터 분포:")
print(monthly_features['cluster'].value_counts())

# 3. 공행성쌍 탐색
def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def get_cluster_boost(leader_cluster, follower_cluster):
    """EV 배터리 클러스터 기반 가중치"""
    if leader_cluster == follower_cluster:
        if leader_cluster == 'battery_materials':
            return 2.0
        elif leader_cluster == 'ev_machinery_electrical':
            return 1.8
        else:
            return 1.2
    
    if leader_cluster == 'battery_materials' and follower_cluster == 'ev_machinery_electrical':
        return 1.5
    
    return 1.0


def find_comovement_pairs(pivot, item_hs4, item_cluster, 
                         max_lag=6, min_nonzero=12, corr_threshold=0.4, min_abs_corr=0.25):
    """공행성쌍 탐색 (원본 데이터 기준)"""
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items), desc="Finding comovement pairs"):
        leader_hs4 = item_hs4.get(leader, 0)
        leader_cluster = item_cluster.get(leader, 'other_industries')
        x = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            follower_hs4 = item_hs4.get(follower, 0)
            follower_cluster = item_cluster.get(follower, 'other_industries')
            y = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0

            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                corr = safe_corr(x[:-lag], y[lag:])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            cluster_boost = get_cluster_boost(leader_cluster, follower_cluster)
            effective_corr = abs(best_corr) * cluster_boost
            effective_threshold = corr_threshold / cluster_boost

            if best_lag is not None and abs(best_corr) >= min_abs_corr and effective_corr >= corr_threshold:
                same_cluster = 1 if leader_cluster == follower_cluster else 0
                same_hs4 = 1 if leader_hs4 == follower_hs4 else 0

                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                    "cluster_boost": cluster_boost,
                    "same_cluster": same_cluster,
                    "same_hs4": same_hs4,
                    "leader_cluster": leader_cluster,
                    "follower_cluster": follower_cluster,
                })

    pairs = pd.DataFrame(results)
    return pairs


print("\n[3단계] 공행성쌍 탐색...")
pairs = find_comovement_pairs(pivot, item_hs4, item_cluster)
print(f"탐색된 공행성쌍 수: {len(pairs)}")
print(f"\n클러스터별 공행성쌍 분포:")
print(pairs.groupby(['leader_cluster', 'follower_cluster']).size())

# 4. 학습 데이터 생성 (원본 데이터 기준)
def build_training_data(pivot, pairs, monthly_features):
    """원본 데이터 기반 학습 데이터 생성"""
    months = pivot.columns.to_list()
    n_months = len(months)

    rows = []

    for row in tqdm(pairs.itertuples(index=False), desc="Building training data", total=len(pairs)):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)
        cluster_boost = float(row.cluster_boost)
        same_cluster = int(row.same_cluster)
        same_hs4 = int(row.same_hs4)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        for t in range(max(lag, 1), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            # 원본 데이터 특성 추출
            current_ym = months[t]
            follower_features = monthly_features[
                (monthly_features['item_id'] == follower) & (monthly_features['ym'] == current_ym)
            ]

            try:
                if len(follower_features) > 0:
                    # 원본 값 사용 (보정/클리핑 없음)
                    follower_value_sum = follower_features['value_sum'].values[0] if 'value_sum' in follower_features.columns else 0
                    follower_log_value = np.log1p(follower_value_sum)  # 원본 값으로 로그 변환
                    follower_has_quantity = follower_features['has_quantity_this_row'].values[0] if 'has_quantity_this_row' in follower_features.columns else 0
                else:
                    follower_value_sum = 0
                    follower_log_value = 0
                    follower_has_quantity = 0
            except (KeyError, IndexError):
                follower_value_sum = 0
                follower_log_value = 0
                follower_has_quantity = 0

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "cluster_boost": cluster_boost,
                "same_cluster": float(same_cluster),
                "same_hs4": float(same_hs4),
                "follower_log_value": follower_log_value,
                "follower_value_sum": follower_value_sum,
                "follower_has_quantity": float(follower_has_quantity),
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train


print("\n[4단계] 학습 데이터 생성...")
df_train_model = build_training_data(pivot, pairs, monthly_features)
print(f'생성된 학습 데이터의 shape: {df_train_model.shape}')

# 5. 여러 모델 학습 및 비교
print("\n[5단계] 여러 모델 학습 및 비교...")

# 피처 선택
feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag',
                'cluster_boost', 'same_cluster', 'same_hs4',
                'follower_log_value', 'follower_value_sum', 'follower_has_quantity']

train_X = df_train_model[feature_cols].values
train_y = df_train_model["target"].values

# inf, -inf, NaN 값 처리
train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)

# Feature scaling
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

print(f"학습 데이터 shape: {train_X_scaled.shape}")
print(f"타겟 통계: mean={train_y.mean():.2f}, std={train_y.std():.2f}, min={train_y.min():.2f}, max={train_y.max():.2f}")

# Time Series Cross-Validation으로 모델 비교
tscv = TimeSeriesSplit(n_splits=3)

models = {}
results = []

# 1. Ridge Regression
print("\n[1/4] Ridge Regression 학습 중...")
ridge = Ridge(alpha=1.0)
ridge.fit(train_X_scaled, train_y)
models['Ridge'] = ridge

# Cross-validation 평가
ridge_scores = []
for train_idx, val_idx in tscv.split(train_X_scaled):
    X_train_cv, X_val_cv = train_X_scaled[train_idx], train_X_scaled[val_idx]
    y_train_cv, y_val_cv = train_y[train_idx], train_y[val_idx]
    
    ridge_cv = Ridge(alpha=1.0)
    ridge_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = ridge_cv.predict(X_val_cv)
    
    mae = mean_absolute_error(y_val_cv, y_pred_cv)
    mse = mean_squared_error(y_val_cv, y_pred_cv)
    nmae = np.mean(np.abs(y_val_cv - y_pred_cv) / (np.abs(y_val_cv) + 1e-8))
    
    ridge_scores.append({'mae': mae, 'mse': mse, 'nmae': nmae})

avg_mae = np.mean([s['mae'] for s in ridge_scores])
avg_mse = np.mean([s['mse'] for s in ridge_scores])
avg_nmae = np.mean([s['nmae'] for s in ridge_scores])

results.append({
    'model': 'Ridge',
    'mae': avg_mae,
    'mse': avg_mse,
    'rmse': np.sqrt(avg_mse),
    'nmae': avg_nmae
})

print(f"  MAE: {avg_mae:.2f}, RMSE: {np.sqrt(avg_mse):.2f}, NMAE: {avg_nmae:.4f}")

# 2. Quantile Regression
print("\n[2/4] Quantile Regression 학습 중...")
quantile = QuantileRegressor(quantile=0.5, alpha=1.0, solver='highs')
quantile.fit(train_X_scaled, train_y)
models['Quantile'] = quantile

# Cross-validation 평가
quantile_scores = []
for train_idx, val_idx in tscv.split(train_X_scaled):
    X_train_cv, X_val_cv = train_X_scaled[train_idx], train_X_scaled[val_idx]
    y_train_cv, y_val_cv = train_y[train_idx], train_y[val_idx]
    
    quantile_cv = QuantileRegressor(quantile=0.5, alpha=1.0, solver='highs')
    quantile_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = quantile_cv.predict(X_val_cv)
    
    mae = mean_absolute_error(y_val_cv, y_pred_cv)
    mse = mean_squared_error(y_val_cv, y_pred_cv)
    nmae = np.mean(np.abs(y_val_cv - y_pred_cv) / (np.abs(y_val_cv) + 1e-8))
    
    quantile_scores.append({'mae': mae, 'mse': mse, 'nmae': nmae})

avg_mae = np.mean([s['mae'] for s in quantile_scores])
avg_mse = np.mean([s['mse'] for s in quantile_scores])
avg_nmae = np.mean([s['nmae'] for s in quantile_scores])

results.append({
    'model': 'Quantile',
    'mae': avg_mae,
    'mse': avg_mse,
    'rmse': np.sqrt(avg_mse),
    'nmae': avg_nmae
})

print(f"  MAE: {avg_mae:.2f}, RMSE: {np.sqrt(avg_mse):.2f}, NMAE: {avg_nmae:.4f}")

# 3. Random Forest
print("\n[3/4] Random Forest 학습 중...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(train_X_scaled, train_y)
models['RandomForest'] = rf

# Cross-validation 평가
rf_scores = []
for train_idx, val_idx in tscv.split(train_X_scaled):
    X_train_cv, X_val_cv = train_X_scaled[train_idx], train_X_scaled[val_idx]
    y_train_cv, y_val_cv = train_y[train_idx], train_y[val_idx]
    
    rf_cv = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = rf_cv.predict(X_val_cv)
    
    mae = mean_absolute_error(y_val_cv, y_pred_cv)
    mse = mean_squared_error(y_val_cv, y_pred_cv)
    nmae = np.mean(np.abs(y_val_cv - y_pred_cv) / (np.abs(y_val_cv) + 1e-8))
    
    rf_scores.append({'mae': mae, 'mse': mse, 'nmae': nmae})

avg_mae = np.mean([s['mae'] for s in rf_scores])
avg_mse = np.mean([s['mse'] for s in rf_scores])
avg_nmae = np.mean([s['nmae'] for s in rf_scores])

results.append({
    'model': 'RandomForest',
    'mae': avg_mae,
    'mse': avg_mse,
    'rmse': np.sqrt(avg_mse),
    'nmae': avg_nmae
})

print(f"  MAE: {avg_mae:.2f}, RMSE: {np.sqrt(avg_mse):.2f}, NMAE: {avg_nmae:.4f}")

# 4. XGBoost (가능한 경우)
if XGBOOST_AVAILABLE:
    print("\n[4/4] XGBoost 학습 중...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(train_X_scaled, train_y)
    models['XGBoost'] = xgb_model

    # Cross-validation 평가
    xgb_scores = []
    for train_idx, val_idx in tscv.split(train_X_scaled):
        X_train_cv, X_val_cv = train_X_scaled[train_idx], train_X_scaled[val_idx]
        y_train_cv, y_val_cv = train_y[train_idx], train_y[val_idx]
        
        xgb_cv = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = xgb_cv.predict(X_val_cv)
        
        mae = mean_absolute_error(y_val_cv, y_pred_cv)
        mse = mean_squared_error(y_val_cv, y_pred_cv)
        nmae = np.mean(np.abs(y_val_cv - y_pred_cv) / (np.abs(y_val_cv) + 1e-8))
        
        xgb_scores.append({'mae': mae, 'mse': mse, 'nmae': nmae})

    avg_mae = np.mean([s['mae'] for s in xgb_scores])
    avg_mse = np.mean([s['mse'] for s in xgb_scores])
    avg_nmae = np.mean([s['nmae'] for s in xgb_scores])

    results.append({
        'model': 'XGBoost',
        'mae': avg_mae,
        'mse': avg_mse,
        'rmse': np.sqrt(avg_mse),
        'nmae': avg_nmae
    })

    print(f"  MAE: {avg_mae:.2f}, RMSE: {np.sqrt(avg_mse):.2f}, NMAE: {avg_nmae:.4f}")
else:
    print("\n[4/4] XGBoost 스킵 (설치되지 않음)")

# 6. 모델 성능 비교
print("\n" + "="*70)
print("모델 성능 비교 (원본 데이터 기준)")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('nmae')

print(results_df.to_string(index=False))

print("\n" + "="*70)
print("최고 성능 모델 (NMAE 기준):", results_df.iloc[0]['model'])
print(f"  - NMAE: {results_df.iloc[0]['nmae']:.4f}")
print(f"  - MAE: {results_df.iloc[0]['mae']:.2f}")
print(f"  - RMSE: {results_df.iloc[0]['rmse']:.2f}")

# 7. 최고 성능 모델로 예측 및 제출 파일 생성
best_model_name = results_df.iloc[0]['model']
best_model = models[best_model_name]

print(f"\n[6단계] 최고 성능 모델({best_model_name})로 예측...")

def predict(pivot, pairs, model, scaler, monthly_features):
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
        cluster_boost = float(row.cluster_boost)
        same_cluster = int(row.same_cluster)
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

        # 원본 데이터 특성
        current_ym = months[t_last]
        try:
            follower_features = monthly_features[
                (monthly_features['item_id'] == follower) & (monthly_features['ym'] == current_ym)
            ]

            if len(follower_features) > 0:
                follower_value_sum = follower_features['value_sum'].values[0] if 'value_sum' in follower_features.columns else 0
                follower_log_value = np.log1p(follower_value_sum)
                follower_has_quantity = follower_features['has_quantity_this_row'].values[0] if 'has_quantity_this_row' in follower_features.columns else 0
            else:
                follower_value_sum = 0
                follower_log_value = 0
                follower_has_quantity = 0
        except (KeyError, IndexError):
            follower_value_sum = 0
            follower_log_value = 0
            follower_has_quantity = 0

        X_test = np.array([[b_t, b_t_1, a_t_lag, corr, float(lag),
                           cluster_boost, float(same_cluster), float(same_hs4),
                           follower_log_value, follower_value_sum, float(follower_has_quantity)]])
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)[0]

        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred


# 결과 디렉토리 생성
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

submission = predict(pivot, pairs, best_model, scaler, monthly_features)
submission.to_csv(str(RESULTS_DIR / 'original_data_submit.csv'), index=False)

print(f"\n제출 파일 생성 완료: {RESULTS_DIR / 'original_data_submit.csv'}")
print(f"예측된 공행성쌍 수: {len(submission)}")
print(f"사용된 모델: {best_model_name}")
print(f"\n제출 파일 샘플:")
print(submission.head(10))

