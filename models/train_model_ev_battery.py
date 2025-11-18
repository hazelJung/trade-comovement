"""
preprocess_ev_battery.ipynb 기준 공행성 쌍 예측 모델
- EV 배터리 전처리 결과(monthly_features)를 사용
- 클러스터 정보 및 보정된 값 활용
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'submissions'
ANALYSIS_DATA_DIR = PROJECT_ROOT / 'analysis' / 'data'

# 전처리 모듈 import
sys.path.append(str(PROJECT_ROOT / 'analysis'))
from preprocess_ev_battery import main as preprocess_main

print("="*70)
print("preprocess_ev_battery.ipynb 기준 공행성 쌍 예측 모델")
print("="*70)

# 1. 전처리된 데이터 로드
print("\n[1단계] 전처리된 데이터 로드...")
try:
    # 전처리 결과 파일이 있으면 로드
    monthly_features_path = ANALYSIS_DATA_DIR / 'monthly_features.csv'
    monthly_features = pd.read_csv(str(monthly_features_path))
    print(f"전처리 결과 로드 완료: {len(monthly_features):,}행")
except FileNotFoundError:
    print("전처리 결과 파일이 없습니다. 전처리를 실행합니다...")
    monthly_features = preprocess_main()
    print(f"전처리 완료: {len(monthly_features):,}행")

# 2. Pivot 테이블 생성 (보정된 값 사용)
print("\n[2단계] Pivot 테이블 생성...")
# value_filled 또는 value_clip 사용 (이상치 처리된 버전)
pivot = (
    monthly_features.pivot_table(
        index='item_id',
        columns='ym',
        values='value_filled',  # 보정된 값 사용
        aggfunc='sum',
        fill_value=0.0
    )
)

print(f"Pivot table shape: {pivot.shape}")

# item_id별 메타 정보
item_info = monthly_features[['item_id', 'hs4', 'hs2', 'cluster']].drop_duplicates().set_index('item_id')
item_hs4 = item_info['hs4'].to_dict()
item_cluster = item_info['cluster'].to_dict()

# 3. 공행성쌍 탐색 (EV 배터리 클러스터 고려)
def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def get_cluster_boost(leader_cluster, follower_cluster):
    """EV 배터리 클러스터 기반 가중치"""
    # 같은 클러스터 내에서 높은 가중치
    if leader_cluster == follower_cluster:
        if leader_cluster == 'battery_materials':
            return 2.0  # 배터리 소재 간 높은 연관성
        elif leader_cluster == 'ev_machinery_electrical':
            return 1.8  # EV 기기 간 연관성
        else:
            return 1.2  # 기타 산업 간 연관성
    
    # 배터리 소재 → EV 기기 (공급망 관계)
    if leader_cluster == 'battery_materials' and follower_cluster == 'ev_machinery_electrical':
        return 1.5
    
    return 1.0  # 기본값


def find_comovement_pairs_ev_battery(pivot, item_hs4, item_cluster, 
                                     max_lag=6, min_nonzero=12, corr_threshold=0.4):
    """EV 배터리 클러스터를 고려한 공행성쌍 탐색"""
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items), desc="Finding comovement pairs (EV battery)"):
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

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                corr = safe_corr(x[:-lag], y[lag:])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            # 클러스터 가중치 적용
            cluster_boost = get_cluster_boost(leader_cluster, follower_cluster)
            effective_corr = abs(best_corr) * cluster_boost
            effective_threshold = corr_threshold / cluster_boost

            # 이중 필터링: 최소 상관계수 + 가중치 적용 임계값
            # 너무 낮은 상관계수는 제외 (노이즈 방지)
            min_abs_corr = 0.25  # 최소 상관계수 조건
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


print("\n[3단계] 공행성쌍 탐색 (EV 배터리 클러스터 고려)...")
# 개선 1: max_pairs 튜닝으로 Precision 최적화 (선택사항)
# ev_battery 모델은 클러스터 가중치로 이미 많은 쌍을 찾으므로
# max_pairs 제한은 선택적으로 사용
MAX_PAIRS = None  # 튜닝 가능: 2500, 3000 등
pairs = find_comovement_pairs_ev_battery(pivot, item_hs4, item_cluster)
# 종합 점수는 없지만, 필요시 추가 가능
print(f"탐색된 공행성쌍 수: {len(pairs)}")
print(f"\n클러스터별 공행성쌍 분포:")
print(pairs.groupby(['leader_cluster', 'follower_cluster']).size())
print(f"\nPairs head:\n{pairs.head()}")

# 4. 학습 데이터 생성 (보정된 값 및 로그 변환 값 활용)
def build_training_data_ev_battery(pivot, pairs, monthly_features):
    """EV 배터리 전처리 특성을 활용한 학습 데이터 생성"""
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

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            # 해당 시점의 전처리 특성 추출
            current_ym = months[t]
            leader_features = monthly_features[
                (monthly_features['item_id'] == leader) & (monthly_features['ym'] == current_ym)
            ]
            follower_features = monthly_features[
                (monthly_features['item_id'] == follower) & (monthly_features['ym'] == current_ym)
            ]

            # 보정된 값 및 로그 변환 값 사용
            try:
                if len(leader_features) > 0 and 'log_value' in leader_features.columns:
                    leader_log_value = leader_features['log_value'].values[0]
                else:
                    leader_log_value = 0

                if len(follower_features) > 0:
                    follower_log_value = follower_features['log_value'].values[0] if 'log_value' in follower_features.columns else 0
                    follower_value_clip = follower_features['value_clip'].values[0] if 'value_clip' in follower_features.columns else 0
                    follower_has_quantity = follower_features['has_quantity_this_row'].values[0] if 'has_quantity_this_row' in follower_features.columns else 0
                else:
                    follower_log_value = 0
                    follower_value_clip = 0
                    follower_has_quantity = 0
            except (KeyError, IndexError):
                # 컬럼이 없거나 데이터가 없는 경우 기본값 사용
                leader_log_value = 0
                follower_log_value = 0
                follower_value_clip = 0
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
                "follower_value_clip": follower_value_clip,
                "follower_has_quantity": float(follower_has_quantity),
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train


print("\n[4단계] 학습 데이터 생성...")
df_train_model = build_training_data_ev_battery(pivot, pairs, monthly_features)
print(f'생성된 학습 데이터의 shape: {df_train_model.shape}')
print(f"Training data head:\n{df_train_model.head()}")

# 5. 회귀 모델 학습 (개선 버전)
print("\n[5단계] 회귀 모델 학습...")
from sklearn.linear_model import QuantileRegressor

feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag',
                'cluster_boost', 'same_cluster', 'same_hs4',
                'follower_log_value', 'follower_value_clip', 'follower_has_quantity']

train_X = df_train_model[feature_cols].values
train_y = df_train_model["target"].values

# inf, -inf, NaN 값 처리
train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)

# Feature scaling (NMAE 개선을 위해)
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

print(f"학습 데이터 shape: {train_X_scaled.shape}")
print(f"타겟 통계: mean={train_y.mean():.2f}, std={train_y.std():.2f}, min={train_y.min():.2f}, max={train_y.max():.2f}")

# ============================================================
# Cross-Validation으로 모델 성능 평가 (Time Series Split)
# ============================================================
print("\n" + "="*70)
print("Cross-Validation 평가 (Time Series Split)")
print("="*70)
print("주의: 시계열 데이터이므로 Time Series Split을 사용합니다.")
print("과거 데이터로 미래를 예측하는 방식으로 검증합니다.")
print("일반 K-Fold는 사용하지 않습니다 (미래 데이터로 과거를 예측하는 것은 불가능).\n")

tscv = TimeSeriesSplit(n_splits=3)

# 개선 2: NMAE 손실 함수 고려
USE_QUANTILE = False  # True로 설정하면 Quantile Regression 사용
RIDGE_ALPHA = 1.0  # 튜닝 가능: 0.1, 1.0, 10.0 등

cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(train_X_scaled), 1):
    X_train_cv, X_val_cv = train_X_scaled[train_idx], train_X_scaled[val_idx]
    y_train_cv, y_val_cv = train_y[train_idx], train_y[val_idx]
    
    if USE_QUANTILE:
        reg_cv = QuantileRegressor(quantile=0.5, alpha=RIDGE_ALPHA, solver='highs')
    else:
        reg_cv = Ridge(alpha=RIDGE_ALPHA)
    
    reg_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = reg_cv.predict(X_val_cv)
    
    mae = mean_absolute_error(y_val_cv, y_pred_cv)
    mse = mean_squared_error(y_val_cv, y_pred_cv)
    rmse = np.sqrt(mse)
    
    # NMAE 계산 (상대 오차)
    nmae = np.mean(np.abs(y_val_cv - y_pred_cv) / (np.abs(y_val_cv) + 1e-8))
    
    cv_scores.append({
        'fold': fold,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'nmae': nmae
    })
    
    print(f"Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, NMAE={nmae:.4f}")

# 평균 성능
avg_mae = np.mean([s['mae'] for s in cv_scores])
avg_rmse = np.mean([s['rmse'] for s in cv_scores])
avg_nmae = np.mean([s['nmae'] for s in cv_scores])

print("\n" + "="*70)
print("Cross-Validation 평균 성능")
print("="*70)
print(f"평균 MAE: {avg_mae:.2f}")
print(f"평균 RMSE: {avg_rmse:.2f}")
print(f"평균 NMAE: {avg_nmae:.4f}")

# ============================================================
# 최종 모델 학습 (전체 데이터)
# ============================================================
print("\n" + "="*70)
print("최종 모델 학습 (전체 데이터)")
print("="*70)

if USE_QUANTILE:
    # Quantile Regression (중앙값 = 0.5 quantile)
    reg = QuantileRegressor(quantile=0.5, alpha=RIDGE_ALPHA, solver='highs')
    print("  - Quantile Regression 사용 (중앙값 예측)")
else:
    # Ridge 회귀 (정규화로 과적합 방지)
    reg = Ridge(alpha=RIDGE_ALPHA)
    print(f"  - Ridge Regression 사용 (alpha={RIDGE_ALPHA})")

reg.fit(train_X_scaled, train_y)

print("Model training completed!")
print(f"\nFeature importance (coefficients):")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {reg.coef_[i]:.6f}")

# 6. 예측 및 제출 파일 생성
def predict_ev_battery(pivot, pairs, reg, monthly_features):
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

        # 전처리 특성
        current_ym = months[t_last]
        try:
            follower_features = monthly_features[
                (monthly_features['item_id'] == follower) & (monthly_features['ym'] == current_ym)
            ]

            if len(follower_features) > 0:
                follower_log_value = follower_features['log_value'].values[0] if 'log_value' in follower_features.columns else 0
                follower_value_clip = follower_features['value_clip'].values[0] if 'value_clip' in follower_features.columns else 0
                follower_has_quantity = follower_features['has_quantity_this_row'].values[0] if 'has_quantity_this_row' in follower_features.columns else 0
            else:
                follower_log_value = 0
                follower_value_clip = 0
                follower_has_quantity = 0
        except (KeyError, IndexError):
            # 컬럼이 없거나 데이터가 없는 경우 기본값 사용
            follower_log_value = 0
            follower_value_clip = 0
            follower_has_quantity = 0

        X_test = np.array([[b_t, b_t_1, a_t_lag, corr, float(lag),
                           cluster_boost, float(same_cluster), float(same_hs4),
                           follower_log_value, follower_value_clip, float(follower_has_quantity)]])
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
submission = predict_ev_battery(pivot, pairs, reg, monthly_features)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_DIR / 'ev_battery_submit.csv'
submission.to_csv(str(output_path), index=False)
print(f"제출 파일 생성 완료: {output_path}")
print(f"예측된 공행성쌍 수: {len(submission)}")

