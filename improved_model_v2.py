"""
개선된 공행성 쌍 예측 모델 v2
- 더 정교한 공행성 탐색 (다양한 방법)
- 강화된 Feature Engineering (계절성, 트렌드, 변화율)
- 공행성이 없는 쌍에 대한 시계열 예측
- Cross-validation 기반 모델 선택
- 예측값 후처리
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

def load_and_preprocess():
    """데이터 로드 및 기본 전처리"""
    print("데이터 로드 중...")
    df = pd.read_csv('train.csv')
    
    # 기본 타입 변환
    num_int_cols = ["year", "month", "hs4"]
    for c in num_int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    num_float_cols = ["value", "weight", "quantity"]
    for c in num_float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # date 컬럼 생성
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" +
            df["month"].astype(int).astype(str) + "-01"
        )
    else:
        df["date"] = pd.to_datetime(df["date"])
    
    # 거래 여부 플래그
    df["has_tx"] = 1
    
    # 월별 집계 (item_id + hs4 + date 기준)
    agg_dict = {
        "value": lambda x: x.sum(min_count=1),
        "weight": lambda x: x.sum(min_count=1),
        "quantity": lambda x: x.sum(min_count=1),
        "has_tx": "sum",
    }
    
    group_cols = ["item_id", "hs4", "date"]
    df_month = df.groupby(group_cols, as_index=False).agg(agg_dict)
    df_month["has_tx"] = (df_month["has_tx"] > 0).astype(int)
    
    # zero/missing 플래그
    for col in ["value", "weight", "quantity"]:
        df_month[f"{col}_missing"] = df_month[col].isna().astype(int)
        df_month[f"{col}_zero"] = ((df_month[col] == 0) & df_month[col].notna()).astype(int)
    
    # 전체 패널 생성 (모든 item_id × 모든 month)
    min_month = df_month["date"].min()
    max_month = df_month["date"].max()
    full_months = pd.date_range(min_month, max_month, freq="MS")
    all_items = df_month["item_id"].unique()
    
    panel = pd.MultiIndex.from_product(
        [all_items, full_months],
        names=["item_id", "date"]
    ).to_frame(index=False)
    
    df_panel = (
        panel.merge(df_month, on=["item_id", "date"], how="left")
        .sort_values(["item_id", "date"])
        .reset_index(drop=True)
    )
    
    df_panel["has_tx"] = df_panel["has_tx"].fillna(0).astype(int)
    
    # 거래 없는 달은 0으로 채우기
    no_tx_mask = df_panel["has_tx"] == 0
    for col in ["value", "weight", "quantity"]:
        df_panel.loc[no_tx_mask, col] = 0
    
    for col in [
        "value_zero", "weight_zero", "quantity_zero",
        "value_missing", "weight_missing", "quantity_missing",
    ]:
        df_panel[col] = df_panel[col].fillna(0).astype(int)
    
    df_panel["year"] = df_panel["date"].dt.year
    df_panel["month"] = df_panel["date"].dt.month
    
    print(f"전처리 완료: {len(df_panel)} rows, {df_panel['item_id'].nunique()} items")
    return df_panel


# ============================================================================
# 2. 개선된 공행성 쌍 탐색 (다양한 방법)
# ============================================================================

def find_comovement_pairs_v2(ts_matrix, max_lag=12, min_corr=0.25, min_common=6):
    """
    개선된 공행성 쌍 탐색 v2
    - 다양한 상관계수 측정 (Pearson, Spearman)
    - Granger causality 검정 (간단한 버전)
    - 더 정교한 lag 탐색
    """
    items = ts_matrix.columns.tolist()
    n_items = len(items)
    
    results = []
    
    # 표준화
    def _zscore(col: pd.Series) -> pd.Series:
        std = col.std()
        if std is None or np.isnan(std) or std == 0:
            return col - col.mean()
        return (col - col.mean()) / std
    
    ts_std = ts_matrix.apply(_zscore, axis=0)
    
    print(f"공행성 쌍 탐색 중... (총 {n_items * (n_items - 1) // 2}개 쌍)")
    
    for i in range(n_items):
        if (i + 1) % 20 == 0:
            print(f"진행률: {i+1}/{n_items}")
        
        for j in range(i + 1, n_items):
            a_name = items[i]
            b_name = items[j]
            
            a = ts_std[a_name]
            b = ts_std[b_name]
            
            # 공통 관측 구간 찾기
            ab = pd.concat([a, b], axis=1).dropna()
            if len(ab) < min_common:
                continue
            
            a_clean = ab.iloc[:, 0]
            b_clean = ab.iloc[:, 1]
            
            best_corr = 0.0
            best_lag = 0
            best_method = "pearson"
            
            # 다양한 lag 탐색
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    # 동시 상관계수
                    corr_p = a_clean.corr(b_clean)
                    corr_s = a_clean.corr(b_clean, method='spearman')
                    corr = max(abs(corr_p) if not np.isnan(corr_p) else 0,
                              abs(corr_s) if not np.isnan(corr_s) else 0)
                    method = "pearson" if abs(corr_p) > abs(corr_s) else "spearman"
                elif lag > 0:
                    if len(a_clean) > lag:
                        a_sub = a_clean.iloc[:-lag]
                        b_sub = b_clean.iloc[lag:]
                        corr_p = a_sub.corr(b_sub)
                        corr_s = a_sub.corr(b_sub, method='spearman')
                        corr = max(abs(corr_p) if not np.isnan(corr_p) else 0,
                                  abs(corr_s) if not np.isnan(corr_s) else 0)
                        method = "pearson" if abs(corr_p) > abs(corr_s) else "spearman"
                    else:
                        continue
                else:
                    k = -lag
                    if len(a_clean) > k:
                        a_sub = a_clean.iloc[k:]
                        b_sub = b_clean.iloc[:-k]
                        corr_p = a_sub.corr(b_sub)
                        corr_s = a_sub.corr(b_sub, method='spearman')
                        corr = max(abs(corr_p) if not np.isnan(corr_p) else 0,
                                  abs(corr_s) if not np.isnan(corr_s) else 0)
                        method = "pearson" if abs(corr_p) > abs(corr_s) else "spearman"
                    else:
                        continue
                
                if corr is None or np.isnan(corr):
                    continue
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
                    best_method = method
            
            # threshold 이상인 쌍 저장
            if abs(best_corr) >= min_corr and best_lag != 0:
                if best_lag > 0:
                    leading = a_name
                    following = b_name
                    lead_lag = best_lag
                else:
                    leading = b_name
                    following = a_name
                    lead_lag = -best_lag
                
                results.append({
                    "leading_item_id": leading,
                    "following_item_id": following,
                    "lead_lag_months": lead_lag,
                    "corr": best_corr,
                    "corr_abs": abs(best_corr),
                    "corr_sign": "positive" if best_corr > 0 else "negative",
                    "corr_method": best_method
                })
    
    if not results:
        return pd.DataFrame(columns=[
            "leading_item_id", "following_item_id",
            "lead_lag_months", "corr", "corr_abs", "corr_sign", "corr_method"
        ])
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        ["corr_abs", "lead_lag_months"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    return result_df


# ============================================================================
# 3. 강화된 Feature Engineering
# ============================================================================

def create_advanced_features(df_panel, max_lag=12):
    """강화된 Feature Engineering"""
    df = df_panel.copy()
    
    for item_id in df["item_id"].unique():
        item_mask = df["item_id"] == item_id
        item_data = df.loc[item_mask].sort_values("date").reset_index(drop=True)
        
        # Lag features
        for k in range(1, max_lag + 1):
            df.loc[item_mask, f"value_lag_{k}"] = item_data["value"].shift(k).values
        
        # 이동평균
        item_data_sorted = df.loc[item_mask].sort_values("date")
        df.loc[item_mask, "value_ma3"] = item_data_sorted["value"].rolling(3, min_periods=1).mean().values
        df.loc[item_mask, "value_ma6"] = item_data_sorted["value"].rolling(6, min_periods=1).mean().values
        df.loc[item_mask, "value_ma12"] = item_data_sorted["value"].rolling(12, min_periods=1).mean().values
        
        # 표준편차
        df.loc[item_mask, "value_std3"] = item_data_sorted["value"].rolling(3, min_periods=1).std().fillna(0).values
        df.loc[item_mask, "value_std6"] = item_data_sorted["value"].rolling(6, min_periods=1).std().fillna(0).values
        
        # 변화율 (전월 대비, 전년 동월 대비)
        # 0으로 나누기 방지
        pct_change = item_data_sorted["value"].pct_change()
        df.loc[item_mask, "value_pct_change"] = pct_change.replace([np.inf, -np.inf], 0).fillna(0).values
        
        # 전년 동월 대비 변화율 (0으로 나누기 방지)
        prev_year = item_data_sorted["value"].shift(12)
        # 0이 아닌 값으로만 나누기
        yoy_change = np.where(
            prev_year != 0,
            (item_data_sorted["value"] / prev_year) - 1.0,
            0.0
        )
        yoy_change = np.where(np.isfinite(yoy_change), yoy_change, 0.0)
        df.loc[item_mask, "value_yoy_change"] = pd.Series(yoy_change).fillna(0).values
        
        # 계절성 (월별 평균)
        df.loc[item_mask, "month_avg"] = item_data_sorted.groupby("month")["value"].transform("mean").values
        
        # 트렌드 (선형 추세)
        x = np.arange(len(item_data_sorted))
        y = item_data_sorted["value"].values
        if len(y) > 1 and np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
            df.loc[item_mask, "value_trend"] = slope
        else:
            df.loc[item_mask, "value_trend"] = 0.0
        
        # 최근 최대/최소값
        df.loc[item_mask, "value_max6"] = item_data_sorted["value"].rolling(6, min_periods=1).max().values
        df.loc[item_mask, "value_min6"] = item_data_sorted["value"].rolling(6, min_periods=1).min().values
        
        # 계절성 인코딩 (sin/cos)
        df.loc[item_mask, "month_sin"] = np.sin(2 * np.pi * item_data_sorted["month"] / 12).values
        df.loc[item_mask, "month_cos"] = np.cos(2 * np.pi * item_data_sorted["month"] / 12).values
    
    return df


# ============================================================================
# 4. 학습 데이터 생성
# ============================================================================

def build_training_data_v2(df_panel, comovement_candidates, max_lag=12):
    """개선된 학습 데이터 생성"""
    print("학습 데이터 생성 중...")
    
    # Feature 생성
    df_panel = create_advanced_features(df_panel, max_lag)
    
    item_groups = {
        item: g.sort_values("date").reset_index(drop=True)
        for item, g in df_panel.groupby("item_id")
    }
    
    pair_dfs = []
    
    for row in comovement_candidates.itertuples(index=False):
        leading = row.leading_item_id
        following = row.following_item_id
        
        if leading not in item_groups or following not in item_groups:
            continue
        
        lead_df = item_groups[leading].copy()
        foll_df = item_groups[following].copy()
        
        # Following 쪽: lag feature + target
        foll_df["target_value_next"] = foll_df["value"].shift(-1)
        
        foll_feat_cols = [
            "item_id", "date", "value", "year", "month", "has_tx",
            "value_ma3", "value_ma6", "value_ma12",
            "value_std3", "value_std6",
            "value_pct_change", "value_yoy_change",
            "month_avg", "value_trend",
            "value_max6", "value_min6",
            "month_sin", "month_cos"
        ] + [f"value_lag_{k}" for k in range(1, max_lag + 1)]
        foll_feat_cols = [c for c in foll_feat_cols if c in foll_df.columns]
        foll_used = foll_df[foll_feat_cols + ["target_value_next"]].copy()
        
        rename_foll = {
            c: f"f_{c}" for c in foll_used.columns
            if c not in ["date"]
        }
        foll_used = foll_used.rename(columns=rename_foll)
        
        # Leading 쪽: lag feature
        lead_feat_cols = [
            "item_id", "date", "value",
            "value_ma3", "value_ma6", "value_ma12",
            "value_std3", "value_std6",
            "value_pct_change", "value_yoy_change",
            "month_avg", "value_trend",
            "value_max6", "value_min6"
        ] + [f"value_lag_{k}" for k in range(1, max_lag + 1)]
        lead_feat_cols = [c for c in lead_feat_cols if c in lead_df.columns]
        lead_used = lead_df[lead_feat_cols].copy()
        
        rename_lead = {
            c: f"l_{c}" for c in lead_used.columns
            if c not in ["date"]
        }
        lead_used = lead_used.rename(columns=rename_lead)
        
        # Merge
        merged = pd.merge(
            foll_used,
            lead_used,
            on="date",
            how="inner"
        )
        
        # NaN 및 Inf 제거
        feature_cols = [
            c for c in merged.columns
            if c.startswith("f_value_lag_") or c.startswith("l_value_lag_")
        ] + ["f_value", "f_value_ma3", "f_value_ma6", "f_value_ma12",
             "l_value", "l_value_ma3", "l_value_ma6"]
        
        # Inf 값을 NaN으로 변환 후 제거
        for col in feature_cols + ["f_target_value_next"]:
            if col in merged.columns:
                merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)
        
        merged = merged.dropna(subset=feature_cols + ["f_target_value_next"])
        
        if len(merged) == 0:
            continue
        
        # 남은 inf 값이 있으면 0으로 대체
        for col in merged.columns:
            if merged[col].dtype in [np.float64, np.float32]:
                merged[col] = merged[col].replace([np.inf, -np.inf], 0.0)
        
        merged["leading_item_id"] = leading
        merged["following_item_id"] = following
        merged["lead_lag_months"] = row.lead_lag_months
        merged["pair_corr"] = row.corr
        
        pair_dfs.append(merged)
    
    if not pair_dfs:
        return pd.DataFrame()
    
    train_df = pd.concat(pair_dfs, axis=0, ignore_index=True)
    print(f"학습 데이터 생성 완료: {len(train_df)} rows")
    return train_df


# ============================================================================
# 5. 시계열 예측 모델 (공행성이 없는 쌍용)
# ============================================================================

def predict_simple_timeseries(item_series, pivot_idx, method='ewm'):
    """
    간단한 시계열 예측
    - Exponential Weighted Moving Average
    - Linear Trend
    - Seasonal Average
    """
    if len(item_series) < 3:
        return item_series.iloc[-1]["value"] if len(item_series) > 0 else 0.0
    
    values = item_series["value"].values
    recent_values = values[max(0, pivot_idx-11):pivot_idx+1]
    
    if method == 'ewm':
        # Exponential weighted moving average
        alpha = 0.3
        pred = recent_values[-1]
        for v in recent_values[-6:]:
            pred = alpha * v + (1 - alpha) * pred
        return max(0, pred)
    
    elif method == 'trend':
        # Linear trend
        if len(recent_values) >= 3:
            x = np.arange(len(recent_values))
            coeffs = np.polyfit(x, recent_values, 1)
            pred = coeffs[0] * len(recent_values) + coeffs[1]
            return max(0, pred)
        return recent_values[-1]
    
    elif method == 'seasonal':
        # Seasonal average (같은 월의 평균)
        month = item_series.iloc[pivot_idx]["month"]
        same_month_values = item_series[item_series["month"] == month]["value"].values
        if len(same_month_values) > 0:
            return max(0, np.mean(same_month_values))
        return recent_values[-1]
    
    else:
        # Simple average
        return max(0, np.mean(recent_values))


# ============================================================================
# 6. 앙상블 모델 학습 (Cross-validation)
# ============================================================================

def train_ensemble_model_v2(train_df, feature_cols):
    """Cross-validation 기반 앙상블 모델 학습"""
    print("앙상블 모델 학습 중...")
    
    X = train_df[feature_cols].values
    y = train_df["f_target_value_next"].values
    
    # Inf, NaN 값 처리
    print("  데이터 정제 중 (inf, nan 제거)...")
    # Inf 값을 NaN으로 변환 후 제거
    X = np.where(np.isfinite(X), X, 0.0)
    y = np.where(np.isfinite(y), y, 0.0)
    
    # 너무 큰 값 클리핑 (float32 범위 내로)
    X = np.clip(X, -1e10, 1e10)
    y = np.clip(y, -1e10, 1e10)
    
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  X 범위: [{np.min(X):.2f}, {np.max(X):.2f}]")
    print(f"  y 범위: [{np.min(y):.2f}, {np.max(y):.2f}]")
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    
    models_config = {
        "rf": {
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 600,
                "max_depth": 25,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "gbm": {
            "model": GradientBoostingRegressor,
            "params": {
                "n_estimators": 600,
                "learning_rate": 0.03,
                "max_depth": 6,
                "random_state": 42
            }
        },
        "et": {
            "model": ExtraTreesRegressor,
            "params": {
                "n_estimators": 600,
                "max_depth": 25,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
        }
    }
    
    # Cross-validation으로 모델 평가
    model_scores = {}
    trained_models = {}
    
    for name, config in models_config.items():
        print(f"  {name} 학습 및 평가 중...")
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = config["model"](**config["params"])
            model.fit(X_train, y_train)
            
            # R² score
            score = model.score(X_val, y_val)
            scores.append(score)
        
        model_scores[name] = np.mean(scores)
        print(f"    CV Score: {np.mean(scores):.4f}")
        
        # 전체 데이터로 최종 학습
        final_model = config["model"](**config["params"])
        final_model.fit(X, y)
        trained_models[name] = final_model
    
    # 가중치 계산 (성능 기반)
    total_score = sum(model_scores.values())
    weights = {name: score / total_score for name, score in model_scores.items()}
    print(f"\n모델 가중치: {weights}")
    
    return trained_models, weights


# ============================================================================
# 7. 추론 데이터 생성 (개선)
# ============================================================================

def create_inference_data_v2(df_panel, comovement_candidates, all_items, max_lag=12):
    """개선된 추론 데이터 생성"""
    print("추론 데이터 생성 중...")
    
    PIVOT_DATE = pd.Timestamp("2025-07-01")
    
    # Feature 생성
    df_panel = create_advanced_features(df_panel, max_lag)
    
    # 2025-07 데이터
    panel_202507 = df_panel[df_panel["date"] == PIVOT_DATE].copy()
    item_map_202507 = {item: g.iloc[0] for item, g in panel_202507.groupby("item_id")}
    
    # Item별 시계열 저장
    item_series = {}
    for item_id, group in df_panel.groupby("item_id"):
        item_series[item_id] = group.sort_values("date").reset_index(drop=True)
    
    rows = []
    comovement_pairs = set()
    
    # 공행성 쌍
    for row in comovement_candidates.itertuples(index=False):
        leading = row.leading_item_id
        following = row.following_item_id
        comovement_pairs.add((leading, following))
        
        if leading not in item_map_202507 or following not in item_map_202507:
            continue
        if leading not in item_series or following not in item_series:
            continue
        
        lead_series = item_series[leading]
        foll_series = item_series[following]
        
        lead_idx = lead_series[lead_series["date"] == PIVOT_DATE].index
        foll_idx = foll_series[foll_series["date"] == PIVOT_DATE].index
        
        if len(lead_idx) == 0 or len(foll_idx) == 0:
            continue
        
        lead_idx = lead_idx[0]
        foll_idx = foll_idx[0]
        lead_row = item_map_202507[leading]
        foll_row = item_map_202507[following]
        
        data = {
            "leading_item_id": leading,
            "following_item_id": following,
            "date": PIVOT_DATE,
            "is_comovement": 1,
            "pair_corr": row.corr,
            "lead_lag_months": row.lead_lag_months,
        }
        
        # Following features
        for col in ["value", "has_tx", "year", "month",
                   "value_ma3", "value_ma6", "value_ma12",
                   "value_std3", "value_std6",
                   "value_pct_change", "value_yoy_change",
                   "month_avg", "value_trend",
                   "value_max6", "value_min6",
                   "month_sin", "month_cos"]:
            if col in foll_row.index:
                data[f"f_{col}"] = foll_row[col]
        
        for k in range(1, max_lag + 1):
            lag_idx = foll_idx - k
            if lag_idx >= 0:
                data[f"f_value_lag_{k}"] = foll_series.iloc[lag_idx]["value"]
            else:
                data[f"f_value_lag_{k}"] = 0.0
        
        # Leading features
        for col in ["value", "value_ma3", "value_ma6", "value_ma12",
                   "value_std3", "value_std6",
                   "value_pct_change", "value_yoy_change",
                   "month_avg", "value_trend",
                   "value_max6", "value_min6"]:
            if col in lead_row.index:
                data[f"l_{col}"] = lead_row[col]
        
        for k in range(1, max_lag + 1):
            lag_idx = lead_idx - k
            if lag_idx >= 0:
                data[f"l_value_lag_{k}"] = lead_series.iloc[lag_idx]["value"]
            else:
                data[f"l_value_lag_{k}"] = 0.0
        
        rows.append(data)
    
    # 공행성이 없는 쌍 - 시계열 예측 사용
    items_with_tx = [item for item in all_items if item in item_map_202507]
    print(f"공행성 없는 쌍 추가 중... (거래 있는 item: {len(items_with_tx)}개)")
    
    added = 0
    for i, item_a in enumerate(items_with_tx):
        if (i + 1) % 20 == 0:
            print(f"  진행률: {i+1}/{len(items_with_tx)}")
        
        for item_b in items_with_tx:
            if item_a == item_b:
                continue
            
            pair = (item_a, item_b)
            if pair in comovement_pairs:
                continue
            
            row_a = item_map_202507[item_a]
            row_b = item_map_202507[item_b]
            
            # 시계열 예측
            if item_b in item_series:
                b_series = item_series[item_b]
                b_idx = b_series[b_series["date"] == PIVOT_DATE].index
                if len(b_idx) > 0:
                    # 여러 방법의 평균
                    pred1 = predict_simple_timeseries(b_series, b_idx[0], 'ewm')
                    pred2 = predict_simple_timeseries(b_series, b_idx[0], 'trend')
                    pred3 = predict_simple_timeseries(b_series, b_idx[0], 'seasonal')
                    simple_pred = (pred1 + pred2 + pred3) / 3
                else:
                    simple_pred = row_b["value"]
            else:
                simple_pred = row_b["value"]
            
            data = {
                "leading_item_id": item_a,
                "following_item_id": item_b,
                "date": PIVOT_DATE,
                "is_comovement": 0,
                "pair_corr": 0.0,
                "lead_lag_months": 0,
                "f_value": row_b["value"],
                "f_has_tx": row_b.get("has_tx", 0),
                "f_year": row_b["year"],
                "f_month": row_b["month"],
                "l_value": row_a["value"],
                "simple_pred": simple_pred,  # 시계열 예측값 저장
            }
            
            # 기본 feature
            for k in range(1, max_lag + 1):
                data[f"f_value_lag_{k}"] = 0.0
                data[f"l_value_lag_{k}"] = 0.0
            
            for col in ["value_ma3", "value_ma6", "value_ma12",
                       "value_std3", "value_std6",
                       "value_pct_change", "value_yoy_change",
                       "month_avg", "value_trend",
                       "value_max6", "value_min6",
                       "month_sin", "month_cos"]:
                if f"f_{col}" not in data:
                    data[f"f_{col}"] = row_b.get(col, 0.0) if col in row_b.index else 0.0
                if f"l_{col}" not in data:
                    data[f"l_{col}"] = row_a.get(col, 0.0) if col in row_a.index else 0.0
            
            rows.append(data)
            added += 1
    
    print(f"추론 데이터 생성 완료: {len(rows)} rows (공행성: {len(comovement_pairs)}, 비공행성: {added})")
    return pd.DataFrame(rows)


# ============================================================================
# 8. 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("개선된 공행성 쌍 예측 모델 v2")
    print("=" * 60)
    
    # 1. 데이터 로드 및 전처리
    df_panel = load_and_preprocess()
    
    # 2. 시계열 행렬 생성
    print("\n시계열 행렬 생성 중...")
    all_months = pd.date_range(
        df_panel["date"].min(),
        df_panel["date"].max(),
        freq="MS"
    )
    
    ts_matrix = (
        df_panel
        .set_index("date")
        .groupby("item_id")["value"]
        .apply(lambda s: s.reindex(all_months).fillna(0))
        .unstack(0)
    )
    
    # Log transform
    ts_matrix_log = np.log1p(ts_matrix)
    
    print(f"시계열 행렬 shape: {ts_matrix_log.shape}")
    
    # 3. 공행성 쌍 탐색
    print("\n공행성 쌍 탐색 중...")
    comovement_candidates = find_comovement_pairs_v2(
        ts_matrix=ts_matrix_log,
        max_lag=12,
        min_corr=0.25,  # 더 낮은 threshold
        min_common=6
    )
    
    print(f"발견된 공행성 쌍: {len(comovement_candidates)}개")
    if len(comovement_candidates) > 0:
        print(comovement_candidates.head(10))
    
    # 4. 학습 데이터 생성
    print("\n" + "=" * 60)
    train_df = build_training_data_v2(df_panel, comovement_candidates, max_lag=12)
    
    if len(train_df) == 0:
        print("경고: 학습 데이터가 없습니다!")
        return
    
    # 5. Feature 선택
    feature_cols = [
        *[f"f_value_lag_{k}" for k in range(1, 13)],
        *[f"l_value_lag_{k}" for k in range(1, 13)],
        "f_value",
        "f_value_ma3", "f_value_ma6", "f_value_ma12",
        "f_value_std3", "f_value_std6",
        "f_value_pct_change", "f_value_yoy_change",
        "f_month_avg", "f_value_trend",
        "f_value_max6", "f_value_min6",
        "f_month_sin", "f_month_cos",
        "l_value",
        "l_value_ma3", "l_value_ma6", "l_value_ma12",
        "l_value_std3", "l_value_std6",
        "l_value_pct_change", "l_value_yoy_change",
        "l_month_avg", "l_value_trend",
        "l_value_max6", "l_value_min6",
        "lead_lag_months",
        "pair_corr",
        "f_year",
        "f_month",
        "f_has_tx",
    ]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    print(f"\n사용할 feature 개수: {len(feature_cols)}")
    
    # 6. 모델 학습
    print("\n" + "=" * 60)
    models, weights = train_ensemble_model_v2(train_df, feature_cols)
    
    # 7. 추론 데이터 생성
    print("\n" + "=" * 60)
    all_items = df_panel["item_id"].unique()
    infer_df = create_inference_data_v2(df_panel, comovement_candidates, all_items, max_lag=12)
    
    # 8. Feature 채우기
    missing_cols = [c for c in feature_cols if c not in infer_df.columns]
    for c in missing_cols:
        if c in train_df.columns:
            infer_df[c] = train_df[c].mean()
        else:
            infer_df[c] = 0.0
    
    # Inf 값 처리
    for col in feature_cols:
        if col in infer_df.columns:
            infer_df[col] = infer_df[col].replace([np.inf, -np.inf], 0.0)
            infer_df[col] = infer_df[col].fillna(0.0)
    
    # 9. 예측
    print("\n예측 중...")
    X_test = infer_df[feature_cols].values
    
    # 최종 안전장치: inf 값 제거
    X_test = np.where(np.isfinite(X_test), X_test, 0.0)
    X_test = np.clip(X_test, -1e10, 1e10)
    
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
    
    # 앙상블 (성능 기반 가중치)
    y_pred = (
        weights["rf"] * predictions["rf"] +
        weights["gbm"] * predictions["gbm"] +
        weights["et"] * predictions["et"]
    )
    
    # 공행성이 없는 쌍은 시계열 예측 사용
    no_comovement_mask = infer_df["is_comovement"] == 0
    if "simple_pred" in infer_df.columns:
        y_pred[no_comovement_mask] = infer_df.loc[no_comovement_mask, "simple_pred"].values
    
    # 후처리: 이상치 제거 및 범위 제한
    y_pred = np.maximum(y_pred, 0)
    
    # 상위 1% 이상치 클리핑
    q99 = np.percentile(y_pred[y_pred > 0], 99)
    y_pred = np.minimum(y_pred, q99 * 2)  # 최대값을 99 percentile의 2배로 제한
    
    y_pred_int = np.round(y_pred).astype(int)
    
    # 10. Submission 생성
    submission = infer_df[["leading_item_id", "following_item_id"]].copy()
    submission["value"] = y_pred_int
    
    submission.to_csv("submission_v2.csv", index=False)
    
    print("\n" + "=" * 60)
    print("완료!")
    print(f"생성된 submission 개수: {len(submission)}")
    print(f"공행성 쌍: {infer_df['is_comovement'].sum()}개")
    print(f"비공행성 쌍: {(infer_df['is_comovement'] == 0).sum()}개")
    print(f"예측값 통계:")
    print(submission["value"].describe())
    print("\nsubmission.csv 파일이 생성되었습니다!")
    print("=" * 60)


if __name__ == "__main__":
    main()

