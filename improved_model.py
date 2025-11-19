"""
개선된 공행성 쌍 예측 모델
- 더 많은 공행성 쌍 탐색
- 향상된 feature engineering
- 앙상블 모델 사용
- 모든 가능한 쌍에 대한 예측 생성
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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
# 2. 개선된 공행성 쌍 탐색
# ============================================================================

def find_comovement_pairs_improved(ts_matrix, max_lag=12, min_corr=0.3, min_common=6):
    """
    개선된 공행성 쌍 탐색
    - 더 넓은 lag 범위
    - 더 낮은 threshold
    - 다양한 상관계수 측정
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
            
            # 다양한 lag 탐색
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = a_clean.corr(b_clean)
                elif lag > 0:
                    if len(a_clean) > lag:
                        corr = a_clean.iloc[:-lag].corr(b_clean.iloc[lag:])
                    else:
                        continue
                else:
                    k = -lag
                    if len(a_clean) > k:
                        corr = a_clean.iloc[k:].corr(b_clean.iloc[:-k])
                    else:
                        continue
                
                if corr is None or np.isnan(corr):
                    continue
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            
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
                    "corr_sign": "positive" if best_corr > 0 else "negative"
                })
    
    if not results:
        return pd.DataFrame(columns=[
            "leading_item_id", "following_item_id",
            "lead_lag_months", "corr", "corr_abs", "corr_sign"
        ])
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        ["corr_abs", "lead_lag_months"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    return result_df


# ============================================================================
# 3. 향상된 Feature Engineering
# ============================================================================

def create_lag_features(df_panel, max_lag=12):
    """Lag feature 생성"""
    df = df_panel.copy()
    
    for item_id in df["item_id"].unique():
        item_mask = df["item_id"] == item_id
        item_data = df.loc[item_mask].sort_values("date").reset_index(drop=True)
        
        for k in range(1, max_lag + 1):
            df.loc[item_mask, f"value_lag_{k}"] = item_data["value"].shift(k).values
        
        # 이동평균, 트렌드 등 추가 feature
        item_data_sorted = df.loc[item_mask].sort_values("date")
        df.loc[item_mask, "value_ma3"] = item_data_sorted["value"].rolling(3, min_periods=1).mean().values
        df.loc[item_mask, "value_ma6"] = item_data_sorted["value"].rolling(6, min_periods=1).mean().values
        df.loc[item_mask, "value_std3"] = item_data_sorted["value"].rolling(3, min_periods=1).std().fillna(0).values
    
    return df


# ============================================================================
# 4. 학습 데이터 생성
# ============================================================================

def build_training_data(df_panel, comovement_candidates, max_lag=12):
    """학습 데이터 생성"""
    print("학습 데이터 생성 중...")
    
    # Lag feature 생성
    df_panel = create_lag_features(df_panel, max_lag)
    
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
            "value_ma3", "value_ma6", "value_std3"
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
            "value_ma3", "value_ma6", "value_std3"
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
        
        # NaN 제거
        feature_cols = [
            c for c in merged.columns
            if c.startswith("f_value_lag_") or c.startswith("l_value_lag_")
        ] + ["f_value", "f_value_ma3", "f_value_ma6"]
        
        merged = merged.dropna(subset=feature_cols + ["f_target_value_next"])
        
        if len(merged) == 0:
            continue
        
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
# 5. 앙상블 모델 학습
# ============================================================================

def train_ensemble_model(train_df, feature_cols):
    """앙상블 모델 학습"""
    print("앙상블 모델 학습 중...")
    
    X = train_df[feature_cols].values
    y = train_df["f_target_value_next"].values
    
    # 모델들
    models = {
        "rf": RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "gbm": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        "et": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # 학습
    trained_models = {}
    for name, model in models.items():
        print(f"  {name} 학습 중...")
        model.fit(X, y)
        trained_models[name] = model
    
    return trained_models


# ============================================================================
# 6. 추론 데이터 생성 (모든 가능한 쌍)
# ============================================================================

def create_inference_data(df_panel, comovement_candidates, all_items, max_lag=12):
    """추론 데이터 생성 - 모든 가능한 쌍 포함"""
    print("추론 데이터 생성 중...")
    
    PIVOT_DATE = pd.Timestamp("2025-07-01")
    
    # Lag feature 생성
    df_panel = create_lag_features(df_panel, max_lag)
    
    # 2025-07 데이터
    panel_202507 = df_panel[df_panel["date"] == PIVOT_DATE].copy()
    item_map_202507 = {item: g.iloc[0] for item, g in panel_202507.groupby("item_id")}
    
    # Item별 시계열 저장
    item_series = {}
    for item_id, group in df_panel.groupby("item_id"):
        item_series[item_id] = group.sort_values("date").reset_index(drop=True)
    
    rows = []
    
    # 공행성 쌍이 있는 경우
    comovement_pairs = set()
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
        data["f_value"] = foll_row["value"]
        data["f_has_tx"] = foll_row.get("has_tx", 0)
        data["f_year"] = foll_row["year"]
        data["f_month"] = foll_row["month"]
        
        for k in range(1, max_lag + 1):
            lag_idx = foll_idx - k
            if lag_idx >= 0:
                data[f"f_value_lag_{k}"] = foll_series.iloc[lag_idx]["value"]
            else:
                data[f"f_value_lag_{k}"] = 0.0
        
        # MA features
        if foll_idx >= 2:
            data["f_value_ma3"] = foll_series.iloc[max(0, foll_idx-2):foll_idx+1]["value"].mean()
        else:
            data["f_value_ma3"] = foll_row["value"]
        
        if foll_idx >= 5:
            data["f_value_ma6"] = foll_series.iloc[max(0, foll_idx-5):foll_idx+1]["value"].mean()
        else:
            data["f_value_ma6"] = foll_row["value"]
        
        # Leading features
        data["l_value"] = lead_row["value"]
        for k in range(1, max_lag + 1):
            lag_idx = lead_idx - k
            if lag_idx >= 0:
                data[f"l_value_lag_{k}"] = lead_series.iloc[lag_idx]["value"]
            else:
                data[f"l_value_lag_{k}"] = 0.0
        
        if lead_idx >= 2:
            data["l_value_ma3"] = lead_series.iloc[max(0, lead_idx-2):lead_idx+1]["value"].mean()
        else:
            data["l_value_ma3"] = lead_row["value"]
        
        if lead_idx >= 5:
            data["l_value_ma6"] = lead_series.iloc[max(0, lead_idx-5):lead_idx+1]["value"].mean()
        else:
            data["l_value_ma6"] = lead_row["value"]
        
        rows.append(data)
    
    # 공행성이 없는 쌍도 추가 (간단한 예측)
    # 모든 쌍을 생성하면 너무 많으므로, 2025-07에 거래가 있는 item만 사용
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
            
            # 간단한 예측을 위한 기본 feature만
            row_a = item_map_202507[item_a]
            row_b = item_map_202507[item_b]
            
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
            }
            
            # 기본 lag feature (0으로 채움)
            for k in range(1, max_lag + 1):
                data[f"f_value_lag_{k}"] = 0.0
                data[f"l_value_lag_{k}"] = 0.0
            
            data["f_value_ma3"] = row_b["value"]
            data["f_value_ma6"] = row_b["value"]
            data["l_value_ma3"] = row_a["value"]
            data["l_value_ma6"] = row_a["value"]
            
            rows.append(data)
            added += 1
    
    print(f"추론 데이터 생성 완료: {len(rows)} rows (공행성: {len(comovement_pairs)}, 비공행성: {added})")
    return pd.DataFrame(rows)


# ============================================================================
# 7. 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("개선된 공행성 쌍 예측 모델")
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
    
    # 3. 공행성 쌍 탐색 (다양한 threshold)
    print("\n공행성 쌍 탐색 중...")
    comovement_candidates = find_comovement_pairs_improved(
        ts_matrix=ts_matrix_log,
        max_lag=12,
        min_corr=0.3,  # 낮은 threshold로 더 많은 쌍 탐색
        min_common=6
    )
    
    print(f"발견된 공행성 쌍: {len(comovement_candidates)}개")
    if len(comovement_candidates) > 0:
        print(comovement_candidates.head(10))
    
    # 4. 학습 데이터 생성
    print("\n" + "=" * 60)
    train_df = build_training_data(df_panel, comovement_candidates, max_lag=12)
    
    if len(train_df) == 0:
        print("경고: 학습 데이터가 없습니다!")
        return
    
    # 5. Feature 선택
    feature_cols = [
        *[f"f_value_lag_{k}" for k in range(1, 13)],
        *[f"l_value_lag_{k}" for k in range(1, 13)],
        "f_value",
        "f_value_ma3",
        "f_value_ma6",
        "l_value",
        "l_value_ma3",
        "l_value_ma6",
        "lead_lag_months",
        "pair_corr",
        "f_year",
        "f_month",
        "f_has_tx",
    ]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    print(f"\n사용할 feature 개수: {len(feature_cols)}")
    print(feature_cols)
    
    # 6. 모델 학습
    print("\n" + "=" * 60)
    models = train_ensemble_model(train_df, feature_cols)
    
    # 7. 추론 데이터 생성
    print("\n" + "=" * 60)
    all_items = df_panel["item_id"].unique()
    infer_df = create_inference_data(df_panel, comovement_candidates, all_items, max_lag=12)
    
    # 8. Feature 채우기
    missing_cols = [c for c in feature_cols if c not in infer_df.columns]
    for c in missing_cols:
        if c in train_df.columns:
            infer_df[c] = train_df[c].mean()
        else:
            infer_df[c] = 0.0
    
    # 9. 예측
    print("\n예측 중...")
    X_test = infer_df[feature_cols].values
    
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
    
    # 앙상블 (가중 평균)
    y_pred = (
        0.4 * predictions["rf"] +
        0.3 * predictions["gbm"] +
        0.3 * predictions["et"]
    )
    
    # 공행성이 없는 쌍은 간단한 예측 사용
    no_comovement_mask = infer_df["is_comovement"] == 0
    y_pred[no_comovement_mask] = infer_df.loc[no_comovement_mask, "f_value"].values
    
    y_pred = np.maximum(y_pred, 0)
    y_pred_int = np.round(y_pred).astype(int)
    
    # 10. Submission 생성
    submission = infer_df[["leading_item_id", "following_item_id"]].copy()
    submission["value"] = y_pred_int
    
    submission.to_csv("submission.csv", index=False)
    
    print("\n" + "=" * 60)
    print("완료!")
    print(f"생성된 submission 개수: {len(submission)}")
    print(f"공행성 쌍: {infer_df['is_comovement'].sum()}개")
    print(f"비공행성 쌍: {(infer_df['is_comovement'] == 0).sum()}개")
    print("\nsubmission.csv 파일이 생성되었습니다!")
    print("=" * 60)


if __name__ == "__main__":
    main()

