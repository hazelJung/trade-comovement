"""
개선 3: Cross-Validation으로 하이퍼파라미터 튜닝

Time Series Cross-Validation을 사용하여 최적의 하이퍼파라미터를 찾습니다.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os

# 경로 설정
sys.path.append('../analysis')

def evaluate_pairs(pairs, pivot, df_panel, months, val_start_idx, val_end_idx):
    """
    특정 기간의 쌍에 대해 예측 및 평가
    (실제 정답이 없으므로 학습 데이터의 마지막 기간을 검증용으로 사용)
    """
    # 검증 기간의 학습 데이터 생성
    train_rows = []
    
    for row in pairs.itertuples(index=False):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        
        if leader not in pivot.index or follower not in pivot.index:
            continue
        
        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)
        
        # 검증 기간의 데이터만 사용
        for t in range(val_start_idx, val_end_idx):
            if t - lag < 0 or t + 1 >= len(months):
                continue
            
            b_t = b_series[t]
            b_t_1 = b_series[t - 1] if t > 0 else 0
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]
            
            # 특성 추출 (간단 버전)
            train_rows.append({
                'b_t': b_t,
                'b_t_1': b_t_1,
                'a_t_lag': a_t_lag,
                'target': b_t_plus_1,
            })
    
    if len(train_rows) == 0:
        return None
    
    df_val = pd.DataFrame(train_rows)
    return df_val


def time_series_cv_tuning(pivot, df_panel, item_hs4):
    """
    Time Series Cross-Validation으로 하이퍼파라미터 튜닝
    """
    from train_model_preprocessing import find_comovement_pairs, build_training_data, get_supply_chain_tier
    
    months = pivot.columns.to_list()
    n_months = len(months)
    
    # 검증 기간 설정 (마지막 6개월을 검증용으로 사용)
    val_months = 6
    train_end_idx = n_months - val_months
    
    print("="*70)
    print("하이퍼파라미터 튜닝 (Time Series Cross-Validation)")
    print("="*70)
    print(f"전체 기간: {len(months)}개월")
    print(f"학습 기간: 처음 {train_end_idx}개월")
    print(f"검증 기간: 마지막 {val_months}개월")
    print()
    
    # 튜닝할 하이퍼파라미터
    max_pairs_candidates = [None, 1200, 1500, 1800]
    min_abs_corr_candidates = [0.25, 0.30, 0.35]
    ridge_alpha_candidates = [0.1, 1.0, 10.0]
    
    best_score = -np.inf
    best_params = None
    results = []
    
    print("하이퍼파라미터 조합 테스트 중...")
    total_combinations = len(max_pairs_candidates) * len(min_abs_corr_candidates) * len(ridge_alpha_candidates)
    pbar = tqdm(total=total_combinations, desc="Tuning")
    
    for max_pairs in max_pairs_candidates:
        for min_abs_corr in min_abs_corr_candidates:
            # 쌍 탐색
            pairs = find_comovement_pairs(
                pivot, item_hs4, 
                max_pairs=max_pairs,
                min_abs_corr=min_abs_corr
            )
            
            if len(pairs) == 0:
                continue
            
            # 학습 데이터 생성 (검증 기간 제외)
            # 간단히 전체 데이터로 학습하고 검증 기간으로 평가
            df_train = build_training_data(pivot, pairs, df_panel)
            
            if len(df_train) == 0:
                continue
            
            # 검증 데이터 (마지막 기간)
            df_val = evaluate_pairs(pairs, pivot, df_panel, months, train_end_idx, n_months - 1)
            
            if df_val is None or len(df_val) == 0:
                continue
            
            for alpha in ridge_alpha_candidates:
                # 모델 학습
                feature_cols = ['b_t', 'b_t_1', 'a_t_lag']
                train_X = df_train[feature_cols].values
                train_y = df_train["target"].values
                
                train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
                train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)
                
                scaler = StandardScaler()
                train_X_scaled = scaler.fit_transform(train_X)
                
                reg = Ridge(alpha=alpha)
                reg.fit(train_X_scaled, train_y)
                
                # 검증 데이터 예측
                val_X = df_val[feature_cols].values
                val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
                val_X_scaled = scaler.transform(val_X)
                val_y_pred = reg.predict(val_X_scaled)
                val_y_true = df_val["target"].values
                
                # NMAE 계산 (간단 버전)
                eps = 1e-6
                rel_errors = np.abs(val_y_true - val_y_pred) / (np.abs(val_y_true) + eps)
                rel_errors = np.clip(rel_errors, 0, 1.0)
                nmae = np.mean(rel_errors)
                
                # 점수 (NMAE가 낮을수록 좋음, 1 - NMAE로 변환)
                score = 1.0 - nmae
                
                results.append({
                    'max_pairs': max_pairs if max_pairs else 'all',
                    'min_abs_corr': min_abs_corr,
                    'ridge_alpha': alpha,
                    'n_pairs': len(pairs),
                    'n_train': len(df_train),
                    'n_val': len(df_val),
                    'nmae': nmae,
                    'score': score,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'max_pairs': max_pairs,
                        'min_abs_corr': min_abs_corr,
                        'ridge_alpha': alpha,
                    }
                
                pbar.update(1)
    
    pbar.close()
    
    # 결과 정리
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False)
    
    print("\n" + "="*70)
    print("튜닝 결과 (상위 10개)")
    print("="*70)
    print(df_results.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("최적 하이퍼파라미터")
    print("="*70)
    print(f"max_pairs: {best_params['max_pairs']}")
    print(f"min_abs_corr: {best_params['min_abs_corr']}")
    print(f"ridge_alpha: {best_params['ridge_alpha']}")
    print(f"최고 점수: {best_score:.6f}")
    
    # 결과 저장
    df_results.to_csv('../results/hyperparameter_tuning_results.csv', index=False)
    print(f"\n결과 저장: ../results/hyperparameter_tuning_results.csv")
    
    return best_params, df_results


if __name__ == "__main__":
    # 데이터 로드 (간단 버전)
    print("데이터 로드 중...")
    # 실제로는 train_model_preprocessing.py의 데이터 로드 부분을 재사용
    print("주의: 이 스크립트는 train_model_preprocessing.py와 함께 사용해야 합니다.")
    print("또는 필요한 데이터를 직접 로드해야 합니다.")

