import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

print("="*70)
print("기계산업 공급망 기반 공행성 예측 모델")
print("="*70)

# 1. 데이터 전처리
print("\n[1단계] 데이터 전처리...")
train = pd.read_csv('./train.csv')

# HS4 코드 정보 추출
item_hs4 = train[['item_id', 'hs4']].drop_duplicates().set_index('item_id')['hs4'].to_dict()

# year, month, item_id 기준으로 value 합산
monthly = (
    train
    .groupby(["item_id", "year", "month"], as_index=False)["value"]
    .sum()
)

# year, month를 하나의 키(ym)로 묶기
monthly["ym"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
)

# item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
pivot = (
    monthly
    .pivot(index="item_id", columns="ym", values="value")
    .fillna(0.0)
)

print(f"Pivot table shape: {pivot.shape}")

# 2. 공급망 계층 구조 정의
def get_supply_chain_tier(hs4):
    """HS4 코드를 기반으로 공급망 계층 반환"""
    if 8400 <= hs4 < 8600:  # 기계류 및 전기기기
        return 3  # 최종제품
    elif 8600 <= hs4 < 9000:  # 운송기기
        return 3  # 최종제품
    elif 9000 <= hs4 < 9300:  # 정밀기기
        return 3  # 최종제품
    elif 7200 <= hs4 < 8400:  # 금속제품
        return 2  # 중간재
    elif 2800 <= hs4 < 3900:  # 화학제품
        return 1  # 원자재
    else:
        return 0  # 기타

def get_supply_chain_direction(leader_hs4, follower_hs4):
    """공급망 방향성 점수 계산 (원자재 → 중간재 → 최종제품)"""
    leader_tier = get_supply_chain_tier(leader_hs4)
    follower_tier = get_supply_chain_tier(follower_hs4)
    
    # 같은 계층 내에서도 공행성 가능 (같은 HS4 그룹)
    if leader_hs4 == follower_hs4:
        return 1.5  # 같은 품목 그룹
    
    # 공급망 방향성: 하위 계층 → 상위 계층
    if leader_tier < follower_tier:
        return 2.0  # 정상적인 공급망 방향 (원자재 → 최종제품)
    elif leader_tier == follower_tier:
        return 1.0  # 같은 계층
    else:
        return 0.5  # 역방향 (가능하지만 낮은 가중치)

def are_same_hs4_group(hs4_1, hs4_2):
    """같은 HS4 그룹인지 확인"""
    return hs4_1 == hs4_2

# 3. 공행성쌍 탐색 (공급망 고려)
def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def find_comovement_pairs_supply_chain(pivot, item_hs4, max_lag=6, min_nonzero=12, 
                                       corr_threshold=0.4, supply_chain_boost=True):
    """
    공급망 구조를 고려한 공행성쌍 탐색
    
    개선사항:
    1. 같은 HS4 그룹 내에서 우선 탐색
    2. 공급망 방향성 고려 (원자재 → 중간재 → 최종제품)
    3. 공급망 방향성에 따른 가중치 적용
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items), desc="Finding comovement pairs (supply chain)"):
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

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                corr = safe_corr(x[:-lag], y[lag:])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            # 공급망 방향성 점수 계산
            supply_chain_score = get_supply_chain_direction(leader_hs4, follower_hs4)
            same_hs4 = are_same_hs4_group(leader_hs4, follower_hs4)
            
            # 공급망 고려한 가중 상관계수
            if supply_chain_boost:
                weighted_corr = abs(best_corr) * supply_chain_score
            else:
                weighted_corr = abs(best_corr)

            # 임계값 이상이면 공행성쌍으로 채택
            # 공급망 방향이 좋으면 임계값을 낮춤
            effective_threshold = corr_threshold / supply_chain_score if supply_chain_boost else corr_threshold
            
            if best_lag is not None and abs(best_corr) >= effective_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                    "weighted_corr": weighted_corr,
                    "supply_chain_score": supply_chain_score,
                    "same_hs4": same_hs4,
                    "leader_hs4": leader_hs4,
                    "follower_hs4": follower_hs4,
                    "leader_tier": get_supply_chain_tier(leader_hs4),
                    "follower_tier": get_supply_chain_tier(follower_hs4),
                })

    pairs = pd.DataFrame(results)
    # 가중 상관계수로 정렬 (공급망 방향이 좋은 것 우선)
    if supply_chain_boost and len(pairs) > 0:
        pairs = pairs.sort_values('weighted_corr', ascending=False)
    
    return pairs


print("\n[2단계] 공급망 기반 공행성쌍 탐색...")
pairs = find_comovement_pairs_supply_chain(pivot, item_hs4, 
                                          corr_threshold=0.35,  # 공급망 고려하여 약간 낮춤
                                          supply_chain_boost=True)
print(f"탐색된 공행성쌍 수: {len(pairs)}")

if len(pairs) > 0:
    print("\n공급망 계층 분포:")
    print(pairs.groupby(['leader_tier', 'follower_tier']).size())
    print("\n같은 HS4 그룹 쌍:", pairs['same_hs4'].sum())
    print("\n상위 10개 공행성쌍:")
    print(pairs[['leading_item_id', 'following_item_id', 'max_corr', 
                'supply_chain_score', 'leader_tier', 'follower_tier']].head(10))

# 4. 회귀 모델 학습 데이터 생성 (HS4 정보 추가)
def build_training_data_supply_chain(pivot, pairs, item_hs4):
    """
    공행성쌍 + 시계열 + 공급망 정보를 이용해 (X, y) 학습 데이터를 만드는 함수
    input X:
      - b_t, b_t_1, a_t_lag, max_corr, best_lag
      - supply_chain_score, same_hs4 (공급망 정보 추가)
    target y:
      - b_t_plus_1
    """
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

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "supply_chain_score": supply_chain_score,
                "same_hs4": float(same_hs4),
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train


print("\n[3단계] 학습 데이터 생성 (공급망 정보 포함)...")
df_train_model = build_training_data_supply_chain(pivot, pairs, item_hs4)
print(f'생성된 학습 데이터의 shape: {df_train_model.shape}')
print(f"Training data head:\n{df_train_model.head()}")

# 회귀모델 학습
print("\n[4단계] 회귀 모델 학습...")
feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                'supply_chain_score', 'same_hs4']  # 공급망 정보 추가

train_X = df_train_model[feature_cols].values
train_y = df_train_model["target"].values

reg = LinearRegression()
reg.fit(train_X, train_y)

print("Model training completed!")
print(f"Feature importance (coefficients):")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {reg.coef_[i]:.6f}")

# 5. 회귀 모델 추론 및 제출(submission) 파일 생성
def predict_supply_chain(pivot, pairs, reg, item_hs4):
    months = pivot.columns.to_list()
    n_months = len(months)

    # 가장 마지막 두 달 index (2025-7, 2025-6)
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

        # t_last - lag 가 0 이상인 경우만 예측
        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        X_test = np.array([[b_t, b_t_1, a_t_lag, corr, float(lag), 
                           supply_chain_score, float(same_hs4)]])
        y_pred = reg.predict(X_test)[0]

        # (후처리 1) 음수 예측 → 0으로 변환
        # (후처리 2) 소수점 → 정수 변환 (무역량은 정수 단위)
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred


print("\n[5단계] 예측 수행...")
submission = predict_supply_chain(pivot, pairs, reg, item_hs4)
print(f"Submission head:\n{submission.head()}")

# 제출 파일 저장
output_file = './supply_chain_submit.csv'
submission.to_csv(output_file, index=False)
print(f"\nSubmission file saved to: {output_file}")
print(f"Total predictions: {len(submission)}")

# 통계 출력
print("\n" + "="*70)
print("모델 통계")
print("="*70)
print(f"총 공행성쌍: {len(pairs)}개")
print(f"예측된 쌍: {len(submission)}개")
print(f"평균 예측값: {submission['value'].mean():,.0f}")
print(f"최대 예측값: {submission['value'].max():,.0f}")
print(f"최소 예측값: {submission['value'].min():,.0f}")

