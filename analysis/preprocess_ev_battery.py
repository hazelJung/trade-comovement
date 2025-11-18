"""
EV 배터리 공급망 데이터 전처리 파이프라인
- 무역 데이터를 EV 배터리 공급망 도메인 관점에서 전처리
- 결측치, 이상치, 0값 처리 및 도메인 특화 피처 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_data(file_path=None):
    """
    1️⃣ 기본 로드 & 보조 컬럼 생성
    - train.csv를 로드하고 year, month로 ym 컬럼 생성
    """
    if file_path is None:
        file_path = DATA_DIR / 'train.csv'
    else:
        # 상대 경로인 경우 프로젝트 루트 기준으로 변환
        if not os.path.isabs(file_path):
            # ../data/train.csv 같은 경우 처리
            file_path = file_path.replace('../', '')
            file_path = PROJECT_ROOT / file_path
    df = pd.read_csv(str(file_path))
    
    # year, month를 이용해 datetime 컬럼 생성
    df['ym'] = pd.to_datetime(
        df['year'].astype(str) + "-" + df['month'].astype(str).str.zfill(2) + "-01"
    )
    
    print(f"데이터 로드 완료: {len(df):,}행")
    return df


def build_item_hs4_mapping(df):
    """
    2️⃣ item_id-hs4 매핑 확보
    - 같은 item_id는 항상 동일한 hs4를 가진다는 가정
    - (item_id, hs4) 유일 조합으로 매핑 테이블 생성
    """
    item_hs4_map = (
        df[['item_id', 'hs4']]
        .drop_duplicates()
        .set_index('item_id')['hs4']
        .to_dict()
    )
    
    print(f"item_id-hs4 매핑 완료: {len(item_hs4_map):,}개 item_id")
    return item_hs4_map


def build_monthly_aggregates(df, item_hs4_map):
    """
    3️⃣ (item_id, year, month) 단위 월별 집계
    - value, weight, quantity를 합산하여 월별 집계 생성
    - 이후 모든 전처리는 이 monthly 기준으로 진행
    """
    # 월별 집계
    monthly = (
        df.groupby(['item_id', 'year', 'month'], as_index=False)
        .agg({
            'value': 'sum',
            'weight': 'sum',
            'quantity': 'sum',
            'ym': 'first'  # ym은 동일하므로 first 사용
        })
        .rename(columns={
            'value': 'value_sum',
            'weight': 'weight_sum',
            'quantity': 'quantity_sum'
        })
    )
    
    # item_id-hs4 매핑 적용
    monthly['hs4'] = monthly['item_id'].map(item_hs4_map)
    
    print(f"월별 집계 완료: {len(monthly):,}행")
    return monthly


def add_cluster_labels(df):
    """
    4️⃣ EV 배터리 공급망 도메인 클러스터 라벨링
    - HS2 코드를 기반으로 배터리 소재, EV 기기, 기타 산업으로 분류
    - 배터리 공급망 관점에서 품목군을 의미있게 그룹화
    """
    # hs4를 문자열로 변환하고 HS2 추출
    hs4_str = df['hs4'].astype(str).str.zfill(4)
    df['hs2'] = hs4_str.str[:2]
    
    # 클러스터 라벨링 함수
    def assign_cluster(hs2):
        if hs2 in ["27", "28", "29", "31", "38", "39", "72", "79", "81"]:
            return "battery_materials"  # 배터리 소재·화학·금속
        elif hs2 in ["84", "85", "87", "90"]:
            return "ev_machinery_electrical"  # EV·설비·전기기기
        else:
            return "other_industries"  # 기타 산업
    
    df['cluster'] = df['hs2'].apply(assign_cluster)
    
    print("클러스터 라벨링 완료")
    print(f"  - battery_materials: {(df['cluster'] == 'battery_materials').sum():,}행")
    print(f"  - ev_machinery_electrical: {(df['cluster'] == 'ev_machinery_electrical').sum():,}행")
    print(f"  - other_industries: {(df['cluster'] == 'other_industries').sum():,}행")
    
    return df


def handle_quantity(df):
    """
    5️⃣ quantity 처리 (구조적 결측 반영)
    - 어떤 hs4는 quantity_sum이 항상 0 (수량 정보가 의미 없는 HS4)
    - 수량 정보가 의미 있는 HS4와 현재 행의 수량 존재 여부를 플래그로 표시
    """
    # hs4별로 quantity_sum > 0인 row가 하나라도 있는지 확인
    hs4_has_quantity = (
        df.groupby('hs4')['quantity_sum']
        .apply(lambda x: (x > 0).any())
        .to_dict()
    )
    
    # 해당 hs4에서 quantity 정보가 의미 있는지 플래그
    df['has_quantity_info_hs4'] = df['hs4'].map(hs4_has_quantity).astype(int)
    
    # 현재 row의 quantity_sum > 0 여부
    df['has_quantity_this_row'] = (df['quantity_sum'] > 0).astype(int)
    
    print("Quantity 처리 완료")
    print(f"  - 수량 정보 의미 있는 HS4: {df['has_quantity_info_hs4'].sum():,}행")
    print(f"  - 수량 존재하는 행: {df['has_quantity_this_row'].sum():,}행")
    
    return df


def add_zero_missing_flags(df):
    """
    6️⃣ 0값 기반 "사실상 결측" 플래그
    - 무역 데이터에서 0값은 실제 거래 없음 또는 정보 미기입을 의미할 수 있음
    - weight/value 중 하나만 0인 경우는 정보 미기입 가능성이 높음
    """
    # 기본 0값 플래그
    df['weight_zero_flag'] = (df['weight_sum'] == 0).astype(int)
    df['value_zero_flag'] = (df['value_sum'] == 0).astype(int)
    
    # 사실상 결측 플래그
    # weight가 0인데 value는 있음 → 무게 정보 미기입 가능성
    df['weight_missing_flag'] = (
        (df['weight_sum'] == 0) & (df['value_sum'] > 0)
    ).astype(int)
    
    # value가 0인데 weight는 있음 → 금액 정보 미기입 가능성
    df['value_missing_flag'] = (
        (df['value_sum'] == 0) & (df['weight_sum'] > 0)
    ).astype(int)
    
    print("0값/결측 플래그 생성 완료")
    print(f"  - weight_zero_flag: {df['weight_zero_flag'].sum():,}행")
    print(f"  - value_zero_flag: {df['value_zero_flag'].sum():,}행")
    print(f"  - weight_missing_flag: {df['weight_missing_flag'].sum():,}행")
    print(f"  - value_missing_flag: {df['value_missing_flag'].sum():,}행")
    
    return df


def fill_missing_values(df):
    """
    7️⃣ weight_sum / value_sum 결측성 보정
    - 원본 값은 유지하고, 모델 학습용 보정 버전을 별도 컬럼으로 생성
    - 계층적 보정: item_id 중앙값 → hs4 중앙값 → 전체 중앙값
    """
    # 중앙값 사전 계산 (효율성을 위해 dict로 저장)
    item_weight_median = df.groupby('item_id')['weight_sum'].median().to_dict()
    hs4_weight_median = df.groupby('hs4')['weight_sum'].median().to_dict()
    global_weight_median = df['weight_sum'].median()
    
    item_value_median = df.groupby('item_id')['value_sum'].median().to_dict()
    hs4_value_median = df.groupby('hs4')['value_sum'].median().to_dict()
    global_value_median = df['value_sum'].median()
    
    # weight_filled: weight_missing_flag == 1인 경우 계층적 보정
    df['weight_filled'] = df['weight_sum'].copy()
    
    missing_weight_mask = df['weight_missing_flag'] == 1
    if missing_weight_mask.sum() > 0:
        # item_id 중앙값으로 먼저 시도
        df.loc[missing_weight_mask, 'weight_filled'] = (
            df.loc[missing_weight_mask, 'item_id'].map(item_weight_median)
        )
        
        # item_id 중앙값이 없거나 0인 경우 hs4 중앙값으로
        still_missing = (df['weight_filled'].isna()) | (df['weight_filled'] == 0)
        df.loc[still_missing & missing_weight_mask, 'weight_filled'] = (
            df.loc[still_missing & missing_weight_mask, 'hs4'].map(hs4_weight_median)
        )
        
        # 그것도 안 되면 전체 중앙값
        still_missing = (df['weight_filled'].isna()) | (df['weight_filled'] == 0)
        df.loc[still_missing & missing_weight_mask, 'weight_filled'] = global_weight_median
    
    # value_filled: value_missing_flag == 1인 경우 계층적 보정
    df['value_filled'] = df['value_sum'].copy()
    
    missing_value_mask = df['value_missing_flag'] == 1
    if missing_value_mask.sum() > 0:
        # item_id 중앙값으로 먼저 시도
        df.loc[missing_value_mask, 'value_filled'] = (
            df.loc[missing_value_mask, 'item_id'].map(item_value_median)
        )
        
        # item_id 중앙값이 없거나 0인 경우 hs4 중앙값으로
        still_missing = (df['value_filled'].isna()) | (df['value_filled'] == 0)
        df.loc[still_missing & missing_value_mask, 'value_filled'] = (
            df.loc[still_missing & missing_value_mask, 'hs4'].map(hs4_value_median)
        )
        
        # 그것도 안 되면 전체 중앙값
        still_missing = (df['value_filled'].isna()) | (df['value_filled'] == 0)
        df.loc[still_missing & missing_value_mask, 'value_filled'] = global_value_median
    
    print("결측값 보정 완료")
    print(f"  - weight_filled 보정: {missing_weight_mask.sum():,}행")
    print(f"  - value_filled 보정: {missing_value_mask.sum():,}행")
    
    return df


def add_outlier_flags_and_clipping(df):
    """
    8️⃣ 극단값(이상치) 처리: 클리핑 + 플래그
    - 99 분위수를 기준으로 극단값 탐지 및 클리핑
    - 무역 데이터에서 극단값은 오기입 또는 특수 거래일 가능성이 높음
    """
    # 99 분위수 계산
    q99_value = df['value_filled'].quantile(0.99)
    q99_weight = df['weight_filled'].quantile(0.99)
    q99_quantity = df['quantity_sum'].quantile(0.99)
    
    # 극단값 플래그
    df['value_extreme_flag'] = (df['value_filled'] > q99_value).astype(int)
    df['weight_extreme_flag'] = (df['weight_filled'] > q99_weight).astype(int)
    df['quantity_extreme_flag'] = (df['quantity_sum'] > q99_quantity).astype(int)
    
    # 클리핑된 컬럼 생성
    df['value_clip'] = np.minimum(df['value_filled'], q99_value)
    df['weight_clip'] = np.minimum(df['weight_filled'], q99_weight)
    df['quantity_clip'] = np.minimum(df['quantity_sum'], q99_quantity)
    
    print("이상치 처리 완료")
    print(f"  - value_extreme_flag: {df['value_extreme_flag'].sum():,}행")
    print(f"  - weight_extreme_flag: {df['weight_extreme_flag'].sum():,}행")
    print(f"  - quantity_extreme_flag: {df['quantity_extreme_flag'].sum():,}행")
    print(f"  - 99 분위수: value={q99_value:,.0f}, weight={q99_weight:,.0f}, quantity={q99_quantity:,.0f}")
    
    return df


def add_log_features(df):
    """
    9️⃣ 로그 변환 피처
    - 0값 처리와 스케일 안정을 위해 log1p 사용
    - 무역 데이터의 스케일 차이가 매우 크므로 로그 변환이 필수적
    """
    df['log_value'] = np.log1p(df['value_clip'])
    df['log_weight'] = np.log1p(df['weight_clip'])
    df['log_quantity'] = np.log1p(df['quantity_clip'])
    
    print("로그 변환 완료")
    return df


def main():
    """
    전체 전처리 파이프라인 실행
    """
    print("="*70)
    print("EV 배터리 공급망 데이터 전처리 시작")
    print("="*70)
    
    # 1. 데이터 로드
    df_train = load_data()
    
    # 2. item_id-hs4 매핑 생성
    item_hs4_map = build_item_hs4_mapping(df_train)
    
    # 3. 월별 집계
    monthly_features = build_monthly_aggregates(df_train, item_hs4_map)
    
    # 4. 클러스터 라벨링
    monthly_features = add_cluster_labels(monthly_features)
    
    # 5. Quantity 처리
    monthly_features = handle_quantity(monthly_features)
    
    # 6. 0값/결측 플래그
    monthly_features = add_zero_missing_flags(monthly_features)
    
    # 7. 결측값 보정
    monthly_features = fill_missing_values(monthly_features)
    
    # 8. 이상치 처리 및 클리핑
    monthly_features = add_outlier_flags_and_clipping(monthly_features)
    
    # 9. 로그 변환
    monthly_features = add_log_features(monthly_features)
    
    # 최종 컬럼 순서 정리
    final_columns = [
        # 식별자/키
        'item_id', 'hs4', 'hs2', 'year', 'month', 'ym', 'cluster',
        # 집계 원본
        'value_sum', 'weight_sum', 'quantity_sum',
        # 결측/0값/플래그
        'has_quantity_info_hs4', 'has_quantity_this_row',
        'weight_zero_flag', 'value_zero_flag',
        'weight_missing_flag', 'value_missing_flag',
        'value_extreme_flag', 'weight_extreme_flag', 'quantity_extreme_flag',
        # 보정·클리핑
        'weight_filled', 'value_filled',
        'value_clip', 'weight_clip', 'quantity_clip',
        # 로그 피처
        'log_value', 'log_weight', 'log_quantity'
    ]
    
    monthly_features = monthly_features[final_columns]
    
    print("\n" + "="*70)
    print("전처리 완료!")
    print("="*70)
    print(f"최종 데이터 shape: {monthly_features.shape}")
    print(f"컬럼 수: {len(monthly_features.columns)}")
    print("\n최종 컬럼 목록:")
    for col in monthly_features.columns:
        print(f"  - {col}")
    
    # CSV 저장
    output_dir = PROJECT_ROOT / 'analysis' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'monthly_features.csv'
    monthly_features.to_csv(str(output_path), index=False)
    print(f"\n결과 저장 완료: {output_path}")
    
    return monthly_features


if __name__ == "__main__":
    monthly_features = main()

