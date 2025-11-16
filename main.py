import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Mac 또는 Windows 환경에 맞게 조정 필요)
 #font_path = "C:/Windows/Fonts/malgun.ttf" # Windows 기준
 #font = font_manager.FontProperties(fname=font_path).get_name()
 #rc('font', family=font)
 #plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# --- 1. 데이터 로드 및 전처리 ---

# 1-1. 데이터 로드
df = pd.read_csv('train.csv')

# 초기 데이터 정보 확인 (사용자가 이미 작성한 부분)
print("--- 1. 초기 데이터 결측치 확인 ---")
print(df.isnull().sum())
print("\n--- 2. 초기 데이터 정보 ---")
df.info()

# 데이터중 weight, quantity, value 컬럼만 선택 (편차가 심해)
numeric_cols = ['weight', 'quantity', 'value']

# --- 로그 변환 데이터 준비 ---
# log(x + 1)을 적용하여 새 데이터프레임 생성
df_log = np.log1p(df[numeric_cols])
df_log.columns = [f'{col}_log' for col in numeric_cols]

# --- 시각화 ---
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
plt.suptitle('Comparison of Raw vs Log(x+1) Data Distribution', fontsize=16, y=1.02)

# 각 컬럼에 대해 Raw와 Log 분포 플롯
for i, col in enumerate(numeric_cols):
    
    # 1. Raw Data Distribution (좌측)
    sns.histplot(df[col], bins=50, kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Raw Distribution: {col}', fontsize=12)
    axes[i, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # 큰 숫자는 과학적 표기법으로

    # 2. Log-Transformed Data Distribution (우측)
    log_col = f'{col}_log'
    sns.histplot(df_log[log_col], bins=50, kde=True, ax=axes[i, 1], color='coral')
    axes[i, 1].set_title(f'Log(x+1) Distribution: {col}', fontsize=12)
    
plt.tight_layout()
plt.savefig('distribution_comparison.png')

print(">> 'distribution_comparison.png' 저장 완료 (로그 변환 전후 분포 비교)")

