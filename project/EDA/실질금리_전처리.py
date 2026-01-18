import pandas as pd

df=pd.read_csv("/home/hyuksu/projects/ml/project/데이터/cpi_baserate_2014_2025_.csv",encoding='utf-8')
# =========================
# 1. YYYYMM → datetime
# =========================
df["YYYYMM"] = pd.to_datetime(df["YYYYMM"].astype(str), format="%Y%m")

# 날짜 정렬
df = df.sort_values("YYYYMM").reset_index(drop=True)

# =========================
# 2. 물가상승률 (전년동월, %)
# =========================
df["CPI_YOY"] = df["CPI"].pct_change(12) * 100

# =========================
# 3. 실질금리
#   실질금리 = 기준금리 - 물가상승률
# =========================
df["REAL_RATE"] = df["BASE_RATE"] - df["CPI_YOY"]

# =========================
# 4. 2015년부터만 사용
# =========================
df = df[df["YYYYMM"] >= "2015-01-01"].reset_index(drop=True)

# =========================
# 5. 결과 확인
# =========================
print(df[["YYYYMM", "CPI", "CPI_YOY", "BASE_RATE", "REAL_RATE"]].head(15))

