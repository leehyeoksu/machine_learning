import requests
import pandas as pd
from time import sleep

SERVICE_KEY = "api_key"

BASE_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"

lawd_codes = {
    "종로": "11110",
    "중구": "11140",
    "용산": "11170",
    "서초": "11650",
    "강남": "11680",
    "송파": "11710"
}

months = pd.period_range("2015-01", "2025-12", freq="M").strftime("%Y%m")

def fetch_month(code: str, ymd: str, gu: str, num_rows: int = 1000):
    """해당 구/월의 모든 페이지를 긁어서 list[dict]로 반환"""
    out = []
    page = 1

    while True:
        params = {
            "serviceKey": SERVICE_KEY,
            "LAWD_CD": code,
            "DEAL_YMD": ymd,
            "pageNo": page,
            "numOfRows": num_rows,
            "_type": "json"
        }

        r = requests.get(BASE_URL, params=params, timeout=30)
        if r.status_code != 200:
            # 디버그용: 필요하면 출력
            # print("HTTP ERR", gu, ymd, r.status_code, r.text[:200])
            break

        data = r.json()
        body = data.get("response", {}).get("body", {})

        # 결과 0건이면 items가 없거나 ""일 수 있음
        items = body.get("items", {}).get("item", [])
        if not items:
            break

        # 단건이면 dict로 와서 list로 바꿔줌
        if isinstance(items, dict):
            items = [items]

        for it in items:
            it["구"] = gu
            it["계약월"] = ymd
            out.append(it)

        # 총 건수 기반 페이지 종료
        total = body.get("totalCount", 0)
        if page * num_rows >= total:
            break

        page += 1
        sleep(0.1)

    return out


rows = []
for gu, code in lawd_codes.items():
    for ymd in months:
        rows.extend(fetch_month(code, ymd, gu))
        sleep(0.2)  # 호출 제한 대비

df = pd.DataFrame(rows)

# --- 전처리: 거래금액 숫자화 ---
# 거래금액 컬럼명은 보통 "거래금액" (문자열 "84,000")
if "거래금액" in df.columns:
    df["거래금액"] = (
        df["거래금액"].astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["거래금액"] = pd.to_numeric(df["거래금액"], errors="coerce")

# 저장
df.to_csv("seoul_apartment_trade_2015_2025_6gu.csv", index=False, encoding="utf-8-sig")

print("✅ 완료")
print("행 수:", len(df))
print("컬럼:", df.columns.tolist()[:20])
print(df.head(3))
