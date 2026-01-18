import requests
import xml.etree.ElementTree as ET
import pandas as pd

AUTH_KEY = 'api_key'
BASE = "https://ecos.bok.or.kr/api"

def ecos_stat_search(stat_code, cycle, start, end,
                     item_code1="", item_code2="", item_code3="", item_code4="",
                     lang="kr", count=100000):
    """
    ECOS StatisticSearch 호출해서 DataFrame으로 반환
    """
    url = (f"{BASE}/StatisticSearch/{AUTH_KEY}/xml/{lang}/1/{count}/"
           f"{stat_code}/{cycle}/{start}/{end}/"
           f"{item_code1}/{item_code2}/{item_code3}/{item_code4}")

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    rows = root.findall(".//row")

    data = []
    for row in rows:
        d = {c.tag: c.text for c in list(row)}
        data.append(d)

    return pd.DataFrame(data)

def ecos_item_list(stat_code, lang="kr", count=2000):
    """
    ECOS StatisticItemList 호출해서 항목코드 목록 반환
    """
    url = f"{BASE}/StatisticItemList/{AUTH_KEY}/xml/{lang}/1/{count}/{stat_code}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    rows = root.findall(".//row")
    data = [{c.tag: c.text for c in list(row)} for row in rows]
    return pd.DataFrame(data)

def pick_item_code(items_df: pd.DataFrame, prefer_keywords=None):
    """
    ITEM_NAME에 특정 키워드가 포함된 항목코드를 우선 선택.
    못 찾으면 첫 번째 항목코드 선택.
    항목이 아예 없는 통계표면 "" 반환.
    """
    if items_df.empty:
        return ""

    code_cols = [c for c in items_df.columns if "ITEM_CODE" in c.upper()]
    name_cols = [c for c in items_df.columns if "ITEM_NAME" in c.upper() or "ITEM_NM" in c.upper() or "NAME" in c.upper()]
    if not code_cols:
        return ""

    code_col = code_cols[0]
    name_col = name_cols[0] if name_cols else None

    if prefer_keywords and name_col:
        for kw in prefer_keywords:
            hit = items_df[items_df[name_col].astype(str).str.contains(kw, na=False)]
            if not hit.empty:
                return hit.iloc[0][code_col]

    return items_df.iloc[0][code_col]

def tidy(df, value_name):
    # ECOS 응답에서 TIME, DATA_VALUE만 뽑아 정리 (월별이면 TIME=YYYYMM)
    if df.empty:
        return df
    out = df[["TIME", "DATA_VALUE"]].copy()
    out["YYYYMM"] = out["TIME"].astype(str).str.slice(0, 6)  # 혹시라도 YYYYMMDD면 앞 6자리만
    out[value_name] = pd.to_numeric(out["DATA_VALUE"], errors="coerce")

    # 같은 YYYYMM이 여러 번 나오면(세부항목 섞였거나 중복) 월별 평균으로 정리
    out = out.groupby("YYYYMM", as_index=False)[value_name].mean()
    return out.sort_values("YYYYMM").reset_index(drop=True)


# -------------------------
# 여기서부터 실행부
# -------------------------

CPI_STAT_CODE  = "901Y009"  # CPI
RATE_STAT_CODE = "721Y001"  # 기준금리

START = "201501"
END   = "202512"

# 1) 항목코드 고정 (CPI는 "총지수/총괄" 같은 대표항목만)
cpi_items = ecos_item_list(CPI_STAT_CODE)
rate_items = ecos_item_list(RATE_STAT_CODE)

# CPI는 보통 총지수/총괄/전체 같은 항목을 선택
CPI_ITEM_CODE = pick_item_code(cpi_items, prefer_keywords=["총", "총지수", "총괄", "전체"])
# 기준금리는 항목이 1개거나, 비워도 되는 경우가 많지만 안전하게 하나 고정
RATE_ITEM_CODE = pick_item_code(rate_items, prefer_keywords=["기준", "한국은행", "금리"])

print("CPI_ITEM_CODE =", CPI_ITEM_CODE)
print("RATE_ITEM_CODE =", RATE_ITEM_CODE)

# 2) 월별 데이터 조회 (중요: item_code를 넣어서 '필요한 것만' 받기)
cpi_raw = ecos_stat_search(CPI_STAT_CODE,  "M", START, END, item_code1=CPI_ITEM_CODE)
rate_raw = ecos_stat_search(RATE_STAT_CODE, "M", START, END, item_code1=RATE_ITEM_CODE)

print("cpi_raw rows:", len(cpi_raw))
print("rate_raw rows:", len(rate_raw))

# 3) 정리 + merge
cpi = tidy(cpi_raw, "CPI")
rate = tidy(rate_raw, "BASE_RATE")

merged = pd.merge(cpi, rate, on="YYYYMM", how="outer").sort_values("YYYYMM").reset_index(drop=True)

print(merged.head(20))
print("merged rows:", len(merged))  # 정상이라면 132 근처(2015-01~2025-12)
