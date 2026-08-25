from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "chouka.csv"
JMA_CSV = APP_DIR / "data" / "data.csv"

TARGET = "竿頭(宝来丸)"
REQUIRED = ["日付", "潮回り", TARGET, "潮の速さ", "月", "天気", "波高", "風速", "気温"]

TIDE_ORDER = ["大潮", "中潮", "小潮", "長潮", "若潮"]
WEATHER_OPTIONS = ["晴", "曇", "雨", "晴曇", "曇晴", "雨曇", "雨晴", "晴雨", "不明"]
SPEED_OPTIONS = ["緩い", "普通", "速い", "カッ飛び", "不明"]

st.set_page_config(page_title="三国沖 イカメタル釣果予測", page_icon="🦑", layout="wide")


def clean(v):
    return "" if pd.isna(v) else str(v).strip()


def moon_band(m):
    if pd.isna(m):
        return "不明"
    m = float(m) % 29.53
    if m < 7.5:
        return "新月〜上弦"
    if m < 15:
        return "上弦〜満月"
    if m < 22.5:
        return "満月〜下弦"
    return "下弦〜新月"


def normalize_header(x):
    return (str(x).replace("\ufeff", "").strip()
            .replace("（", "(").replace("）", ")")
            .replace(" ", ""))


def find_column(df, wanted):
    wanted_n = normalize_header(wanted)
    for c in df.columns:
        if normalize_header(c) == wanted_n:
            return c
    if wanted == TARGET:
        for c in df.columns:
            n = normalize_header(c)
            if "竿頭" in n and "宝来丸" in n:
                return c
    return None


def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce"
    )


# --- data/data.csv から18〜23時の平均風速・平均気温を引くキャッシュ用辞書 ---
@st.cache_data
def load_jma_cache():
    if not JMA_CSV.exists():
        return {}, {}
        
    try:
        df_jma = None
        for enc in ['utf-8-sig', 'cp932', 'utf-8']:
            try:
                df_jma = pd.read_csv(JMA_CSV, encoding=enc, header=None)
                break
            except Exception:
                continue
                
        if df_jma is None or df_jma.shape[1] < 3:
            return {}, {}
            
        start_row = 0
        first_val = str(df_jma.iloc[0, 2])
        if "風速" in first_val or "wind" in first_val.lower() or "気温" in first_val:
            start_row = 1
            
        df_jma = df_jma.iloc[start_row:].copy()
        df_jma['dt'] = pd.to_datetime(df_jma.iloc[:, 0].astype(str).str.strip() + ' ' + df_jma.iloc[:, 1].astype(str).str.strip(), errors='coerce')
        
        df_jma['wind'] = pd.to_numeric(df_jma.iloc[:, 2], errors='coerce')
        
        temp_col_idx = None
        for c_idx in range(3, df_jma.shape[1]):
            col_vals = pd.to_numeric(df_jma.iloc[:, c_idx], errors='coerce')
            if col_vals.dropna().between(-15, 45).mean() > 0.8:
                temp_col_idx = c_idx
                break
        
        if temp_col_idx is not None:
            df_jma['temp'] = pd.to_numeric(df_jma.iloc[:, temp_col_idx], errors='coerce')
        
        df_jma['hour'] = df_jma['dt'].dt.hour
        df_jma['date_str'] = df_jma['dt'].dt.strftime('%Y/%m/%d')
        
        mask = df_jma['hour'].isin([18, 19, 20, 21, 22, 23])
        sub = df_jma[mask]
        
        wind_dict = sub.groupby('date_str')['wind'].mean().to_dict() if 'wind' in sub.columns else {}
        temp_dict = sub.groupby('date_str')['temp'].mean().to_dict() if 'temp' in sub.columns else {}
        return wind_dict, temp_dict
    except Exception as e:
        print(f"気象庁CSV読込エラー: {e}")
        return {}, {}


def fetch_jma_val(date_str, cache_dict):
    if not cache_dict:
        return np.nan
    try:
        dt = pd.to_datetime(date_str)
        key1 = dt.strftime('%Y/%m/%d')
        if key1 in cache_dict:
            return cache_dict[key1]
        key2 = f"{dt.year}/{dt.month}/{dt.day}"
        if key2 in cache_dict:
            return cache_dict[key2]
        return np.nan
    except Exception:
        return np.nan


def normalize_columns(df):
    aliases = {
        TARGET: [TARGET, "竿頭（宝来丸）", "竿頭(宝来丸) ", "竿頭（宝来丸） "],
    }
    rename = {}
    cols = {str(c).strip(): c for c in df.columns}
    for canonical, names in aliases.items():
        for n in names:
            if n in cols:
                rename[cols[n]] = canonical
                break
    return df.rename(columns=rename)


def prepare(df):
    df = normalize_columns(df.copy())
    df.columns = [normalize_header(c) for c in df.columns]
    for c in REQUIRED:
        if c not in df.columns:
            found = find_column(df, c)
            df[c] = df[found] if found is not None else ""
            
    df["波高"] = to_number(df["波高"])
    df["風速"] = to_number(df["風速"])
    df["気温"] = to_number(df["気温"])
    
    wind_cache, temp_cache = load_jma_cache()
    for idx, row in df.iterrows():
        date_val = row["日付"]
        if pd.notna(date_val):
            if pd.isna(row["風速"]):
                w_val = fetch_jma_val(date_val, wind_cache)
                if pd.notna(w_val):
                    df.at[idx, "風速"] = w_val
            if pd.isna(row["気温"]):
                t_val = fetch_jma_val(date_val, temp_cache)
                if pd.notna(t_val):
                    df.at[idx, "気温"] = t_val
                    
    df["風速"] = df["風速"].fillna(2.5)
    df["気温"] = df["気温"].fillna(22.0)
    df["月"] = to_number(df["月"])
    df[TARGET] = to_number(df[TARGET])
    for c in ["潮の速さ", "天気", "潮回り"]:
        df[c] = df[c].map(clean).replace("", "不明")
    df["月齢帯"] = df["月"].map(moon_band)
    df["__date"] = pd.to_datetime(df["日付"], errors="coerce")
    return df


def read_csv_bytes(file_obj):
    raw = file_obj.getvalue()
    last = None
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            from io import BytesIO
            return prepare(pd.read_csv(BytesIO(raw), encoding=enc))
        except UnicodeDecodeError as e:
            last = e
    raise last


@st.cache_data
def load_default_cached():
    last = None
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            if DEFAULT_CSV.exists():
                return prepare(pd.read_csv(DEFAULT_CSV, encoding=enc))
        except UnicodeDecodeError as e:
            last = e
    return pd.DataFrame()


def parse_count_series(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)")[0], errors="coerce"
    )


def analysis_frame(df):
    if TARGET not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["釣果"] = parse_count_series(x[TARGET])
    x["船"] = "宝来丸"
    return x.dropna(subset=["釣果"])


def probability(df, mask):
    x = df.loc[mask, "釣果"].dropna()
    if len(x) == 0:
        return None
    return {"n": len(x), "p20": (x >= 20).mean() * 100,
            "p30": (x >= 30).mean() * 100,
            "p40": (x >= 40).mean() * 100,
            "mean": x.mean(), "median": x.median()}


def similarity(row, inp):
    score = 0
    if clean(row["潮回り"]) == inp["tide"]:
        score += 15
    if pd.notna(row["月"]):
        d = abs(float(row["月"]) - inp["moon"])
        if d <= 1.5:
            score += 15
        elif d <= 3:
            score += 8
    if clean(row["潮の速さ"]) == inp["speed"]:
        score += 15
    if clean(row["天気"]) == inp["weather"]:
        score += 10
    
    if pd.notna(row["波高"]):
        h_diff = abs(float(row["波高"]) - inp["wave"])
        if h_diff <= 0.3:
            score += 15
        elif h_diff <= 0.8:
            score += 8
        
    if pd.notna(row["風速"]):
        w_diff = abs(float(row["風速"]) - inp["wind"])
        if w_diff <= 1.5:
            score += 15
        elif w_diff <= 3.0:
            score += 5

    if pd.notna(row["気温"]):
        t_diff = abs(float(row["気温"]) - inp["temp"])
        if t_diff <= 2.0:
            score += 15
        elif t_diff <= 4.5:
            score += 5
            
    return score


def predict(df, inp):
    v = analysis_frame(df)
    if v.empty:
        return None
    scored = [(idx, similarity(r, inp)) for idx, r in v.iterrows()]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:10]
    rows = v.loc[[x[0] for x in scored]].copy()
    w = np.array([max(1, x[1]) for x in scored], dtype=float)
    y = rows["釣果"].to_numpy(float)
    p20 = np.average((y >= 20).astype(float), weights=w) * 100
    p30 = np.average((y >= 30).astype(float), weights=w) * 100
    p40 = np.average((y >= 40).astype(float), weights=w) * 100
    mean = np.average(y, weights=w)
    q25, q75 = np.percentile(y, [25, 75]) if len(y) > 1 else (y[0], y[0])
    
    exact = (
        (v["潮回り"] == inp["tide"]) &
        (v["潮の速さ"] == inp["speed"]) &
        (v["天気"] == inp["weather"]) &
        (v["月齢帯"] == moon_band(inp["moon"])) &
        (v["波高"].notna()) &
        ((v["波高"] - inp["wave"]).abs() <= 0.5)
    )
    ex = probability(v, exact)

    avg_score = np.mean([x[1] for x in scored]) if scored else 0.0
    max_score = 100.0
    similarity_factor = min(1.0, max(0.0, avg_score / max_score))
    n_factor = 1.0 - np.exp(-len(rows) / 8.0)
    dispersion = float(np.std(y)) if len(y) > 1 else 0.0
    mean_abs = max(float(np.mean(np.abs(y))), 1.0)
    consistency_factor = 1.0 / (1.0 + dispersion / mean_abs)
    confidence = 100.0 * (0.45 * n_factor + 0.35 * similarity_factor + 0.20 * consistency_factor)
    
    if len(rows) < 5:
        confidence = min(confidence, 65.0)
    elif len(rows) < 10:
        confidence = min(confidence, 80.0)
    elif len(rows) < 20:
        confidence = min(confidence, 90.0)
    else:
        confidence = min(confidence, 97.0)
        
    return dict(p20=p20, p30=p30, p40=p40, mean=mean, low=q25, high=q75,
                confidence=confidence, rows=rows, scored=scored, exact=ex)


def make_ranking(df, min_n=3):
    base = analysis_frame(df)
    if base.empty:
        return []
    fields = ["潮回り", "月齢帯", "潮の速さ", "天気"]
    work = base.copy()
    for c in fields:
        work[c] = work[c].map(clean)
    work = work[~work[fields].isin(["", "不明"]).any(axis=1)].copy()
    out = []
    for keys, g in work.groupby(fields, dropna=False):
        if len(g) < min_n:
            continue
        y = g["釣果"].astype(float)
        n = len(y)
        raw = (y >= 20).mean() * 100
        corrected = (raw * n + 50.0 * 3) / (n + 3)
        out.append((" × ".join(map(str, keys)), n, raw, corrected, y.mean()))
    out.sort(key=lambda x: (x[3], x[1], x[4]), reverse=True)
    return out[:20]


# --- CSS ---
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
.hero {background:#172033;color:white;padding:22px 24px;border-radius:14px;margin-bottom:18px;}
.hero h1 {margin:0;font-size:30px;}
.hero p {margin:6px 0 0;color:#cbd3df;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🦑 三国沖 イカメタル釣果予測</h1><p>宝来丸の過去釣果と気象庁の風速・気温データから高精度予測</p></div>', unsafe_allow_html=True)

# セッション内データ初期化
if "df" not in st.session_state:
    df_default = load_default_cached()
    st.session_state.df = df_default
    st.session_state.source = "data/chouka.csv" if not df_default.empty else ""

with st.sidebar:
    st.header("📁 データ設定")
    uploaded = st.file_uploader("釣果CSVを読み込む", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.df = read_csv_bytes(uploaded)
            st.session_state.source = uploaded.name
            st.success(f"{uploaded.name} を読み込みました")
        except Exception as e:
            st.error(f"CSV読込エラー：{e}")
            
    st.divider()
    wind_c, temp_c = load_jma_cache()
    jma_status = "有効 (data/data.csv)" if (wind_c or temp_c) else "未検出"
    st.caption(f"気象庁データ連携: {jma_status}")


df = st.session_state.df
if df.empty:
    st.warning("CSVが読み込まれていません。左側からCSVをアップロードするか、リポジトリに `data/chouka.csv` を配置してください。")
    st.stop()

# 入力欄
st.subheader("🔮 予測条件")
c1, c2, c3 = st.columns(3)
with c1:
    prediction_date = st.date_input("予測日", value=datetime.now().date())
    tide_present = [x for x in TIDE_ORDER if x in set(df["潮回り"])]
    tide = st.selectbox("潮回り", tide_present or TIDE_ORDER, index=(tide_present or TIDE_ORDER).index("中潮") if "中潮" in (tide_present or TIDE_ORDER) else 0)
with c2:
    moon = st.number_input("月齢", min_value=0.0, max_value=29.53, value=12.4, step=0.1)
    st.info(f"月齢帯：**{moon_band(moon)}**")
    weather = st.selectbox("天気", WEATHER_OPTIONS, index=0)
with c3:
    speed = st.selectbox("潮の速さ", SPEED_OPTIONS, index=2)
    wave = st.number_input("波高 (m)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    wind = st.number_input("18-23時風速 (m/s)", min_value=0.0, max_value=25.0, value=3.0, step=0.5)
    temp = st.number_input("18-23時気温 (℃)", min_value=-10.0, max_value=40.0, value=22.0, step=0.5)

inp = {"tide": tide, "moon": moon, "weather": weather, "speed": speed, "wave": wave, "wind": wind, "temp": temp}

if st.button("🔍 釣果を予測する", type="primary", use_container_width=True):
    r = predict(df, inp)
    if r is None:
        st.warning("分析できる釣果データがありません。")
    else:
        st.session_state.result = r
        st.session_state.result_date = prediction_date

if "result" in st.session_state:
    r = st.session_state.result
    st.subheader(f"📅 {st.session_state.result_date:%Y/%m/%d} の予測結果（宝来丸）")
    a, b, c, d = st.columns(4)
    a.metric("20杯以上", f"{r['p20']:.0f}%")
    b.metric("30杯以上", f"{r['p30']:.0f}%")
    c.metric("40杯以上", f"{r['p40']:.0f}%")
    d.metric("予想レンジ", f"{r['low']:.0f}〜{r['high']:.0f}杯")
    st.progress(min(1.0, r["confidence"] / 100))
    st.caption(f"信頼度 {r['confidence']:.0f}/100 ※類似件数や条件一致度から算出した指標")

    st.write(f"**分析対象：** 宝来丸 ｜ **入力：** {tide} × 月齢{moon:.1f} × {weather} × 潮速:{speed} × 波高{wave}m × 風速{wind}m/s × 気温{temp}℃")
    if r["exact"]:
        st.info(f"★ 近接条件一致: {r['exact']['n']}件 / 20杯以上 {r['exact']['p20']:.1f}%")

    st.subheader("🔎 似ている過去データ（風速・気温考慮）")
    rows = r["rows"].copy()
    rows["日付"] = rows["__date"].dt.strftime("%Y/%m/%d")
    score_map = {idx: score for idx, score in r["scored"]}
    rows["一致点"] = [score_map.get(i, 0) for i in rows.index]
    cols = ["日付", "船", "釣果", "潮回り", "月", "天気", "波高", "風速", "気温", "一致点"]
    display_df = rows[[c for c in cols if c in rows.columns]].sort_values("一致点", ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

rank_tab, data_tab = st.tabs(["📊 条件ランキング", "📁 データ"])
with rank_tab:
    st.subheader("釣れやすい条件ランキング（宝来丸）")
    ranking = make_ranking(df)
    if not ranking:
        st.info("3件以上そろった完全一致条件がまだありません。")
    else:
        rank_df = pd.DataFrame(ranking, columns=["条件", "件数", "20杯以上（生）", "20杯以上（補正）", "平均"])
        rank_df["20杯以上（生）"] = rank_df["20杯以上（生）"].map(lambda x: f"{x:.0f}%")
        rank_df["20杯以上（補正）"] = rank_df["20杯以上（補正）"].map(lambda x: f"{x:.0f}%")
        rank_df["平均"] = rank_df["平均"].map(lambda x: f"{x:.1f}杯")
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("読み込んだCSVデータ")
    st.write(f"**ファイル：** {st.session_state.source} ｜ **日数：** {len(df)}")
    # データ画面ではすべての列（飛龍などのデータも含めて）そのまま確認できるように表示します
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("※風速・気温が空欄の日は `data/data.csv` の18〜23時平均データから自動補完されています。")