from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "chouka.csv"
TARGET = "竿頭(宝来丸)"
OTHER_TARGET = "竿頭(飛翔)"
REQUIRED = ["日付", "潮回り", TARGET, OTHER_TARGET, "二枚潮", "潮の速さ", "月", "天気"]
TIDE_ORDER = ["大潮", "中潮", "小潮", "長潮", "若潮"]
WEATHER_OPTIONS = ["晴", "曇", "雨", "晴曇", "曇晴", "雨曇", "雨晴", "晴雨", "不明"]
SPEED_OPTIONS = ["緩い", "普通", "速い", "カッ飛び", "不明"]
TWO_TIDE_OPTIONS = ["無", "有", "不明"]

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
            .replace("　", ""))


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
    if wanted == OTHER_TARGET:
        for c in df.columns:
            n = normalize_header(c)
            if "竿頭" in n and ("飛翔" in n or "飛龍" in n):
                return c
    return None


def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce"
    )


def normalize_columns(df):
    aliases = {
        TARGET: [TARGET, "竿頭（宝来丸）", "竿頭(宝来丸) ", "竿頭（宝来丸） "],
        OTHER_TARGET: [OTHER_TARGET, "竿頭（飛翔）", "竿頭(飛龍)", "竿頭（飛龍）", "竿頭(飛翔) ", "竿頭（飛翔） "],
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
    df["月"] = to_number(df["月"])
    df[TARGET] = to_number(df[TARGET])
    df[OTHER_TARGET] = to_number(df[OTHER_TARGET])
    for c in ["二枚潮", "潮の速さ", "天気", "潮回り"]:
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


def load_default():
    last = None
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return prepare(pd.read_csv(DEFAULT_CSV, encoding=enc))
        except UnicodeDecodeError as e:
            last = e
    raise last


def parse_count_series(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)")[0], errors="coerce"
    )


def analysis_frame(df, include_other=False):
    parts = []
    if TARGET in df.columns:
        x = df.copy()
        x["釣果"] = parse_count_series(x[TARGET])
        x["船"] = "宝来丸"
        parts.append(x)
    if include_other and OTHER_TARGET in df.columns:
        x = df.copy()
        x["釣果"] = parse_count_series(x[OTHER_TARGET])
        x["船"] = "飛翔"
        parts.append(x)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).dropna(subset=["釣果"])


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
        score += 30
    if pd.notna(row["月"]):
        d = abs(float(row["月"]) - inp["moon"])
        if d <= 1.5:
            score += 25
        elif d <= 3:
            score += 15
    # 二枚潮は重要度を小さく設定
    if clean(row["二枚潮"]) == inp["two"]:
        score += 5
    if clean(row["潮の速さ"]) == inp["speed"]:
        score += 15
    if clean(row["天気"]) == inp["weather"]:
        score += 10
    return score


def predict(df, inp, include_other=False):
    v = analysis_frame(df, include_other)
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
        (v["二枚潮"] == inp["two"]) &
        (v["潮の速さ"] == inp["speed"]) &
        (v["天気"] == inp["weather"]) &
        (v["月齢帯"] == moon_band(inp["moon"]))
    )
    ex = probability(v, exact)

    avg_score = np.mean([x[1] for x in scored]) if scored else 0.0
    max_score = 85.0
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
                confidence=confidence, rows=rows, scored=scored, exact=ex,
                include_other=include_other)


def make_ranking(df, min_n=3, include_other=False):
    base = analysis_frame(df, include_other)
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


# --- CSS: desktopでもスマホでも見やすく ---
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
.hero {background:#172033;color:white;padding:22px 24px;border-radius:14px;margin-bottom:18px;}
.hero h1 {margin:0;font-size:30px;}
.hero p {margin:6px 0 0;color:#cbd3df;}
.small-note {color:#6b7280;font-size:13px;}
.result-card {border:1px solid #e3e7ee;border-radius:12px;padding:16px;background:#fff;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🦑 三国沖 イカメタル釣果予測</h1><p>宝来丸・飛翔の過去釣果から、指定条件に近い日を分析</p></div>', unsafe_allow_html=True)

# セッション内データ
if "df" not in st.session_state:
    try:
        st.session_state.df = load_default()
        st.session_state.source = "data/chouka.csv"
    except Exception:
        st.session_state.df = pd.DataFrame()
        st.session_state.source = ""

with st.sidebar:
    st.header("📁 データ")
    uploaded = st.file_uploader("CSVを読み込む", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.df = read_csv_bytes(uploaded)
            st.session_state.source = uploaded.name
            st.success(f"{uploaded.name} を読み込みました")
        except Exception as e:
            st.error(f"CSV読込エラー：{e}")
    include_other = st.checkbox("飛翔（隣船）の釣果も予測に含める", value=False)
    st.caption("ON：宝来丸＋飛翔 / OFF：宝来丸のみ")
    st.divider()
    st.caption("二枚潮の予測スコアは5点。重要度を低めに設定しています。")


df = st.session_state.df
if df.empty:
    st.warning("CSVが読み込まれていません。左側からCSVを選択してください。")
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
with c3:
    weather = st.selectbox("天気", WEATHER_OPTIONS, index=0)
    speed = st.selectbox("潮の速さ", SPEED_OPTIONS, index=2)
    two = st.selectbox("二枚潮", TWO_TIDE_OPTIONS, index=0)

inp = {"tide": tide, "moon": moon, "weather": weather, "speed": speed, "two": two}

if st.button("🔍 釣果を予測する", type="primary", use_container_width=True):
    r = predict(df, inp, include_other=include_other)
    if r is None:
        st.warning("分析できる釣果データがありません。")
    else:
        st.session_state.result = r
        st.session_state.result_date = prediction_date

if "result" in st.session_state:
    r = st.session_state.result
    st.subheader(f"📅 {st.session_state.result_date:%Y/%m/%d} の予測")
    a, b, c, d = st.columns(4)
    a.metric("20杯以上", f"{r['p20']:.0f}%")
    b.metric("30杯以上", f"{r['p30']:.0f}%")
    c.metric("40杯以上", f"{r['p40']:.0f}%")
    d.metric("予想レンジ", f"{r['low']:.0f}〜{r['high']:.0f}杯")
    st.progress(min(1.0, r["confidence"] / 100))
    st.caption(f"信頼度 {r['confidence']:.0f}/100　※当たる確率ではなく、過去データの支え具合を示す指標")

    mode = "宝来丸＋飛翔" if include_other else "宝来丸のみ"
    st.write(f"**分析対象：** {mode}　｜　**入力：** {tide} × 月齢{moon:.1f} × {weather} × {speed} × 二枚潮{two}")
    if r["exact"]:
        st.info(f"★ 5条件一致（同じ月齢帯）：{r['exact']['n']}件 / 20杯以上 {r['exact']['p20']:.1f}%")

    st.subheader("🔎 似ている過去データ")
    rows = r["rows"].copy()
    rows["日付"] = rows["__date"].dt.strftime("%Y/%m/%d")
    score_map = {idx: score for idx, score in r["scored"]}
    rows["一致点"] = [score_map.get(i, 0) for i in rows.index]
    cols = ["日付", "船", "釣果", "潮回り", "月", "天気", "潮の速さ", "二枚潮", "一致点"]
    st.dataframe(rows[cols].sort_values("一致点", ascending=False), use_container_width=True, hide_index=True)

    with st.expander("予測の詳細"):
        st.write(f"重み付き平均：{r['mean']:.1f}杯")
        st.write(f"分析した類似データ：{len(r['rows'])}件")
        st.write("二枚潮の一致は5点として計算。")

st.divider()

# タブ
rank_tab, data_tab = st.tabs(["📊 条件ランキング", "📁 データ"])
with rank_tab:
    st.subheader("釣れやすい条件ランキング")
    ranking = make_ranking(df, include_other=include_other)
    if not ranking:
        st.info("3件以上そろった完全一致条件がまだありません。")
    else:
        rank_df = pd.DataFrame(ranking, columns=["条件", "件数", "20杯以上（生）", "20杯以上（補正）", "平均"])
        rank_df["20杯以上（生）"] = rank_df["20杯以上（生）"].map(lambda x: f"{x:.0f}%")
        rank_df["20杯以上（補正）"] = rank_df["20杯以上（補正）"].map(lambda x: f"{x:.0f}%")
        rank_df["平均"] = rank_df["平均"].map(lambda x: f"{x:.1f}杯")
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
    st.caption("※同じデータを親条件・子条件に重複表示しないよう、4条件（潮回り×月齢帯×潮速×天気）の完全一致だけを比較。3件未満は除外。20杯以上率は少数データを補正しています。")

with data_tab:
    st.subheader("読み込んだCSV")
    st.write(f"**ファイル：** {st.session_state.source}　｜　**日数：** {len(df)}")
    display_cols = [c for c in ["日付", "潮回り", TARGET, "船中合計", "人数", OTHER_TARGET, "二枚潮", "潮の速さ", "月", "天気"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
