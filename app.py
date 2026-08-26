from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "chouka.csv"
JMA_CSV = APP_DIR / "data" / "data.csv"

TARGET = "竿頭(宝来丸)"

REQUIRED = [
    "日付",
    "潮回り",
    TARGET,
    "潮の速さ",
    "月",
    "天気",
    "波高",
    "風速",
    "気温"
]

TIDE_ORDER = ["大潮", "中潮", "小潮", "長潮", "若潮"]

WEATHER_OPTIONS = [
    "晴",
    "曇",
    "雨",
    "晴曇",
    "曇晴",
    "雨曇",
    "曇雨",
    "晴雨",
    "雨晴",
    "晴曇雨",
    "晴雨曇",
    "曇晴雨",
    "曇雨晴",
    "雨晴曇",
    "雨曇晴",
    "不明"
]

SPEED_OPTIONS = [
    "緩い",
    "普通",
    "速い",
    "カッ飛び",
    "不明"
]

st.set_page_config(
    page_title="宝来丸 イカメタル釣果予測",
    page_icon="🦑",
    layout="wide"
)


# =========================================================
# 基本関数
# =========================================================

def clean(v):
    return "" if pd.isna(v) else str(v).strip()


def normalize_weather(weather):
    """
    天気表記を統一。
    「雲」と「曇」は同じものとして扱う。
    """
    weather = clean(weather)

    if not weather:
        return ""

    return weather.replace("雲", "曇")


def weather_sequence(weather):
    """
    晴曇雨 → ["晴", "曇", "雨"]
    曇 → ["曇"]
    """
    weather = normalize_weather(weather)

    if not weather:
        return []

    return list(weather)


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


def calculate_moon_age(date_value):
    """
    2000/1/7 03:14 JST付近の新月を基準に
    29.530588日の周期で月齢を概算。
    """

    try:
        dt = pd.to_datetime(date_value)

        reference = pd.Timestamp("2000-01-07 03:14")

        days = (dt - reference).total_seconds() / 86400

        return days % 29.530588

    except Exception:
        return np.nan


def normalize_header(x):
    return (
        str(x)
        .replace("\ufeff", "")
        .strip()
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
    )


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
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(
            r"(-?\d+(?:\.\d+)?)",
            expand=False
        ),
        errors="coerce"
    )


# =========================================================
# 気象庁CSV
# =========================================================

@st.cache_data
def load_jma_cache():

    if not JMA_CSV.exists():
        return {}, {}

    try:

        df_jma = None

        for enc in ["utf-8-sig", "cp932", "utf-8"]:

            try:

                df_jma = pd.read_csv(
                    JMA_CSV,
                    encoding=enc,
                    header=None
                )

                break

            except Exception:
                continue

        if df_jma is None:
            return {}, {}

        if df_jma.shape[1] < 3:
            return {}, {}

        start_row = 0

        first_val = str(df_jma.iloc[0, 2])

        if (
            "風速" in first_val
            or "wind" in first_val.lower()
            or "気温" in first_val
        ):
            start_row = 1

        df_jma = df_jma.iloc[start_row:].copy()

        df_jma["dt"] = pd.to_datetime(
            df_jma.iloc[:, 0].astype(str).str.strip()
            + " "
            + df_jma.iloc[:, 1].astype(str).str.strip(),
            errors="coerce"
        )

        # 風速
        df_jma["wind"] = pd.to_numeric(
            df_jma.iloc[:, 2],
            errors="coerce"
        )

        # 気温列を探す
        temp_col_idx = None

        for c_idx in range(3, df_jma.shape[1]):

            col_vals = pd.to_numeric(
                df_jma.iloc[:, c_idx],
                errors="coerce"
            )

            valid = col_vals.dropna()

            if len(valid) == 0:
                continue

            if valid.between(-15, 45).mean() > 0.8:

                temp_col_idx = c_idx
                break

        if temp_col_idx is not None:

            df_jma["temp"] = pd.to_numeric(
                df_jma.iloc[:, temp_col_idx],
                errors="coerce"
            )

        df_jma["hour"] = df_jma["dt"].dt.hour

        df_jma["date_str"] = (
            df_jma["dt"]
            .dt.strftime("%Y/%m/%d")
        )

        # 18〜23時
        mask = df_jma["hour"].isin(
            [18, 19, 20, 21, 22, 23]
        )

        sub = df_jma[mask]

        if "wind" in sub.columns:

            wind_dict = (
                sub.groupby("date_str")["wind"]
                .mean()
                .to_dict()
            )

        else:
            wind_dict = {}

        if "temp" in sub.columns:

            temp_dict = (
                sub.groupby("date_str")["temp"]
                .mean()
                .to_dict()
            )

        else:
            temp_dict = {}

        return wind_dict, temp_dict

    except Exception as e:

        print(
            f"気象庁CSV読込エラー: {e}"
        )

        return {}, {}


def fetch_jma_val(date_str, cache_dict):

    if not cache_dict:
        return np.nan

    try:

        dt = pd.to_datetime(date_str)

        key = dt.strftime("%Y/%m/%d")

        if key in cache_dict:
            return cache_dict[key]

        return np.nan

    except Exception:
        return np.nan


# =========================================================
# CSV読み込み
# =========================================================

def normalize_columns(df):

    aliases = {

        TARGET: [
            TARGET,
            "竿頭（宝来丸）",
            "竿頭(宝来丸) ",
            "竿頭（宝来丸） "
        ]

    }

    rename = {}

    cols = {
        str(c).strip(): c
        for c in df.columns
    }

    for canonical, names in aliases.items():

        for n in names:

            if n in cols:

                rename[cols[n]] = canonical

                break

    return df.rename(columns=rename)


def prepare(df):

    df = normalize_columns(df.copy())

    df.columns = [
        normalize_header(c)
        for c in df.columns
    ]

    for c in REQUIRED:

        if c not in df.columns:

            found = find_column(
                df,
                c
            )

            if found is not None:
                df[c] = df[found]

            else:
                df[c] = ""

    # 数値化
    df["波高"] = to_number(df["波高"])
    df["風速"] = to_number(df["風速"])
    df["気温"] = to_number(df["気温"])
    df["月"] = to_number(df["月"])

    # 気象庁データ
    wind_cache, temp_cache = load_jma_cache()

    for idx, row in df.iterrows():

        date_val = row["日付"]

        if pd.isna(date_val):
            continue

        if pd.isna(row["風速"]):

            val = fetch_jma_val(
                date_val,
                wind_cache
            )

            if pd.notna(val):
                df.at[idx, "風速"] = val

        if pd.isna(row["気温"]):

            val = fetch_jma_val(
                date_val,
                temp_cache
            )

            if pd.notna(val):
                df.at[idx, "気温"] = val

    # 最終的にデータがない場合のデフォルト
    df["風速"] = df["風速"].fillna(2.5)
    df["気温"] = df["気温"].fillna(22.0)

    # 月齢
    # CSVに月が入っていればそれを使用
    # なければ日付から自動計算
    date_series = pd.to_datetime(
        df["日付"],
        errors="coerce"
    )

    calculated_moon = date_series.map(
        calculate_moon_age
    )

    df["月"] = df["月"].where(
        df["月"].notna(),
        calculated_moon
    )

    df[TARGET] = to_number(df[TARGET])

    # 文字列項目
    for c in [
        "潮の速さ",
        "天気",
        "潮回り"
    ]:

        df[c] = (
            df[c]
            .map(clean)
            .replace("", "不明")
        )

    df["天気"] = (
        df["天気"]
        .map(normalize_weather)
        .replace("", "不明")
    )

    df["月齢帯"] = df["月"].map(
        moon_band
    )

    df["__date"] = pd.to_datetime(
        df["日付"],
        errors="coerce"
    ).dt.normalize()

    return df


def read_csv_bytes(file_obj):

    raw = file_obj.getvalue()

    last = None

    for enc in (
        "utf-8-sig",
        "cp932",
        "utf-8"
    ):

        try:

            from io import BytesIO

            return prepare(
                pd.read_csv(
                    BytesIO(raw),
                    encoding=enc
                )
            )

        except UnicodeDecodeError as e:

            last = e

    raise last


@st.cache_data
def load_default_cached():

    if not DEFAULT_CSV.exists():
        return pd.DataFrame()

    last = None

    for enc in (
        "utf-8-sig",
        "cp932",
        "utf-8"
    ):

        try:

            return prepare(
                pd.read_csv(
                    DEFAULT_CSV,
                    encoding=enc
                )
            )

        except UnicodeDecodeError as e:

            last = e

    if last:
        raise last

    return pd.DataFrame()


# =========================================================
# 天気パターン
# =========================================================

def get_daily_weather(df):

    """
    日付ごとの天気を辞書化。

    同じ日に複数行ある場合は、
    最初に取得できた有効な天気を使用。
    """

    daily = {}

    work = df.copy()

    if "__date" not in work.columns:

        work["__date"] = pd.to_datetime(
            work["日付"],
            errors="coerce"
        ).dt.normalize()

    for _, row in work.sort_values(
        "__date"
    ).iterrows():

        d = row["__date"]

        if pd.isna(d):
            continue

        weather = normalize_weather(
            row.get("天気", "")
        )

        if (
            weather
            and weather != "不明"
            and d not in daily
        ):

            daily[d] = weather

    return daily


def get_weather_pattern(
    df,
    date_value,
    current_weather=None
):

    """
    前日→当日の天気パターンを取得。

    重要：
    当日の天気は予測条件として入力されたものを使用。
    前日の天気は過去CSVから取得。

    例：
    前日 雨
    当日 雨曇

    → 雨→雨曇
    """

    try:

        target_date = pd.to_datetime(
            date_value
        ).normalize()

    except Exception:

        return {
            "previous": "",
            "current": "",
            "pattern": "",
            "change_type": "不明"
        }

    daily = get_daily_weather(df)

    previous_date = (
        target_date
        - pd.Timedelta(days=1)
    )

    previous = daily.get(
        previous_date,
        ""
    )

    if current_weather is None:

        current = daily.get(
            target_date,
            ""
        )

    else:

        current = normalize_weather(
            current_weather
        )

    if previous and current:

        pattern = (
            f"{previous}→{current}"
        )

    elif current:

        pattern = current

    else:

        pattern = ""

    return {
        "previous": previous,
        "current": current,
        "pattern": pattern,
        "change_type": classify_weather_change(
            previous,
            current
        )
    }


def weather_close(a, b):

    a = normalize_weather(a)
    b = normalize_weather(b)

    if not a or not b:
        return False

    if a == b:
        return True

    a_set = set(
        weather_sequence(a)
    )

    b_set = set(
        weather_sequence(b)
    )

    if not a_set or not b_set:
        return False

    return len(
        a_set & b_set
    ) > 0


def classify_weather_change(
    previous,
    current
):

    if not previous or not current:
        return "不明"

    p = weather_sequence(previous)
    c = weather_sequence(current)

    if not p or not c:
        return "不明"

    p_last = p[-1]
    c_last = c[-1]

    if p == c:
        return "安定"

    if p_last == c_last:
        return "小変化"

    order = {
        "晴": 0,
        "曇": 1,
        "雨": 2
    }

    if (
        p_last in order
        and c_last in order
    ):

        if order[c_last] > order[p_last]:
            return "悪化"

        if order[c_last] < order[p_last]:
            return "回復"

    return "変化"


# =========================================================
# 天気パターンの点数
# =========================================================

def weather_pattern_score(
    target_pattern,
    row_pattern
):

    """
    天気パターンの近さを0〜12点。

    例：

    雨→雨曇
    雨→雨曇

    → 12点

    雨→雨曇
    雨→曇

    → 10点前後

    晴→雨
    雨→晴

    → 低得点
    """

    if (
        not target_pattern
        or not row_pattern
    ):
        return 0

    target = target_pattern.split("→")
    row = row_pattern.split("→")

    if len(target) != 2:
        return 0

    if len(row) != 2:
        return 0

    target_prev = normalize_weather(
        target[0]
    )

    target_now = normalize_weather(
        target[1]
    )

    row_prev = normalize_weather(
        row[0]
    )

    row_now = normalize_weather(
        row[1]
    )

    score = 0

    # -----------------------------------------------------
    # 前日の天気
    # -----------------------------------------------------

    if target_prev == row_prev:

        score += 4

    elif weather_close(
        target_prev,
        row_prev
    ):

        score += 2

    # -----------------------------------------------------
    # 当日の天気
    # -----------------------------------------------------

    if target_now == row_now:

        score += 4

    elif weather_close(
        target_now,
        row_now
    ):

        score += 2

    # -----------------------------------------------------
    # 変化方向
    # -----------------------------------------------------

    target_change = classify_weather_change(
        target_prev,
        target_now
    )

    row_change = classify_weather_change(
        row_prev,
        row_now
    )

    if target_change == row_change:

        score += 4

    # 最大12点
    return min(score, 12)


# =========================================================
# 釣果データ
# =========================================================

def parse_count_series(series):

    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(
            r"([-+]?\d+(?:\.\d+)?)"
        )[0],
        errors="coerce"
    )


def analysis_frame(df):

    if TARGET not in df.columns:
        return pd.DataFrame()

    x = df.copy()

    x["釣果"] = parse_count_series(
        x[TARGET]
    )

    x["船"] = "宝来丸"

    return x.dropna(
        subset=["釣果"]
    )


def probability(df, mask):

    x = df.loc[
        mask,
        "釣果"
    ].dropna()

    if len(x) == 0:
        return None

    return {
        "n": len(x),
        "p20": (x >= 20).mean() * 100,
        "p30": (x >= 30).mean() * 100,
        "p40": (x >= 40).mean() * 100,
        "mean": x.mean(),
        "median": x.median()
    }


# =========================================================
# スコア計算
# =========================================================

def similarity(
    row,
    inp,
    df
):

    """
    100点満点。

    潮回り      10
    月齢         8
    潮の速さ    10
    当日天気     8
    天気パターン 12
    波高        15
    風速        25
    気温        12
    ----------------
    合計       100
    """

    score = 0

    breakdown = {}

    # -----------------------------------------------------
    # ① 潮回り 10点
    # -----------------------------------------------------

    tide_score = 0

    if clean(row["潮回り"]) == inp["tide"]:

        tide_score = 10

    score += tide_score
    breakdown["潮回り"] = tide_score

    # -----------------------------------------------------
    # ② 月齢 8点
    # -----------------------------------------------------

    moon_score = 0

    if pd.notna(row["月"]):

        d = abs(
            float(row["月"])
            - inp["moon"]
        )

        # 月齢は29.53日周期なので
        # 端と端も近いものとして扱う
        d = min(
            d,
            29.530588 - d
        )

        if d <= 1.5:

            moon_score = 8

        elif d <= 3:

            moon_score = 4

    score += moon_score
    breakdown["月齢"] = moon_score

    # -----------------------------------------------------
    # ③ 潮の速さ 10点
    # -----------------------------------------------------

    speed_score = 0

    if clean(
        row["潮の速さ"]
    ) == inp["speed"]:

        speed_score = 10

    score += speed_score
    breakdown["潮の速さ"] = speed_score

    # -----------------------------------------------------
    # ④ 当日の天気 8点
    # -----------------------------------------------------

    weather_score = 0

    row_weather = normalize_weather(
        row["天気"]
    )

    input_weather = normalize_weather(
        inp["weather"]
    )

    if row_weather == input_weather:

        weather_score = 8

    elif weather_close(
        row_weather,
        input_weather
    ):

        weather_score = 4

    score += weather_score
    breakdown["当日天気"] = weather_score

    # -----------------------------------------------------
    # ⑤ 天気パターン 12点
    # -----------------------------------------------------

    pattern_score = 0

    target_pattern = inp.get(
        "weather_pattern",
        ""
    )

    if target_pattern:

        row_weather_info = get_weather_pattern(
            df,
            row["日付"],
            row["天気"]
        )

        row_pattern = (
            row_weather_info["pattern"]
        )

        pattern_score = weather_pattern_score(
            target_pattern,
            row_pattern
        )

    score += pattern_score
    breakdown["天気パターン"] = pattern_score

    # -----------------------------------------------------
    # ⑥ 波高 15点
    # -----------------------------------------------------

    wave_score = 0

    if pd.notna(row["波高"]):

        diff = abs(
            float(row["波高"])
            - inp["wave"]
        )

        if diff <= 0.3:

            wave_score = 15

        elif diff <= 0.8:

            wave_score = 8

        elif diff <= 1.2:

            wave_score = 3

    score += wave_score
    breakdown["波高"] = wave_score

    # -----------------------------------------------------
    # ⑦ 風速 25点
    # -----------------------------------------------------

    wind_score = 0

    if pd.notna(row["風速"]):

        diff = abs(
            float(row["風速"])
            - inp["wind"]
        )

        if diff <= 0.5:

            wind_score = 25

        elif diff <= 1.0:

            wind_score = 20

        elif diff <= 2.0:

            wind_score = 12

        elif diff <= 3.5:

            wind_score = 5

    score += wind_score
    breakdown["風速"] = wind_score

    # -----------------------------------------------------
    # ⑧ 気温 12点
    # -----------------------------------------------------

    temp_score = 0

    if pd.notna(row["気温"]):

        diff = abs(
            float(row["気温"])
            - inp["temp"]
        )

        if diff <= 1.0:

            temp_score = 12

        elif diff <= 2.5:

            temp_score = 8

        elif diff <= 4.5:

            temp_score = 4

    score += temp_score
    breakdown["気温"] = temp_score

    return score, breakdown


# =========================================================
# 予測
# =========================================================

def predict(df, inp):

    v = analysis_frame(df)

    if v.empty:
        return None

    scored = []

    for idx, row in v.iterrows():

        score, breakdown = similarity(
            row,
            inp,
            df
        )

        scored.append(
            (
                idx,
                score,
                breakdown
            )
        )

    scored.sort(
        key=lambda x: x[1],
        reverse=True
    )

    scored = scored[:10]

    rows = v.loc[
        [x[0] for x in scored]
    ].copy()

    # 重み
    w = np.array(
        [
            max(1, x[1])
            for x in scored
        ],
        dtype=float
    )

    y = rows[
        "釣果"
    ].to_numpy(float)

    p20 = np.average(
        (y >= 20).astype(float),
        weights=w
    ) * 100

    p30 = np.average(
        (y >= 30).astype(float),
        weights=w
    ) * 100

    p40 = np.average(
        (y >= 40).astype(float),
        weights=w
    ) * 100

    mean = np.average(
        y,
        weights=w
    )

    q25, q75 = (
        np.percentile(
            y,
            [25, 75]
        )
        if len(y) > 1
        else (y[0], y[0])
    )

    # -----------------------------------------------------
    # 近接条件一致
    # -----------------------------------------------------

    exact = (
        (v["潮回り"] == inp["tide"])
        &
        (v["潮の速さ"] == inp["speed"])
        &
        (v["天気"] == inp["weather"])
        &
        (
            v["月齢帯"]
            == moon_band(inp["moon"])
        )
        &
        v["波高"].notna()
        &
        (
            (
                v["波高"]
                - inp["wave"]
            ).abs()
            <= 0.5
        )
    )

    ex = probability(
        v,
        exact
    )

    # -----------------------------------------------------
    # 信頼度
    # -----------------------------------------------------

    avg_score = np.mean(
        [x[1] for x in scored]
    ) if scored else 0

    similarity_factor = min(
        1.0,
        avg_score / 100.0
    )

    n_factor = (
        1.0
        - np.exp(
            -len(rows) / 8.0
        )
    )

    dispersion = (
        float(np.std(y))
        if len(y) > 1
        else 0.0
    )

    mean_abs = max(
        float(
            np.mean(
                np.abs(y)
            )
        ),
        1.0
    )

    consistency_factor = (
        1.0
        /
        (
            1.0
            + dispersion
            / mean_abs
        )
    )

    confidence = 100.0 * (
        0.45 * n_factor
        + 0.35 * similarity_factor
        + 0.20 * consistency_factor
    )

    if len(rows) < 5:

        confidence = min(
            confidence,
            65.0
        )

    elif len(rows) < 10:

        confidence = min(
            confidence,
            80.0
        )

    elif len(rows) < 20:

        confidence = min(
            confidence,
            90.0
        )

    else:

        confidence = min(
            confidence,
            97.0
        )

    return {
        "p20": p20,
        "p30": p30,
        "p40": p40,
        "mean": mean,
        "low": q25,
        "high": q75,
        "confidence": confidence,
        "rows": rows,
        "scored": scored,
        "exact": ex
    }


# =========================================================
# 条件ランキング
# =========================================================

def make_ranking(
    df,
    min_n=3
):

    base = analysis_frame(df)

    if base.empty:
        return []

    fields = [
        "潮回り",
        "月齢帯",
        "潮の速さ",
        "天気"
    ]

    work = base.copy()

    for c in fields:

        work[c] = (
            work[c]
            .map(clean)
        )

    work = work[
        ~work[fields]
        .isin(["", "不明"])
        .any(axis=1)
    ].copy()

    out = []

    for keys, g in work.groupby(
        fields,
        dropna=False
    ):

        if len(g) < min_n:
            continue

        y = g[
            "釣果"
        ].astype(float)

        n = len(y)

        raw = (
            (y >= 20).mean()
            * 100
        )

        corrected = (
            raw * n
            + 50.0 * 3
        ) / (
            n + 3
        )

        out.append(
            (
                " × ".join(
                    map(str, keys)
                ),
                n,
                raw,
                corrected,
                y.mean()
            )
        )

    out.sort(
        key=lambda x: (
            x[3],
            x[1],
            x[4]
        ),
        reverse=True
    )

    return out[:20]


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.hero {
    background:#172033;
    color:white;
    padding:22px 24px;
    border-radius:14px;
    margin-bottom:18px;
}

.hero h1 {
    margin:0;
    font-size:30px;
}

.hero p {
    margin:6px 0 0;
    color:#cbd3df;
}

</style>
""",
    unsafe_allow_html=True
)


st.markdown(
    """
<div class="hero">
<h1>🦑 宝来丸 イカメタル釣果予測</h1>
<p>
宝来丸の過去釣果と潮・天気・波・風速・気温から類似条件を分析
</p>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# セッションデータ
# =========================================================

if "df" not in st.session_state:

    df_default = load_default_cached()

    st.session_state.df = df_default

    st.session_state.source = (
        "data/chouka.csv"
        if not df_default.empty
        else ""
    )


# =========================================================
# サイドバー
# =========================================================

with st.sidebar:

    st.header("📁 データ設定")

    uploaded = st.file_uploader(
        "釣果CSVを読み込む",
        type=["csv"]
    )

    if uploaded is not None:

        try:

            st.session_state.df = (
                read_csv_bytes(uploaded)
            )

            st.session_state.source = (
                uploaded.name
            )

            st.success(
                f"{uploaded.name} を読み込みました"
            )

        except Exception as e:

            st.error(
                f"CSV読込エラー：{e}"
            )

    st.divider()

    wind_c, temp_c = (
        load_jma_cache()
    )

    jma_status = (
        "有効 (data/data.csv)"
        if (wind_c or temp_c)
        else "未検出"
    )

    st.caption(
        f"気象庁データ連携: {jma_status}"
    )

    st.divider()

    st.markdown(
        """
### 📊 スコア配分

- 潮回り：10点
- 月齢：8点
- 潮の速さ：10点
- 当日天気：8点
- 天気パターン：12点
- 波高：15点
- **風速：25点**
- 気温：12点

**合計：100点**
"""
    )


# =========================================================
# データ確認
# =========================================================

df = st.session_state.df

if df.empty:

    st.warning(
        "CSVが読み込まれていません。"
        "左側からCSVをアップロードするか、"
        "data/chouka.csvを配置してください。"
    )

    st.stop()


# =========================================================
# 入力欄
# =========================================================

st.subheader("🔮 予測条件")

c1, c2, c3 = st.columns(3)


with c1:

    prediction_date = st.date_input(
        "予測日",
        value=datetime.now().date()
    )

    tide_present = [
        x
        for x in TIDE_ORDER
        if x in set(df["潮回り"])
    ]

    tide_list = (
        tide_present
        or TIDE_ORDER
    )

    tide = st.selectbox(
        "潮回り",
        tide_list,
        index=(
            tide_list.index("中潮")
            if "中潮" in tide_list
            else 0
        )
    )


with c2:

    # 日付から月齢を自動計算
    calculated_moon = calculate_moon_age(
        prediction_date
    )

    moon = st.number_input(
        "月齢",
        min_value=0.0,
        max_value=29.53,
        value=float(
            round(
                calculated_moon,
                1
            )
        ),
        step=0.1
    )

    st.info(
        f"月齢帯：**{moon_band(moon)}**"
    )

    weather = st.selectbox(
        "天気",
        WEATHER_OPTIONS,
        index=0
    )


with c3:

    speed = st.selectbox(
        "潮の速さ",
        SPEED_OPTIONS,
        index=2
    )

    wave = st.number_input(
        "波高 (m)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1
    )

    wind = st.number_input(
        "18-23時風速 (m/s)",
        min_value=0.0,
        max_value=25.0,
        value=3.0,
        step=0.5
    )

    temp = st.number_input(
        "18-23時気温 (℃)",
        min_value=-10.0,
        max_value=40.0,
        value=22.0,
        step=0.5
    )


# =========================================================
# 天気パターン表示
# =========================================================

weather_info = get_weather_pattern(
    df,
    prediction_date,
    weather
)

st.info(
    f"🌤️ 前日の天気："
    f"**{weather_info['previous'] or 'データなし'}**"
    f"　→　"
    f"当日：**{weather}**"
    f"　｜ 天気パターン："
    f"**{weather_info['pattern'] or '判定不可'}**"
    f"　｜ 変化："
    f"**{weather_info['change_type']}**"
)


# =========================================================
# 予測
# =========================================================

inp = {

    "tide": tide,

    "moon": moon,

    "weather": weather,

    "speed": speed,

    "wave": wave,

    "wind": wind,

    "temp": temp,

    "previous_weather":
        weather_info["previous"],

    "weather_pattern":
        weather_info["pattern"],

    "weather_change":
        weather_info["change_type"]
}


if st.button(
    "🔍 釣果を予測する",
    type="primary",
    use_container_width=True
):

    r = predict(
        df,
        inp
    )

    if r is None:

        st.warning(
            "分析できる釣果データがありません。"
        )

    else:

        st.session_state.result = r

        st.session_state.result_date = (
            prediction_date
        )


# =========================================================
# 結果
# =========================================================

if "result" in st.session_state:

    r = st.session_state.result

    result_date = (
        st.session_state.result_date
    )

    st.subheader(
        f"📅 {result_date:%Y/%m/%d} "
        "の予測結果（宝来丸）"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "20杯以上",
        f"{r['p20']:.0f}%"
    )

    b.metric(
        "30杯以上",
        f"{r['p30']:.0f}%"
    )

    c.metric(
        "40杯以上",
        f"{r['p40']:.0f}%"
    )

    d.metric(
        "予想レンジ",
        f"{r['low']:.0f}〜{r['high']:.0f}杯"
    )

    st.progress(
        min(
            1.0,
            r["confidence"] / 100
        )
    )

    st.caption(
        f"信頼度 {r['confidence']:.0f}/100 "
        "※類似件数・条件一致度・釣果のばらつきから算出"
    )

    st.write(
        f"**分析対象：** 宝来丸"
        f" ｜ **入力：** "
        f"{tide}"
        f" × 月齢{moon:.1f}"
        f" × {weather}"
        f" × 潮速:{speed}"
        f" × 波高{wave}m"
        f" × 風速{wind}m/s"
        f" × 気温{temp}℃"
    )

    st.write(
        f"**天気パターン：** "
        f"{weather_info['pattern'] or '判定不可'}"
        f"　（{weather_info['change_type']}）"
    )

    if r["exact"]:

        st.info(
            f"★ 近接条件一致: "
            f"{r['exact']['n']}件"
            f" / 20杯以上 "
            f"{r['exact']['p20']:.1f}%"
        )

    # =====================================================
    # 類似過去データ
    # =====================================================

    st.subheader(
        "🔎 似ている過去データ"
    )

    rows = r["rows"].copy()

    rows["日付"] = (
        rows["__date"]
        .dt.strftime("%Y/%m/%d")
    )

    score_map = {
        idx: score
        for idx, score, breakdown
        in r["scored"]
    }

    breakdown_map = {
        idx: breakdown
        for idx, score, breakdown
        in r["scored"]
    }

    rows["一致点"] = [
        score_map.get(
            i,
            0
        )
        for i in rows.index
    ]

    # -----------------------------------------------------
    # 天気パターンを追加
    # -----------------------------------------------------

    rows["天気パターン"] = ""

    for idx in rows.index:

        x = rows.loc[idx]

        info = get_weather_pattern(
            df,
            x["日付"],
            x["天気"]
        )

        rows.at[
            idx,
            "天気パターン"
        ] = info["pattern"]

    # -----------------------------------------------------
    # 点数内訳
    # -----------------------------------------------------

    rows["点数内訳"] = ""

    for idx in rows.index:

        bd = breakdown_map.get(
            idx,
            {}
        )

        rows.at[
            idx,
            "点数内訳"
        ] = (
            f"潮{bd.get('潮回り',0)}"
            f" / 月{bd.get('月齢',0)}"
            f" / 潮速{bd.get('潮の速さ',0)}"
            f" / 天気{bd.get('当日天気',0)}"
            f" / パターン{bd.get('天気パターン',0)}"
            f" / 波{bd.get('波高',0)}"
            f" / 風{bd.get('風速',0)}"
            f" / 気温{bd.get('気温',0)}"
        )

    cols = [
        "日付",
        "船",
        "釣果",
        "潮回り",
        "月",
        "天気",
        "天気パターン",
        "波高",
        "風速",
        "気温",
        "一致点",
        "点数内訳"
    ]

    display_df = rows[
        [
            c
            for c in cols
            if c in rows.columns
        ]
    ].sort_values(
        "一致点",
        ascending=False
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "点数内訳："
        "潮回り / 月齢 / 潮速 / 当日天気 / "
        "天気パターン / 波高 / 風速 / 気温"
    )


# =========================================================
# ランキング・データ
# =========================================================

st.divider()

rank_tab, data_tab = st.tabs(
    [
        "📊 条件ランキング",
        "📁 データ"
    ]
)


with rank_tab:

    st.subheader(
        "釣れやすい条件ランキング（宝来丸）"
    )

    ranking = make_ranking(df)

    if not ranking:

        st.info(
            "3件以上そろった完全一致条件がまだありません。"
        )

    else:

        rank_df = pd.DataFrame(
            ranking,
            columns=[
                "条件",
                "件数",
                "20杯以上（生）",
                "20杯以上（補正）",
                "平均"
            ]
        )

        rank_df[
            "20杯以上（生）"
        ] = rank_df[
            "20杯以上（生）"
        ].map(
            lambda x:
            f"{x:.0f}%"
        )

        rank_df[
            "20杯以上（補正）"
        ] = rank_df[
            "20杯以上（補正）"
        ].map(
            lambda x:
            f"{x:.0f}%"
        )

        rank_df[
            "平均"
        ] = rank_df[
            "平均"
        ].map(
            lambda x:
            f"{x:.1f}杯"
        )

        st.dataframe(
            rank_df,
            use_container_width=True,
            hide_index=True
        )


with data_tab:

    st.subheader(
        "読み込んだCSVデータ"
    )

    st.write(
        f"**ファイル：** "
        f"{st.session_state.source}"
        f" ｜ **日数：** {len(df)}"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "※風速・気温が空欄の日は "
        "`data/data.csv` の18〜23時平均データから自動補完されています。"
    )