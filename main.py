import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "chouka.csv"
JMA_CSV = APP_DIR / "data" / "data.csv"

TARGET = "竿頭(宝来丸)"
OTHER_TARGET = "竿頭(飛翔)"
# 必須列に気温を追加（CSV側にない場合は空欄になります）
REQUIRED = ["日付", "潮回り", TARGET, OTHER_TARGET, "潮の速さ", "月", "天気", "波高", "風速", "気温"]

TIDE_ORDER = ["大潮","中潮","小潮","長潮","若潮"]
WEATHER_OPTIONS = ["晴","曇","雨","晴曇","曇晴","雨曇","雨晴","晴雨","不明"]
SPEED_OPTIONS = ["緩い","普通","速い","カッ飛び","不明"]


def clean(v):
    return "" if pd.isna(v) else str(v).strip()

def moon_band(m):
    if pd.isna(m): return "不明"
    m = float(m) % 29.53
    if m < 7.5: return "新月〜上弦"
    if m < 15: return "上弦〜満月"
    if m < 22.5: return "満月〜下弦"
    return "下弦〜新月"

def _normalize_header(x):
    return (str(x).replace("\ufeff", "").strip()
            .replace("（", "(").replace("）", ")")
            .replace(" ", ""))

def _find_column(df, wanted):
    wanted_n = _normalize_header(wanted)
    for c in df.columns:
        if _normalize_header(c) == wanted_n:
            return c
    if wanted == TARGET:
        for c in df.columns:
            n = _normalize_header(c)
            if "竿頭" in n and "宝来丸" in n:
                return c
    return None

def _to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
                      .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce"
    )

# --- data/data.csv から18〜23時の平均風速・平均気温を引くキャッシュ用辞書 ---
_jma_wind_cache = {}
_jma_temp_cache = {}

def load_jma_cache():
    global _jma_wind_cache, _jma_temp_cache
    if not JMA_CSV.exists():
        return
        
    try:
        df_jma = None
        for enc in ['utf-8-sig', 'cp932', 'utf-8']:
            try:
                df_jma = pd.read_csv(JMA_CSV, encoding=enc, header=None)
                break
            except Exception:
                continue
                
        if df_jma is None:
            return
            
        # 気象庁CSVの列構成を推測 (日付, 時間, 風速, 気温など)
        # 一般的な気象庁ダウンロードCSVの場合: 列0=日付, 列1=時間, 列2=風速, 列4=気温 などになりやすいが、
        # ファイルのフォーマットに合わせて柔軟に数値列を探すか、位置を指定します。
        # ここでは一般的なフォーマット（列2=風速, 列4=気温 またはそれに準ずるもの）を想定、
        # もしくは数値が入っている列を安全に探します。
        if df_jma.shape[1] >= 3:
            start_row = 0
            first_val = str(df_jma.iloc[0, 2])
            if "風速" in first_val or "wind" in first_val.lower() or "気温" in first_val:
                start_row = 1
                
            df_jma = df_jma.iloc[start_row:].copy()
            df_jma['dt'] = pd.to_datetime(df_jma.iloc[:, 0].astype(str).str.strip() + ' ' + df_jma.iloc[:, 1].astype(str).str.strip(), errors='coerce')
            
            # 風速列（通常列2、存在すれば）
            if df_jma.shape[1] > 2:
                df_jma['wind'] = pd.to_numeric(df_jma.iloc[:, 2], errors='coerce')
            # 気温列を探す（「気温」という文字がある列、または通常列4など。ここでは数値に変換できる4列目以降も走査）
            temp_col_idx = None
            for c_idx in range(3, df_jma.shape[1]):
                col_vals = pd.to_numeric(df_jma.iloc[:, c_idx], errors='coerce')
                # 気温らしい数値範囲 (-10〜40度くらい) が多ければ気温列とみなす
                if col_vals.dropna().between(-15, 45).mean() > 0.8:
                    temp_col_idx = c_idx
                    break
            
            if temp_col_idx is not None:
                df_jma['temp'] = pd.to_numeric(df_jma.iloc[:, temp_col_idx], errors='coerce')
            
            df_jma['hour'] = df_jma['dt'].dt.hour
            df_jma['date_str'] = df_jma['dt'].dt.strftime('%Y/%m/%d')
            
            # 18時〜23時のデータに絞る
            mask = df_jma['hour'].isin([18, 19, 20, 21, 22, 23])
            sub = df_jma[mask]
            
            if 'wind' in sub.columns:
                _jma_wind_cache = sub.groupby('date_str')['wind'].mean().to_dict()
            if 'temp' in sub.columns:
                _jma_temp_cache = sub.groupby('date_str')['temp'].mean().to_dict()
                
    except Exception as e:
        print(f"気象庁CSV読込エラー: {e}")

def fetch_jma_val(date_str, cache_dict):
    if not cache_dict:
        load_jma_cache()
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


def prepare(df):
    df = df.copy()
    df.columns = [_normalize_header(c) for c in df.columns]
    for c in REQUIRED:
        if c not in df.columns:
            found = _find_column(df, c)
            if found is not None and found != c:
                df[c] = df[found]
            else:
                df[c] = ""
                
    df["波高"] = _to_number(df["波高"])
    df["風速"] = _to_number(df["風速"])
    df["気温"] = _to_number(df["気温"])
    
    for idx, row in df.iterrows():
        date_val = row["日付"]
        if pd.notna(date_val):
            # 風速の自動補完
            if pd.isna(row["風速"]):
                w_val = fetch_jma_val(date_val, _jma_wind_cache)
                if pd.notna(w_val): df.at[idx, "風速"] = w_val
            # 気温の自動補完
            if pd.isna(row["気温"]):
                t_val = fetch_jma_val(date_val, _jma_temp_cache)
                if pd.notna(t_val): df.at[idx, "気温"] = t_val
                
    df["風速"] = df["風速"].fillna(2.5)
    df["気温"] = df["気温"].fillna(20.0)  # デフォルト気温目安
    df["月"] = _to_number(df["月"])
    df[TARGET] = _to_number(df[TARGET])
    
    for c in ["潮の速さ","天気","潮回り"]:
        df[c] = df[c].map(clean)
    df["潮の速さ"] = df["潮の速さ"].replace("", "不明")
    df["天気"] = df["天気"].replace("", "不明")
    df["月齢帯"] = df["月"].map(moon_band)
    return df

def normalize_columns(df):
    aliases = {
        "竿頭(宝来丸)": ["竿頭(宝来丸)", "竿頭（宝来丸）", "竿頭(宝来丸) ", "竿頭（宝来丸） "],
        "竿頭(飛翔)": ["竿頭(飛翔)", "竿頭（飛翔）", "竿頭(飛龍)", "竿頭（飛龍）", "竿頭(飛翔) ", "竿頭（飛翔） "],
    }
    rename = {}
    cols = {str(c).strip(): c for c in df.columns}
    for canonical, names in aliases.items():
        for n in names:
            if n in cols:
                rename[cols[n]] = canonical
                break
    return df.rename(columns=rename)

def parse_count_series(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.extract(r"([-+]?\d+(?:\.\d+)?)")[0],
        errors="coerce"
    )

def analysis_frame(df, include_other=False):
    cols = [c for c in [TARGET, OTHER_TARGET] if c in df.columns]
    parts = []
    if TARGET in cols:
        x = df.copy(); x["釣果"] = parse_count_series(x[TARGET]); x["船"] = "宝来丸"; parts.append(x)
    if include_other and OTHER_TARGET in cols:
        x = df.copy(); x["釣果"] = parse_count_series(x[OTHER_TARGET]); x["船"] = "飛翔"; parts.append(x)
    if not parts: return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["釣果"])

def load_csv(path):
    last = None
    for enc in ("utf-8-sig","cp932","utf-8"):
        try: return prepare(normalize_columns(pd.read_csv(path, encoding=enc)))
        except UnicodeDecodeError as e: last = e
    raise last

def probability(df, mask):
    x = df.loc[mask, TARGET].dropna()
    if len(x) == 0: return None
    return {
        "n": len(x),
        "p20": (x >= 20).mean()*100,
        "p30": (x >= 30).mean()*100,
        "p40": (x >= 40).mean()*100,
        "mean": x.mean(),
        "median": x.median()
    }

def similarity(row, inp):
    s = 0
    if clean(row["潮回り"]) == inp["tide"]: s += 15
    if pd.notna(row["月"]):
        d = abs(float(row["月"]) - inp["moon"])
        if d <= 1.5: s += 15
        elif d <= 3: s += 8
    if clean(row["潮の速さ"]) == inp["speed"]: s += 15
    if clean(row["天気"]) == inp["weather"]: s += 10
    
    if pd.notna(row["波高"]):
        h_diff = abs(float(row["波高"]) - inp["wave"])
        if h_diff <= 0.3: s += 15
        elif h_diff <= 0.8: s += 8
        
    if pd.notna(row["風速"]):
        w_diff = abs(float(row["風速"]) - inp["wind"])
        if w_diff <= 1.5: s += 15
        elif w_diff <= 3.0: s += 5

    if pd.notna(row["気温"]):
        t_diff = abs(float(row["気温"]) - inp["temp"])
        if t_diff <= 2.0: s += 15
        elif t_diff <= 4.5: s += 5
        
    return s

def predict(df, inp, include_other=False):
    v = analysis_frame(df, include_other)
    if v.empty: return None
    scored = [(idx, similarity(r, inp)) for idx,r in v.iterrows()]
    scored = sorted(scored, key=lambda x:x[1], reverse=True)[:10]
    rows = v.loc[[x[0] for x in scored]]
    w = np.array([max(1,x[1]) for x in scored], dtype=float)
    y = rows["釣果"].to_numpy(float)
    p20 = np.average((y>=20).astype(float), weights=w)*100
    p30 = np.average((y>=30).astype(float), weights=w)*100
    p40 = np.average((y>=40).astype(float), weights=w)*100
    mean = np.average(y, weights=w)
    q25,q75 = np.percentile(y,[25,75]) if len(y)>1 else (y[0],y[0])
    
    exact = (
        (v["潮回り"]==inp["tide"]) &
        (v["潮の速さ"]==inp["speed"]) &
        (v["天気"]==inp["weather"]) &
        (v["月齢帯"]==moon_band(inp["moon"])) &
        (v["波高"].notna()) &
        ((v["波高"] - inp["wave"]).abs() <= 0.5)
    )
    ex = probability(v.assign(**{TARGET:v["釣果"]}), exact)

    avg_score = np.mean([x[1] for x in scored]) if scored else 0.0
    similarity_factor = min(1.0, max(0.0, avg_score / 100.0))
    n_factor = 1.0 - np.exp(-len(rows) / 8.0)
    dispersion = float(np.std(y)) if len(y) > 1 else 0.0
    mean_abs = max(float(np.mean(np.abs(y))), 1.0)
    consistency_factor = 1.0 / (1.0 + dispersion / mean_abs)
    confidence = 100.0 * (0.45 * n_factor + 0.35 * similarity_factor + 0.20 * consistency_factor)
    
    if len(rows) < 5: confidence = min(confidence, 65.0)
    elif len(rows) < 10: confidence = min(confidence, 80.0)
    elif len(rows) < 20: confidence = min(confidence, 90.0)
    else: confidence = min(confidence, 97.0)
        
    return dict(p20=p20,p30=p30,p40=p40,mean=mean,low=q25,high=q75,
                confidence=confidence,rows=rows,scored=scored,exact=ex,include_other=include_other)


def make_ranking(df, min_n=3, include_other=False):
    base = analysis_frame(df, include_other)
    if base.empty: return []

    fields = ["潮回り", "月齢帯", "潮の速さ", "天気"]
    work = base.copy()
    for c in fields: work[c] = work[c].map(clean)
    work = work[~work[fields].isin(["", "不明"]).any(axis=1)].copy()

    out = []
    for keys, g in work.groupby(fields, dropna=False):
        if len(g) < min_n: continue
        y = g["釣果"].astype(float)
        n = len(y)
        raw = (y >= 20).mean() * 100
        corrected = (raw * n + 50.0 * 3) / (n + 3)
        out.append((" × ".join(map(str, keys)), n, raw, corrected, y.mean()))

    out.sort(key=lambda x: (x[3], x[1], x[4]), reverse=True)
    return out[:20]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🦑 三国沖 イカメタル釣果予測（気象庁CSV連携・気温対応）")
        self.geometry("1120x820"); self.minsize(980,700); self.configure(bg="#F5F7FA")
        self.df=pd.DataFrame(); self.csv_path=None
        
        load_jma_cache()
        
        self.setup_style(); self.build(); self.load_default()

    def setup_style(self):
        s=ttk.Style(self)
        try:s.theme_use("clam")
        except:pass
        s.configure("TButton",font=("Yu Gothic UI",11),padding=(12,8))
        s.configure("TCombobox",font=("Yu Gothic UI",11))
        s.configure("TNotebook",background="#F5F7FA",borderwidth=0)
        s.configure("TNotebook.Tab",font=("Yu Gothic UI",11),padding=(16,8))
        s.configure("Treeview",font=("Yu Gothic UI",10),rowheight=30)
        s.configure("Treeview.Heading",font=("Yu Gothic UI",10,"bold"))

    def card(self,p):
        return tk.Frame(p,bg="white",highlightbackground="#E1E5EA",highlightthickness=1)

    def build(self):
        h=tk.Frame(self,bg="#172033",height=82); h.pack(fill="x")
        tk.Label(h,text="🦑 三国沖 イカメタル釣果予測",bg="#172033",fg="white",
                 font=("Yu Gothic UI",22,"bold")).pack(side="left",padx=28,pady=18)
        self.file_label=tk.Label(h,text="CSV: 未読込",bg="#172033",fg="#B9C2D0",font=("Yu Gothic UI",10))
        self.file_label.pack(side="right",padx=28)
        nb=ttk.Notebook(self); nb.pack(fill="both",expand=True,padx=18,pady=18)
        self.pred=tk.Frame(nb,bg="#F5F7FA"); self.rank=tk.Frame(nb,bg="#F5F7FA"); self.data=tk.Frame(nb,bg="#F5F7FA")
        nb.add(self.pred,text="  🔮 予測  "); nb.add(self.rank,text="  📊 条件ランキング  "); nb.add(self.data,text="  📁 データ  ")
        self.build_prediction(); self.build_ranking(); self.build_data()

    def build_prediction(self):
        left=self.card(self.pred); left.pack(side="left",fill="y",padx=(0,10))
        right=tk.Frame(self.pred,bg="#F5F7FA"); right.pack(side="left",fill="both",expand=True)
        tk.Label(left,text="予測条件",bg="white",fg="#172033",font=("Yu Gothic UI",16,"bold")).pack(anchor="w",padx=20,pady=(20,15))
        
        self.vars = {
            "date": tk.StringVar(value=datetime.now().strftime("%Y/%m/%d")),
            "tide": tk.StringVar(value="中潮"),
            "moon": tk.StringVar(value="17.4"),
            "weather": tk.StringVar(value="晴"),
            "speed": tk.StringVar(value="緩い"),
            "wave": tk.StringVar(value="0.5"),
            "wind": tk.StringVar(value="3.0"),
            "temp": tk.StringVar(value="22.0")
        }

        self.include_other=tk.BooleanVar(value=False)
        fields=[
            ("予測日","date",None),
            ("潮回り","tide",TIDE_ORDER),
            ("月齢","moon",None),
            ("天気","weather",WEATHER_OPTIONS),
            ("潮の速さ","speed",SPEED_OPTIONS),
            ("波高(m)","wave",None),
            ("18-23時風速","wind",None),
            ("18-23時気温","temp",None)
        ]
        for label,key,opts in fields:
            f=tk.Frame(left,bg="white"); f.pack(fill="x",padx=20,pady=5)
            tk.Label(f,text=label,width=11,anchor="w",bg="white",fg="#586174",font=("Yu Gothic UI",10)).pack(side="left")
            w=ttk.Combobox(f,textvariable=self.vars[key],values=opts,state="readonly",width=16) if opts else ttk.Entry(f,textvariable=self.vars[key],width=18)
            w.pack(side="right")
            
        cb=tk.Checkbutton(left,text="飛翔（飛龍）の釣果も予測に含める",variable=self.include_other,command=self.refresh_rank,
                          bg="white",fg="#2B3445",activebackground="white",font=("Yu Gothic UI",10),selectcolor="white")
        cb.pack(anchor="w",padx=20,pady=(8,2))
        tk.Label(left,text="☑ 宝来丸＋飛翔をまとめて分析\n☐ 宝来丸だけで分析",bg="white",fg="#7A8494",justify="left",font=("Yu Gothic UI",9)).pack(anchor="w",padx=20,pady=(0,10))
        ttk.Button(left,text="🔍 釣果を予測する",command=self.do_predict).pack(fill="x",padx=20,pady=(6,18))
        
        tk.Label(right,text="予測結果",bg="#F5F7FA",fg="#172033",font=("Yu Gothic UI",18,"bold")).pack(anchor="w",pady=(0,12))
        cards=tk.Frame(right,bg="#F5F7FA"); cards.pack(fill="x")
        self.p20=self.metric(cards,"20杯以上確率","—",0)
        self.mean=self.metric(cards,"予想レンジ","—",1)
        self.conf=self.metric(cards,"信頼度","—",2)
        detail=self.card(right); detail.pack(fill="both",expand=True,pady=(14,0))
        self.text=tk.Text(detail,bg="white",fg="#2B3445",bd=0,wrap="word",font=("Yu Gothic UI",10),padx=18,pady=15)
        self.text.pack(fill="both",expand=True)

    def metric(self,p,title,value,col):
        f=self.card(p); f.grid(row=0,column=col,sticky="nsew",padx=(0 if col==0 else 7,0)); p.grid_columnconfigure(col,weight=1)
        tk.Label(f,text=title,bg="white",fg="#697386",font=("Yu Gothic UI",10)).pack(anchor="w",padx=15,pady=(13,3))
        l=tk.Label(f,text=value,bg="white",fg="#0B6E4F",font=("Yu Gothic UI",25,"bold")); l.pack(anchor="w",padx=15,pady=(0,14)); return l

    def build_ranking(self):
        top=tk.Frame(self.rank,bg="#F5F7FA"); top.pack(fill="x",pady=(0,12))
        tk.Label(top,text="釣れやすい条件ランキング",bg="#F5F7FA",fg="#172033",font=("Yu Gothic UI",18,"bold")).pack(side="left")
        ttk.Button(top,text="🔄 更新",command=self.refresh_rank).pack(side="right")
        f=self.card(self.rank); f.pack(fill="both",expand=True)
        self.rt=ttk.Treeview(f,columns=("cond","n","p","adj","avg"),show="headings")
        for c,t,w in [("cond","条件",470),("n","件数",70),("p","20杯以上(補正)",120),("adj","補正値",90),("avg","平均",100)]:
            self.rt.heading(c,text=t); self.rt.column(c,width=w,anchor="center")
        self.rt.pack(fill="both",expand=True,padx=12,pady=12)
        tk.Label(self.rank, text="※ 4条件（潮回り×月齢帯×潮速×天気）の完全一致を比較。「不明」を除外し3件未満は除外。",
                 bg="#F5F7FA", fg="#697386", justify="left", anchor="w", font=("Yu Gothic UI",9)).pack(fill="x", pady=(8,0))

    def build_data(self):
        top=tk.Frame(self.data,bg="#F5F7FA"); top.pack(fill="x",pady=(0,12))
        ttk.Button(top,text="📂 CSVを開く",command=self.open_csv).pack(side="left")
        ttk.Button(top,text="🔄 再読込",command=self.reload).pack(side="left",padx=8)
        self.info=tk.Label(top,text="",bg="#F5F7FA",fg="#697386",font=("Yu Gothic UI",10)); self.info.pack(side="right")
        f=self.card(self.data); f.pack(fill="both",expand=True)
        self.dt=ttk.Treeview(f,show="headings"); self.dt.pack(fill="both",expand=True,padx=10,pady=10)

    def load_default(self):
        if DEFAULT_CSV.exists(): self.load_path(DEFAULT_CSV)
        else:self.file_label.config(text="CSV: data/chouka.csv を配置してください")

    def open_csv(self):
        p=filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if p:self.load_path(Path(p))

    def load_path(self,p):
        try:
            self.df=load_csv(p); self.csv_path=p; self.file_label.config(text=f"CSV: {p.name}")
            jma_status = f"（気象庁CSV：{'有効' if (_jma_wind_cache or _jma_temp_cache) else '未検出'}）"
            self.info.config(text=f"{len(self.df)}件読み込み {jma_status}"); self.refresh_data(); self.refresh_rank()
        except Exception as e: messagebox.showerror("CSV読込エラー",str(e))

    def reload(self):
        if self.csv_path:self.load_path(self.csv_path)

    def refresh_data(self):
        self.dt.delete(*self.dt.get_children())
        cols=[c for c in ["日付","潮回り",TARGET,"船中合計","人数","竿頭(飛翔)","潮の速さ","月","天気","波高","風速","気温"] if c in self.df.columns]
        self.dt["columns"]=cols
        for c in cols:self.dt.heading(c,text=c); self.dt.column(c,width=90,anchor="center")
        for _,r in self.df.tail(100).iterrows():self.dt.insert("", "end",values=[clean(r.get(c,"")) for c in cols])

    def refresh_rank(self):
        if self.df.empty:return
        self.rt.delete(*self.rt.get_children())
        for cond,n,p,adj,avg in make_ranking(self.df, include_other=self.include_other.get()):
            self.rt.insert("", "end",values=(cond,n,f"{adj:.1f}%",f"{adj:.1f}",f"{avg:.1f}"))

    def do_predict(self):
        if self.df.empty:
            messagebox.showwarning("データなし","先にCSVを読み込んでください。"); return
        try: 
            moon=float(self.vars["moon"].get())
            wave=float(self.vars["wave"].get())
            wind=float(self.vars["wind"].get())
            temp=float(self.vars["temp"].get())
        except ValueError:
            messagebox.showwarning("入力エラー","月齢・波高・風速・気温は数値で入力してください。"); return
            
        inp={"tide":self.vars["tide"].get(),"moon":moon,"weather":self.vars["weather"].get(),
             "speed":self.vars["speed"].get(),"wave":wave,"wind":wind,"temp":temp}
             
        r=predict(self.df,inp,include_other=self.include_other.get())
        self.p20.config(text=f"{r['p20']:.0f}%"); self.mean.config(text=f"{r['low']:.0f}〜{r['high']:.0f}杯")
        stars=max(1,min(5,round(r["confidence"]/20))); self.conf.config(text="★"*stars+"☆"*(5-stars))
        self.text.delete("1.0","end")
        mode="宝来丸＋飛翔" if self.include_other.get() else "宝来丸のみ"
        
        self.text.insert("end",f"分析対象：{mode}\n入力条件\n{inp['tide']} × 月齢{moon} × {inp['weather']} × {inp['speed']} × 波高{wave}m × 風速{wind}m/s × 気温{temp}℃\n\n")
        self.text.insert("end",f"20杯以上 {r['p20']:.1f}% / 30杯以上 {r['p30']:.1f}% / 40杯以上 {r['p40']:.1f}%\n")
        self.text.insert("end",f"信頼度 {r['confidence']:.0f}/100（類似件数・条件一致度・釣果のばらつきから算出）\n")
        self.text.insert("end",f"類似データ {len(r['rows'])}件 / 重み付き平均 {r['mean']:.1f}杯 / 中央値 {np.median(r['rows']['釣果']):.1f}杯\n")
        
        if r["exact"]:
            e=r["exact"]; self.text.insert("end",f"\n★ 近接条件一致: {e['n']}件 / 20杯以上 {e['p20']:.1f}%\n")
            
        self.text.insert("end","\n🔎 似ている過去データ（風速・気温考慮）\n"+"-"*75+"\n")
        for idx,score in r["scored"]:
            x=r["rows"].loc[idx]
            wave_val = f"波{float(x['波高']):.1f}m" if pd.notna(x['波高']) else "波不明"
            wind_val = f"風{float(x['風速']):.1f}m" if pd.notna(x['風速']) else "風不明"
            temp_val = f"気温{float(x['気温']):.1f}℃" if pd.notna(x['気温']) else "気温不明"
            self.text.insert("end",f"{clean(x['日付']):12} {clean(x['船']):5} {clean(x['潮回り']):4} 月{float(x['月']):4.1f} {wave_val:7} {wind_val:8} {temp_val:9} → {x['釣果']:.0f}杯（{score}点）\n")
            
        self.text.insert("end","\n※風速・気温欄が空欄の日付は、data/data.csv（18〜23時）から自動算出・補完しています。")

if __name__=="__main__":
    App().mainloop()