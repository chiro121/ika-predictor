import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "chouka.csv"
TARGET = "竿頭(宝来丸)"
OTHER_TARGET = "竿頭(飛翔)"
REQUIRED = ["日付","潮回り",TARGET,OTHER_TARGET,"二枚潮","潮の速さ","月","天気"]

TIDE_ORDER = ["大潮","中潮","小潮","長潮","若潮"]
WEATHER_OPTIONS = ["晴","曇","雨","晴曇","曇晴","雨曇","雨晴","晴雨","不明"]
SPEED_OPTIONS = ["緩い","普通","速い","カッ飛び","不明"]
TWO_TIDE_OPTIONS = ["無","有","不明"]

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
    # Excel/CSV由来の空白・全角括弧などの揺れを吸収
    return (str(x).replace("\ufeff", "").strip()
            .replace("（", "(").replace("）", ")")
            .replace("　", ""))

def _find_column(df, wanted):
    wanted_n = _normalize_header(wanted)
    for c in df.columns:
        if _normalize_header(c) == wanted_n:
            return c
    # 「竿頭」「宝来丸」を含む列も救済
    if wanted == TARGET:
        for c in df.columns:
            n = _normalize_header(c)
            if "竿頭" in n and "宝来丸" in n:
                return c
    return None

def _to_number(series):
    # 「20杯」「20.0杯」「20」などを数値化
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
                      .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce"
    )

def prepare(df):
    df = df.copy()
    # 列名の表記揺れを正規化
    df.columns = [_normalize_header(c) for c in df.columns]
    for c in REQUIRED:
        if c not in df.columns:
            found = _find_column(df, c)
            if found is not None and found != c:
                df[c] = df[found]
            else:
                df[c] = ""
    df["月"] = _to_number(df["月"])
    df[TARGET] = _to_number(df[TARGET])
    for c in ["二枚潮","潮の速さ","天気","潮回り"]:
        df[c] = df[c].map(clean)
    df["二枚潮"] = df["二枚潮"].replace("", "不明")
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
    # 20 / 20.0 / 20杯 / "20 杯" などを数値化
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
    if clean(row["潮回り"]) == inp["tide"]: s += 30
    if pd.notna(row["月"]):
        d = abs(float(row["月"]) - inp["moon"])
        if d <= 1.5: s += 25
        elif d <= 3: s += 15
    if clean(row["二枚潮"]) == inp["two"]: s += 5
    if clean(row["潮の速さ"]) == inp["speed"]: s += 15
    if clean(row["天気"]) == inp["weather"]: s += 10
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
        (v["二枚潮"]==inp["two"]) &
        (v["潮の速さ"]==inp["speed"]) &
        (v["天気"]==inp["weather"]) &
        (v["月齢帯"]==moon_band(inp["moon"]))
    )
    ex = probability(v.assign(**{TARGET:v["釣果"]}), exact)
    # 信頼度は「予測が当たる確率」ではなく、今回の予測をどれだけ
    # 過去データが支えているかを表す指標。旧式は上位10件を使うだけで
    # ほぼ必ず100に飽和していたため、件数・条件一致度・結果のばらつきを
    # 分けて評価する。十分なデータがない限り100%にはならない。
    avg_score = np.mean([x[1] for x in scored]) if scored else 0.0
    max_score = 85.0  # 潮30 + 月25 + 二枚潮5 + 潮速15 + 天気10
    similarity_factor = min(1.0, max(0.0, avg_score / max_score))
    n_factor = 1.0 - np.exp(-len(rows) / 8.0)
    dispersion = float(np.std(y)) if len(y) > 1 else 0.0
    mean_abs = max(float(np.mean(np.abs(y))), 1.0)
    consistency_factor = 1.0 / (1.0 + dispersion / mean_abs)
    confidence = 100.0 * (0.45 * n_factor + 0.35 * similarity_factor + 0.20 * consistency_factor)
    # 「データが少ないのに100%」を防ぐ安全上限
    if len(rows) < 5:
        confidence = min(confidence, 65.0)
    elif len(rows) < 10:
        confidence = min(confidence, 80.0)
    elif len(rows) < 20:
        confidence = min(confidence, 90.0)
    else:
        confidence = min(confidence, 97.0)
    return dict(p20=p20,p30=p30,p40=p40,mean=mean,low=q25,high=q75,
                confidence=confidence,rows=rows,scored=scored,exact=ex,include_other=include_other)


def make_ranking(df, min_n=3, include_other=False):
    """重複を避けた詳細条件ランキング。

    同じ観測データを「4条件」「3条件」の両方に出すことはしない。
    メインランキングは、予測に使う4条件
      潮回り × 月齢帯 × 潮の速さ × 天気
    の完全一致グループだけを比較する。
    「不明」は条件ランキングから除外する。
    """
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
        # 少数データの100%をそのままランキングに使わないための
        # 弱めのベイズ平滑化。事前確率50%、仮想3件。
        corrected = (raw * n + 50.0 * 3) / (n + 3)
        out.append((" × ".join(map(str, keys)), n, raw, corrected, y.mean()))

    # 「補正後確率」→「件数」→「平均釣果」の順で評価。
    out.sort(key=lambda x: (x[3], x[1], x[4]), reverse=True)
    return out[:20]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🦑 三国沖 イカメタル釣果予測")
        self.geometry("1120x760"); self.minsize(980,680); self.configure(bg="#F5F7FA")
        self.df=pd.DataFrame(); self.csv_path=None
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
        self.vars={k:tk.StringVar(v="") for k,v in {}.items()}
        self.vars={"date":tk.StringVar(value=datetime.now().strftime("%Y/%m/%d")),
                   "tide":tk.StringVar(value="中潮"),"moon":tk.StringVar(value="17.4"),
                   "weather":tk.StringVar(value="晴"),"speed":tk.StringVar(value="緩い"),"two":tk.StringVar(value="無") }
        self.include_other=tk.BooleanVar(value=False)
        fields=[("予測日","date",None),("潮回り","tide",TIDE_ORDER),("月齢","moon",None),
                ("天気","weather",WEATHER_OPTIONS),("潮の速さ","speed",SPEED_OPTIONS),("二枚潮","two",TWO_TIDE_OPTIONS)]
        for label,key,opts in fields:
            f=tk.Frame(left,bg="white"); f.pack(fill="x",padx=20,pady=7)
            tk.Label(f,text=label,width=10,anchor="w",bg="white",fg="#586174",font=("Yu Gothic UI",10)).pack(side="left")
            w=ttk.Combobox(f,textvariable=self.vars[key],values=opts,state="readonly",width=17) if opts else ttk.Entry(f,textvariable=self.vars[key],width=20)
            w.pack(side="right")
        cb=tk.Checkbutton(left,text="飛翔（飛龍）の釣果も予測に含める",variable=self.include_other,command=self.refresh_rank,
                          bg="white",fg="#2B3445",activebackground="white",font=("Yu Gothic UI",10),selectcolor="white")
        cb.pack(anchor="w",padx=20,pady=(10,4))
        tk.Label(left,text="☑ ON：宝来丸＋飛翔をまとめて分析\n☐ OFF：宝来丸だけで分析",bg="white",fg="#7A8494",justify="left",font=("Yu Gothic UI",9)).pack(anchor="w",padx=20,pady=(0,10))
        ttk.Button(left,text="🔍 釣果を予測する",command=self.do_predict).pack(fill="x",padx=20,pady=(8,22))
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
        tk.Label(self.rank, text="※ 同じデータを複数条件に重複表示しないため、4条件（潮回り×月齢帯×潮速×天気）の完全一致だけを比較。\n　「不明」を除外し、3件未満の条件はランキング対象外。20杯以上率は少数データを補正しています。",
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
            self.info.config(text=f"{len(self.df)}件読み込み"); self.refresh_data(); self.refresh_rank()
        except Exception as e: messagebox.showerror("CSV読込エラー",str(e))

    def reload(self):
        if self.csv_path:self.load_path(self.csv_path)

    def refresh_data(self):
        self.dt.delete(*self.dt.get_children())
        cols=[c for c in ["日付","潮回り",TARGET,"船中合計","人数","竿頭(飛翔)","二枚潮","潮の速さ","月","天気"] if c in self.df.columns]
        self.dt["columns"]=cols
        for c in cols:self.dt.heading(c,text=c); self.dt.column(c,width=110,anchor="center")
        for _,r in self.df.tail(100).iterrows():self.dt.insert("", "end",values=[clean(r.get(c,"")) for c in cols])

    def refresh_rank(self):
        if self.df.empty:return
        self.rt.delete(*self.rt.get_children())
        for cond,n,p,adj,avg in make_ranking(self.df, include_other=self.include_other.get()):
            self.rt.insert("", "end",values=(cond,n,f"{adj:.1f}%",f"{adj:.1f}",f"{avg:.1f}"))

    def do_predict(self):
        if self.df.empty:
            messagebox.showwarning("データなし","先にCSVを読み込んでください。"); return
        try: moon=float(self.vars["moon"].get())
        except ValueError:
            messagebox.showwarning("入力エラー","月齢は数値で入力してください。"); return
        inp={"tide":self.vars["tide"].get(),"moon":moon,"weather":self.vars["weather"].get(),
             "speed":self.vars["speed"].get(),"two":self.vars["two"].get()}
        r=predict(self.df,inp,include_other=self.include_other.get())
        self.p20.config(text=f"{r['p20']:.0f}%"); self.mean.config(text=f"{r['low']:.0f}〜{r['high']:.0f}杯")
        stars=max(1,min(5,round(r["confidence"]/20))); self.conf.config(text="★"*stars+"☆"*(5-stars))
        self.text.delete("1.0","end")
        mode="宝来丸＋飛翔" if self.include_other.get() else "宝来丸のみ"
        self.text.insert("end",f"分析対象：{mode}\n入力条件\n{inp['tide']} × 月齢{moon} × {inp['weather']} × {inp['speed']} × 二枚潮{inp['two']}\n\n")
        self.text.insert("end",f"20杯以上 {r['p20']:.1f}% / 30杯以上 {r['p30']:.1f}% / 40杯以上 {r['p40']:.1f}%\n")
        self.text.insert("end",f"信頼度 {r['confidence']:.0f}/100（類似件数・条件一致度・釣果のばらつきから算出）\n")
        self.text.insert("end",f"類似データ {len(r['rows'])}件 / 重み付き平均 {r['mean']:.1f}杯 / 中央値 {np.median(r['rows']['釣果']):.1f}杯\n")
        if r["exact"]:
            e=r["exact"]; self.text.insert("end",f"\n★ 5条件一致（同じ月齢帯）: {e['n']}件 / 20杯以上 {e['p20']:.1f}%\n")
        self.text.insert("end","\n🔎 似ている過去データ\n"+"-"*72+"\n")
        for idx,score in r["scored"]:
            x=r["rows"].loc[idx]
            self.text.insert("end",f"{clean(x['日付']):12} {clean(x['船']):5} {clean(x['潮回り']):4} 月{float(x['月']):4.1f} {clean(x['天気']):4} {clean(x['潮の速さ']):4} 二枚潮{clean(x['二枚潮']):2} → {x['釣果']:.0f}杯（{score}点）\n")
        self.text.insert("end","\n※二枚潮の一致は予測スコア5点。現時点では二枚潮データが少ないため、結果を大きく左右しない設計です。\n※条件ランキングは「4条件完全一致」のみ。少数条件の100%をそのまま信用せず、補正後確率で順位付けしています。")
        self.text.insert("end","\n※信頼度は「当たる確率」ではなく、今回の予測を過去データがどれだけ支えているかの目安です。")

if __name__=="__main__":
    App().mainloop()
