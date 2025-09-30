# app.py
# -*- coding: utf-8 -*-
"""
Cross-Asset Market Dashboard (Streamlit + yfinance)
- Aktien, Anleihen, Rohstoffe, FX, Krypto
- Intraday- und Daily-Modus mit adaptiven Intervallen
- Übersicht (Treemap nach Tagesperformance), Gruppentabs, Vergleichscharts, KPI-Tabelle

Install:
  pip install streamlit yfinance plotly pandas numpy python-dateutil
Run:
  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cross‑Asset Market Dashboard", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Zurich")
PLOT_TEMPLATE = "plotly_white"

# Yahoo yield symbols are x10. Adjusters maps to multiplier on Close
ADJUSTERS = {"^TNX": 0.1, "^TYX": 0.1, "^FVX": 0.1}  # 10Y, 30Y, 5Y yields

UNIVERSE = {
    "Aktien": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones": "^DJI",
        "Euro Stoxx 50": "^STOXX50E",
        "DAX": "^GDAXI",
        "SMI": "^SSMI",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225",
        "HSI": "^HSI",
        "MSCI EM (EEM)": "EEM",
        "VIX": "^VIX",
    },
    "Anleihen": {
        "US 10Y Yield": "^TNX",
        "US 30Y Yield": "^TYX",
        "IEF (7–10Y)": "IEF",
        "TLT (20+Y)": "TLT",
        "HYG (HY Corp)": "HYG",
        "LQD (IG Corp)": "LQD",
        "BNDX (Global ex‑US)": "BNDX",
        "EMB (EM Bonds)": "EMB",
    },
    "Rohstoffe": {
        "Gold": "GC=F",
        "Silber": "SI=F",
        "WTI": "CL=F",
        "Brent": "BZ=F",
        "Erdgas": "NG=F",
        "Kupfer": "HG=F",
        "Mais": "ZC=F",
        "Weizen": "ZW=F",
        "Sojabohnen": "ZS=F",
        "Kaffee": "KC=F",
    },
    "FX": {
        "EURUSD": "EURUSD=X",
        "USDCHF": "USDCHF=X",
        "USDJPY": "USDJPY=X",
        "GBPUSD": "GBPUSD=X",
        "USDCNH": "USDCNH=X",
        # DXY-Proxy über ETF, da DX=F/DX-Y.NYB teils unzuverlässig
        "USD Index (UUP)": "UUP",
    },
    "Krypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
    },
}

RANGE_MAP = {
    "1D": ("1d", "1m"),
    "5D": ("5d", "5m"),
    "1M": ("1mo", "30m"),
    "3M": ("3mo", "1h"),
    "6M": ("6mo", "2h"),
    "YTD": ("ytd", "1d"),
    "1Y": ("1y", "1d"),
    "3Y": ("3y", "1d"),
    "5Y": ("5y", "1d"),
    "Max": ("max", "1d"),
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _tz_convert_index(idx):
    # yfinance returns tz-aware for intraday, naive for daily; standardize
    try:
        if getattr(idx, "tz", None) is None:
            return idx.tz_localize("UTC").tz_convert(LOCAL_TZ)
        return idx.tz_convert(LOCAL_TZ)
    except Exception:
        return idx


def adjust_series(ticker: str, s: pd.Series) -> pd.Series:
    if ticker in ADJUSTERS:
        return s * ADJUSTERS[ticker]
    return s


@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(ticker: str, period: str, interval: str) -> pd.Series | None:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        prepost=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return None
    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
        s = df["Close"].copy()
    else:
        # MultiIndex or Series
        try:
            s = df[ticker]["Close"].copy()
        except Exception:
            s = pd.Series(dtype=float, name=ticker)
    s.index = _tz_convert_index(s.index)
    s = adjust_series(ticker, s)
    s.name = ticker
    return s


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_2y(ticker: str) -> pd.Series | None:
    # For robust 1D/MTD/YTD/1Y calculations
    s = fetch_history(ticker, period="2y", interval="1d")
    return s


def pct_change_since(s: pd.Series, start_ts: pd.Timestamp) -> float | None:
    if s is None or s.empty:
        return None
    s2 = s.dropna()
    s2 = s2[s2.index <= s2.index.max()]
    try:
        anchor_idx = s2.index.searchsorted(start_ts)
        if anchor_idx >= len(s2):
            return None
        base = float(s2.iloc[anchor_idx])
        last = float(s2.iloc[-1])
        return (last / base) - 1.0
    except Exception:
        return None


def calc_period_returns(s: pd.Series) -> dict:
    if s is None or s.empty:
        return {k: None for k in ["1D", "5D", "MTD", "YTD", "1Y"]}
    s = s.dropna()
    # 1D and 5D via daily changes
    d1 = None
    if len(s) >= 2:
        d1 = float(s.iloc[-1] / s.iloc[-2] - 1)
    d5 = None
    if len(s) >= 6:
        d5 = float(s.iloc[-1] / s.iloc[-6] - 1)
    today = pd.Timestamp.now(tz=LOCAL_TZ)
    start_mtd = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_ytd = today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    start_1y = today - relativedelta(years=1)
    mtd = pct_change_since(s, start_mtd)
    ytd = pct_change_since(s, start_ytd)
    y1 = pct_change_since(s, start_1y)
    return {"1D": d1, "5D": d5, "MTD": mtd, "YTD": ytd, "1Y": y1}


def summarize_group(group_name: str, tickers: dict) -> pd.DataFrame:
    rows = []
    for name, t in tickers.items():
        s = fetch_daily_2y(t)
        last = float(s.iloc[-1]) if s is not None and not s.empty else np.nan
        rets = calc_period_returns(s)
        rows.append({
            "Name": name,
            "Ticker": t,
            "Last": last,
            "1D": rets["1D"],
            "5D": rets["5D"],
            "MTD": rets["MTD"],
            "YTD": rets["YTD"],
            "1Y": rets["1Y"],
        })
    df = pd.DataFrame(rows)
    return df


def percent_format(x):
    if pd.isna(x):
        return ""
    return f"{x*100:.2f}%"


def number_format(x):
    if pd.isna(x):
        return ""
    # Yields should show % directly
    return f"{x:,.2f}"


def style_return_table(df: pd.DataFrame):
    fmt = {"Last": number_format, "1D": percent_format, "5D": percent_format,
           "MTD": percent_format, "YTD": percent_format, "1Y": percent_format}
    styled = (df.style
              .format(fmt)
              .hide(axis="index")
              .background_gradient(subset=["1D", "5D", "MTD", "YTD", "1Y"], cmap="RdYlGn", vmin=-0.05, vmax=0.05)
              )
    return styled


def normalized_index(df: pd.DataFrame) -> pd.DataFrame:
    # Rebase to 100 at first valid point per column
    df = df.copy()
    for c in df.columns:
        series = df[c].dropna()
        if series.empty:
            continue
        df.loc[series.index, c] = 100 * series / series.iloc[0]
    return df


def plot_multi_time_series(data: pd.DataFrame, title: str, yaxis_title: str = "Index (rebased = 100)", log_scale=False):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode="lines", name=col))
    fig.update_layout(template=PLOT_TEMPLATE, title=title, legend=dict(orientation="h", y= -0.2), height=520)
    fig.update_yaxes(title=yaxis_title, type="log" if log_scale else "linear")
    return fig


def plot_sparkline(s: pd.Series, title: str):
    if s is None or s.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOT_TEMPLATE, height=120, margin=dict(l=10,r=10,t=20,b=10))
        return fig
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", name=title))
    fig.update_layout(template=PLOT_TEMPLATE, height=120, margin=dict(l=10,r=10,t=20,b=10), title=title)
    return fig


def treemap_overview(grouped_summaries: dict) -> go.Figure:
    # Build treemap by 1D return size/color
    names, parents, values, labels, colors = [], [], [], [], []
    for g, df in grouped_summaries.items():
        names.append(g)
        parents.append("")
        values.append(max(1, len(df)))
        labels.append(g)
        colors.append(0.0)  # neutral at group level
        for _, row in df.iterrows():
            names.append(row["Name"])
            parents.append(g)
            values.append(1)
            labels.append(f"{row['Name']}\n{row['Ticker']}\n1D: {percent_format(row['1D'])}")
            colors.append(0.0 if pd.isna(row["1D"]) else row["1D"])  # color by 1D
    fig = px.treemap(
        names=names,
        parents=parents,
        values=values,
        color=colors,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )
    fig.update_layout(template=PLOT_TEMPLATE, title="Übersicht: Tagesperformance (Treemap)")
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Steuerung")
    sel_range = st.select_slider("Zeithorizont", options=list(RANGE_MAP.keys()), value="6M")
    period, interval = RANGE_MAP[sel_range]
    ttl_sec = st.number_input("Refresh-Cache TTL (Sek.)", min_value=15, max_value=600, value=60, step=15, help="Fetch-Cache Ablauf. Niedriger = aktueller, höher = weniger API-Calls.")
    if ttl_sec != 60:
        # reset caches with new TTL by keying function; simple workaround: clear all caches
        st.cache_data.clear()
    custom_tickers = st.text_input("Custom Ticker (Yahoo, kommasepariert)", value="", help="Beispiele: AAPL,MSFT,NVDA,ASML.AS")
    log_scale = st.checkbox("Log-Skala im Vergleichs-Chart", value=False)
    if st.button("Manuell aktualisieren"):
        st.cache_data.clear()
        st.experimental_rerun()

# ─────────────────────────────────────────────────────────────
# Data Fetch for selected range
# ─────────────────────────────────────────────────────────────

def fetch_group_timeframes(tickers: dict, period: str, interval: str) -> dict:
    out = {}
    with ThreadPoolExecutor(max_workers=min(8, len(tickers))) as ex:
        futs = {ex.submit(fetch_history, t, period, interval): (name, t) for name, t in tickers.items()}
        for fut in as_completed(futs):
            name, t = futs[fut]
            try:
                s = fut.result()
            except Exception:
                s = None
            out[name] = (t, s)
    return out

# Build summaries and sparkline data per group
summaries = {}
sparklines = {}
for grp, tkmap in UNIVERSE.items():
    summaries[grp] = summarize_group(grp, tkmap)
    fetched = fetch_group_timeframes(tkmap, period, interval)
    sparklines[grp] = fetched

# Custom tickers for comparison
custom_list = []
if custom_tickers.strip():
    for tk in [t.strip() for t in custom_tickers.split(',') if t.strip()]:
        custom_list.append(tk)

# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────
st.title("Cross‑Asset Market Dashboard")
st.caption("Quelle: Yahoo Finance. Intraday nahe Echtzeit, abhängig von Datenanbieter-Latenz.")

# Overview Treemap
try:
    fig_tm = treemap_overview(summaries)
    st.plotly_chart(fig_tm, use_container_width=True)
except Exception:
    st.info("Treemap aktuell nicht verfügbar.")

# Vergleichs‑Chart über Gruppen + Custom
all_choices = []
name_to_ticker = {}
for grp, mp in UNIVERSE.items():
    for n, t in mp.items():
        all_choices.append(f"{grp}: {n}")
        name_to_ticker[f"{grp}: {n}"] = t
for ct in custom_list:
    all_choices.append(f"Custom: {ct}")
    name_to_ticker[f"Custom: {ct}"] = ct

sel_multi = st.multiselect("Vergleichsauswahl (rebased=100)", options=all_choices, default=["Aktien: S&P 500", "Aktien: Nasdaq 100", "Anleihen: TLT (20+Y)", "Rohstoffe: Gold", "FX: EURUSD"], max_selections=12)

if sel_multi:
    # Fetch visible selection with current period/interval
    series_map = {}
    for label in sel_multi:
        tk = name_to_ticker[label]
        s = fetch_history(tk, period=period, interval=interval)
        if s is not None and not s.empty:
            series_map[label] = s.rename(label)
    if series_map:
        df = pd.concat(series_map.values(), axis=1).dropna(how="all")
        df_norm = normalized_index(df)
        st.plotly_chart(plot_multi_time_series(df_norm, title=f"Vergleich {sel_range}" , log_scale=log_scale), use_container_width=True)

# Tabs per group
_tabs = st.tabs(list(UNIVERSE.keys()))

for tab, (grp, tkmap) in zip(_tabs, UNIVERSE.items()):
    with tab:
        st.subheader(grp)
        df = summaries[grp]
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.markdown("**KPI‑Tabelle**")
            st.dataframe(style_return_table(df), use_container_width=True)
        with c2:
            st.markdown("**Sparklines**")
            # 3 Spalten Raster
            cols = st.columns(3)
            i = 0
            for name, (tk, s) in sparklines[grp].items():
                with cols[i % 3]:
                    st.plotly_chart(plot_sparkline(s, f"{name} ({tk})"), use_container_width=True)
                i += 1
        # Vergleich innerhalb der Gruppe
        st.markdown("**Vergleich innerhalb der Gruppe (rebased=100)**")
        choices = [f"{name} ({tk})" for name, tk in tkmap.items()]
        sel = st.multiselect(f"Auswahl {grp}", options=choices, default=choices[:4])
        series_map = {}
        for item in sel:
            name = item.split(" (")[0]
            tk = tkmap[name]
            s = fetch_history(tk, period=period, interval=interval)
            if s is not None and not s.empty:
                series_map[item] = s.rename(item)
        if series_map:
            df = pd.concat(series_map.values(), axis=1).dropna(how="all")
            st.plotly_chart(plot_multi_time_series(normalized_index(df), title=f"{grp} – {sel_range}"), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.caption(
    "Hinweis: Yields (^TNX,^TYX,^FVX) als Prozentwerte normalisiert. Daten und Intraday‑Latenzen abhängig von Yahoo. "
    "Cache‑TTL in der Sidebar steuerbar. Button ‘Manuell aktualisieren’ leert Cache und lädt neu."
)
