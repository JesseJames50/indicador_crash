"""
Monitor de Liquidez Sistêmica — app_main.py
Painel unificado: v6 (baseline) | v7 (Framework MOVE) | Comparação estatística
Um único load_all() calcula ambos os scores; tabs separam a apresentação.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from plotly.subplots import make_subplots
from fredapi import Fred

from data_loader import merge_data
from config import FRED_API_KEY, START_DATE

# ─── Parâmetros compartilhados ────────────────────────────────────────────────
ROLL_WINDOW   = 252
EMA_SPAN      = 21
PERSIST_DAYS  = 3
VEL_PCT       = 0.90
VEL_MIN_ABS   = 0.04
DEADZONE      = 0.25
DEADZONE_PER_COMP = {          # componentes mais ruidosos — deadzone maior
    "move_z":     0.35,
    "ig_z":       0.35,
    "move_vix_z": 0.30,
}
WARN_PCT      = 0.70
CRIT_PCT      = 0.92
CALIB_START   = "2018-01-01"
CALIB_END     = "2024-12-31"
LOOKBACK_DAYS = 180

# v7 — MOVE framework (DXY removido do score; mantido em Sinais Avançados)
V7_WEIGHTS = {
    "hy_z":       0.21,
    "ig_z":       0.07,
    "move_z":     0.15,
    "move_vix_z": 0.05,
    "tbill_z":    0.15,
    "kre_z":      0.17,
    "curve_z":    0.12,
    "vix_z":      0.05,
    "t10y_z":     0.02,
    "funding_z":  0.01,
}

# v6 — baseline (7 indicadores)
V6_WEIGHTS = {
    "hy_z":      0.25,
    "tbill_z":   0.20,
    "kre_z":     0.20,
    "curve_z":   0.15,
    "vix_z":     0.10,
    "t10y_z":    0.05,
    "funding_z": 0.05,
}

CRISIS_EVENTS = [
    ("2018-12-24", "Q4/2018"),
    ("2020-02-20", "COVID início"),
    ("2020-03-16", "COVID fundo"),
    ("2022-01-03", "Bear 2022"),
    ("2023-03-10", "SVB"),
    ("2025-04-02", "Tarifas"),
]

CRISIS_WINDOWS = [
    ("2018-10-01", "2019-01-31", "Selloff Q4/2018"),
    ("2020-02-15", "2020-05-31", "COVID 2020"),
    ("2022-01-01", "2022-12-31", "Bear Market 2022"),
    ("2023-03-01", "2023-05-31", "SVB 2023"),
    ("2025-02-01", "2025-04-30", "Tarifas 2025"),
]

# ─── Funções de indicadores ───────────────────────────────────────────────────
def zscore_rolling(series: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    roll = series.rolling(window, min_periods=window)
    std  = roll.std().where(lambda s: s > 0)
    return (series - roll.mean()) / std


def kre_stress(kre: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    peak = kre.rolling(window, min_periods=window).max()
    return -(kre - peak) / peak * 100


def persistence_signal(above: pd.Series, days: int = PERSIST_DAYS) -> pd.Series:
    return above.fillna(False).astype(int).rolling(days, min_periods=days).sum() >= days


def safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns and df[name].notna().any():
        return df[name]
    return pd.Series(dtype=float, index=df.index, name=name)


def detect_qe_periods(walcl: pd.Series, pct_thresh: float = 0.05,
                      window: int = 90) -> list:
    if walcl.empty:
        return []
    growth = walcl.pct_change(window).dropna()
    periods, in_qe, start = [], False, None
    for date, val in growth.items():
        if val > pct_thresh and not in_qe:
            start, in_qe = date, True
        elif val <= pct_thresh and in_qe:
            periods.append((start, date))
            in_qe = False
    if in_qe:
        periods.append((start, growth.index[-1]))
    return periods


def _download_close(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    arr = df.filter(like="Close").to_numpy()
    col = arr[:, 0] if arr.ndim == 2 else arr
    return pd.Series(col, index=df.index, name=ticker, dtype=float)


def _build_score(ind: pd.DataFrame, weights: dict) -> pd.Series:
    score, total_w = pd.Series(0.0, index=ind.index), 0.0
    for key, w in weights.items():
        if key not in ind.columns:
            continue
        s = ind[key]
        if s.notna().any():
            dz = DEADZONE_PER_COMP.get(key, DEADZONE)
            score   += w * (s - dz).clip(lower=0).fillna(0)
            total_w += w
    return score / total_w if total_w > 0 else score


def _calibrate_thresholds(score_series: pd.Series):
    calib = score_series.loc[
        pd.Timestamp(CALIB_START):pd.Timestamp(CALIB_END)
    ].dropna()
    if len(calib) >= ROLL_WINDOW:
        return float(calib.quantile(WARN_PCT)), float(calib.quantile(CRIT_PCT))
    return 0.20, 0.45


def _quadrant_stats(move_z: pd.Series, vix_z: pd.Series,
                    qqq: pd.Series, lookahead: int = 21) -> dict:
    common = (move_z.dropna().index
              .intersection(vix_z.dropna().index)
              .intersection(qqq.index))
    if len(common) < ROLL_WINDOW:
        return {}
    mz  = move_z.reindex(common)
    vz  = vix_z.reindex(common)
    fwd = qqq.reindex(common).pct_change(lookahead).shift(-lookahead)
    result = {}
    for q, mask in [
        ("q1", (mz >= 0) & (vz >= 0)),
        ("q2", (mz >= 0) & (vz < 0)),
        ("q3", (mz < 0)  & (vz >= 0)),
        ("q4", (mz < 0)  & (vz < 0)),
    ]:
        r = fwd[mask].dropna()
        result[q] = {
            "freq": float(mask.mean() * 100),
            "mean": float(r.mean() * 100) if len(r) >= 20 else None,
            "std":  float(r.std()  * 100) if len(r) >= 20 else None,
            "n":    len(r),
        }
    return result


def _run_backtest(alert_crit: pd.Series, data_start,
                  alert_vel=None) -> list:
    rows = []
    for c_start, c_end, c_label in CRISIS_WINDOWS:
        ts_start = pd.Timestamp(c_start)
        ts_end   = pd.Timestamp(c_end)
        pre_from = max(ts_start - pd.Timedelta(days=LOOKBACK_DAYS), data_start)

        if ts_end < data_start:
            rows.append({"evento": c_label, "det": "⬜", "lead": None, "vel_lead": None})
            continue

        vel_lead = None
        if alert_vel is not None:
            pre_vel = alert_vel.loc[pre_from:ts_start]
            if not pre_vel.empty and bool(pre_vel.any()):
                first_vel = pre_vel[pre_vel].index[0]
                vel_lead  = int((ts_start - first_vel).days)

        pre_crit  = alert_crit.loc[pre_from:ts_start]
        cris_crit = alert_crit.loc[ts_start:ts_end]

        if not pre_crit.empty and bool(pre_crit.any()):
            first = pre_crit[pre_crit].index[0]
            rows.append({"evento": c_label, "det": "✅",
                         "lead": int((ts_start - first).days), "vel_lead": vel_lead})
        elif not cris_crit.empty and bool(cris_crit.any()):
            first = cris_crit[cris_crit].index[0]
            rows.append({"evento": c_label, "det": "⚠️",
                         "lead": -int((first - ts_start).days), "vel_lead": vel_lead})
        else:
            rows.append({"evento": c_label, "det": "❌",
                         "lead": None, "vel_lead": vel_lead})
    return rows


# ─── Carregamento e cálculo ───────────────────────────────────────────────────
@st.cache_data(ttl=43200)
def load_all():
    df   = merge_data()
    fred = Fred(api_key=FRED_API_KEY)

    for series_id, col_name in [
        ("DTB3",       "dtb3"),
        ("T10Y2Y",     "t10y2y"),
        ("BAMLC0A0CM", "ig_oas"),
    ]:
        try:
            df = df.join(
                fred.get_series(series_id, observation_start=START_DATE).rename(col_name),
                how="left")
        except Exception:
            df[col_name] = float("nan")

    qe_periods = []
    try:
        walcl = fred.get_series("WALCL", observation_start=START_DATE).resample("D").ffill()
        qe_periods = detect_qe_periods(walcl)
    except Exception:
        pass

    dxy_raw = _download_close("DX-Y.NYB")
    if dxy_raw.notna().sum() > ROLL_WINDOW:
        df = df.join(dxy_raw.rename("dxy"), how="left")
    else:
        df["dxy"] = float("nan")

    df = df.ffill()

    move_raw   = _download_close("^MOVE")
    move_label = "MOVE Index (^MOVE)"
    if move_raw.notna().sum() < ROLL_WINDOW:
        move_raw   = safe_col(df, "t10y").diff().rolling(21, min_periods=15).std() * (252 ** 0.5)
        move_label = "Vol realizada T10Y (proxy MOVE)"
        df["move_raw"] = move_raw
    else:
        df = df.join(move_raw.rename("move_raw"), how="left")
    df = df.ffill()

    # ── Componentes ───────────────────────────────────────────────────────────
    ind = pd.DataFrame(index=df.index)
    ind["t10y_z"]    = zscore_rolling(safe_col(df, "t10y"))
    ind["kre_z"]     = zscore_rolling(kre_stress(safe_col(df, "kre")))
    ind["hy_z"]      = zscore_rolling(safe_col(df, "hy_spread"))
    ind["vix_z"]     = zscore_rolling(safe_col(df, "vix"))
    ind["tbill_z"]   = zscore_rolling(safe_col(df, "fed_funds") - safe_col(df, "dtb3"))
    ind["curve_z"]   = zscore_rolling(-safe_col(df, "t10y2y"))
    ind["funding_z"] = zscore_rolling(safe_col(df, "sofr") - safe_col(df, "fed_funds"))
    ind["ig_z"]      = zscore_rolling(safe_col(df, "ig_oas"))
    ind["dxy_z"]     = zscore_rolling(safe_col(df, "dxy").pct_change(21))
    ind["move_z"]    = zscore_rolling(safe_col(df, "move_raw"))
    ind["move_vix_z"] = (ind["move_z"] - ind["vix_z"]).clip(lower=0)

    # ── Scores ────────────────────────────────────────────────────────────────
    ind["v7_score_raw"] = _build_score(ind, V7_WEIGHTS)
    ind["v7_score"]     = ind["v7_score_raw"].ewm(span=EMA_SPAN, adjust=False).mean()
    ind["v6_score_raw"] = _build_score(ind, V6_WEIGHTS)
    ind["v6_score"]     = ind["v6_score_raw"].ewm(span=EMA_SPAN, adjust=False).mean()

    # ── Thresholds ────────────────────────────────────────────────────────────
    v7_warn, v7_crit = _calibrate_thresholds(ind["v7_score"])
    v6_warn, v6_crit = _calibrate_thresholds(ind["v6_score"])

    for prefix, warn, crit in [("v7", v7_warn, v7_crit), ("v6", v6_warn, v6_crit)]:
        sc = ind[f"{prefix}_score"]
        ind[f"{prefix}_thresh_warn"] = warn
        ind[f"{prefix}_thresh_crit"] = crit
        ind[f"{prefix}_alert_warn"]  = persistence_signal(sc >= warn)
        ind[f"{prefix}_alert_crit"]  = persistence_signal(sc >= crit)

    # ── Velocidade ────────────────────────────────────────────────────────────
    for prefix, warn in [("v7", v7_warn), ("v6", v6_warn)]:
        sc = ind[f"{prefix}_score"]
        vel = sc.diff(5)
        ind[f"{prefix}_velocity"] = vel
        vel_thresh = vel.rolling(756, min_periods=504).quantile(VEL_PCT)
        ind[f"{prefix}_alert_velocity"] = (
            (vel >= vel_thresh) &
            (vel >= VEL_MIN_ABS) &
            (sc  >= warn * 0.7)
        )

    # ── QQQ ───────────────────────────────────────────────────────────────────
    try:
        qqq_df = yf.download("QQQ", start=START_DATE, auto_adjust=True, progress=False)
        if qqq_df is not None and not qqq_df.empty:
            arr = qqq_df.filter(like="Close").to_numpy()
            qqq = pd.Series(arr[:, 0] if arr.ndim == 2 else arr,
                            index=qqq_df.index, name="QQQ", dtype=float)
        else:
            qqq = pd.Series(dtype=float, name="QQQ")
    except Exception:
        qqq = pd.Series(dtype=float, name="QQQ")

    return df, ind, qqq, qe_periods, v7_warn, v7_crit, v6_warn, v6_crit, move_label


# ─── Layout global ────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Liquidity Monitor")
st.title("📊 Monitor de Liquidez Sistêmica")
st.caption(
    f"Z-score {ROLL_WINDOW}d · EMA-{EMA_SPAN} · "
    f"Thresholds P{int(WARN_PCT*100)}/P{int(CRIT_PCT*100)} fixos ({CALIB_START[:4]}–{CALIB_END[:4]}) · "
    "v6 (baseline) + v7 (Framework MOVE)"
)

with st.sidebar:
    if st.button("🔄 Forçar atualização", use_container_width=True,
                 help="Recarrega todos os dados (TTL normal: 12h)"):
        load_all.clear()
        st.rerun()

(df, ind, qqq, qe_periods,
 v7_warn_thresh, v7_crit_thresh,
 v6_warn_thresh, v6_crit_thresh,
 move_label) = load_all()

st.caption(
    f"Dados carregados em: **{datetime.now().strftime('%d/%m/%Y %H:%M')}** · "
    f"atualização automática a cada 12h · fonte MOVE: *{move_label}*"
)

# ─── Helpers visuais ──────────────────────────────────────────────────────────
_XAXIS  = dict(tickangle=-45, tickformat="%b/%y", showgrid=False)
_YAXIS  = dict(showgrid=True, gridcolor="#e5e7eb")
_MARGIN = dict(l=8, r=8, t=44, b=8)


def _add_crisis_bg(fig):
    for c_start, c_end, c_label in CRISIS_WINDOWS:
        fig.add_vrect(x0=c_start, x1=c_end,
                      fillcolor="rgba(239,68,68,0.08)",
                      layer="below", line_width=0,
                      annotation_text=c_label,
                      annotation_position="top left",
                      annotation_font=dict(size=9, color="#b91c1c"))


def _add_qe_bg(fig):
    for qe_s, qe_e in qe_periods:
        fig.add_vrect(x0=qe_s, x1=qe_e,
                      fillcolor="rgba(107,114,128,0.07)",
                      layer="below", line_width=0)


def _add_events(fig, min_date):
    for ev_date, ev_label in CRISIS_EVENTS:
        if pd.Timestamp(ev_date) < min_date:
            continue
        fig.add_vline(
            x=pd.Timestamp(ev_date).timestamp() * 1000,
            line_dash="dot", line_color="#dc2626", line_width=1,
            annotation_text=ev_label,
            annotation_position="top",
            annotation_font=dict(size=8, color="#dc2626"))


def _add_thresholds(fig, warn, crit):
    fig.add_hline(y=warn, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Atenção P{int(WARN_PCT*100)} ({warn:.2f})",
                  annotation_position="top right",
                  annotation_font=dict(size=9))
    fig.add_hline(y=crit, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Crítico P{int(CRIT_PCT*100)} ({crit:.2f})",
                  annotation_position="top right",
                  annotation_font=dict(size=9))


def _score_charts(prefix: str, score_col: str, raw_col: str, vel_col: str,
                  alert_vel_col: str, alert_crit_col: str,
                  warn: float, crit: float, version_label: str):
    score_s     = ind[score_col].dropna()
    score_raw_s = ind[raw_col].dropna()
    vel_s       = ind[vel_col].dropna()

    # ── Gráfico principal ──────────────────────────────────────────────────
    st.subheader(f"Score histórico {version_label}")
    fig_main = go.Figure()
    _add_qe_bg(fig_main)
    _add_crisis_bg(fig_main)
    fig_main.add_scatter(x=score_raw_s.index, y=score_raw_s,
                         mode="lines", name="Score bruto",
                         line=dict(color="#cbd5e1", width=0.9, dash="dot"), opacity=0.7)
    fig_main.add_scatter(x=score_s.index, y=score_s,
                         mode="lines", name=f"Score EMA-{EMA_SPAN}",
                         line=dict(color="#3b82f6", width=2.0))
    vel_idx  = ind[ind[alert_vel_col]].index
    vel_vals = score_s.reindex(vel_idx).dropna()
    if not vel_vals.empty:
        fig_main.add_scatter(x=vel_vals.index, y=vel_vals,
                             mode="markers", name="⚡ Aceleração",
                             marker=dict(color="#f97316", size=6, symbol="triangle-up"))
    _add_thresholds(fig_main, warn, crit)
    _add_events(fig_main, score_s.index.min())
    fig_main.update_layout(height=380, margin=_MARGIN,
                           paper_bgcolor="white", plot_bgcolor="white",
                           showlegend=True,
                           legend=dict(orientation="h", y=1.06, x=0,
                                       xanchor="left", font=dict(size=10)),
                           xaxis=_XAXIS, yaxis=_YAXIS)
    st.plotly_chart(fig_main, width="stretch")

    # ── Velocidade ─────────────────────────────────────────────────────────
    st.subheader(f"Velocidade ({version_label}) — variação 5 dias · alerta > P{int(VEL_PCT*100)}")
    fig_vel = go.Figure()
    _add_qe_bg(fig_vel)
    _add_crisis_bg(fig_vel)
    vel_pos = vel_s.where(vel_s > 0, 0)
    vel_neg = vel_s.where(vel_s < 0, 0)
    vel_thr = vel_s.expanding(min_periods=ROLL_WINDOW).quantile(VEL_PCT).dropna()
    fig_vel.add_scatter(x=vel_pos.index, y=vel_pos,
                        fill="tozeroy", fillcolor="rgba(59,130,246,0.20)",
                        line=dict(color="#3b82f6", width=0.8), name="Vel ↑")
    fig_vel.add_scatter(x=vel_neg.index, y=vel_neg,
                        fill="tozeroy", fillcolor="rgba(156,163,175,0.15)",
                        line=dict(color="#9ca3af", width=0.8), name="Vel ↓")
    fig_vel.add_scatter(x=vel_thr.index, y=vel_thr, mode="lines",
                        name=f"P{int(VEL_PCT*100)}",
                        line=dict(color="#f97316", width=1.2, dash="dot"))
    va_vals = vel_s.reindex(ind[ind[alert_vel_col]].index).dropna()
    if not va_vals.empty:
        fig_vel.add_scatter(x=va_vals.index, y=va_vals,
                            mode="markers", name="⚡",
                            marker=dict(color="#f97316", size=7, symbol="triangle-up"))
    fig_vel.add_hline(y=0, line_color="#d1d5db", line_width=0.8)
    fig_vel.update_layout(height=200, margin=dict(l=8, r=8, t=32, b=8),
                          paper_bgcolor="white", plot_bgcolor="white",
                          showlegend=True,
                          legend=dict(orientation="h", y=1.10, x=0,
                                      xanchor="left", font=dict(size=10)),
                          xaxis=_XAXIS, yaxis=_YAXIS)
    st.plotly_chart(fig_vel, width="stretch")

    # ── Score vs QQQ ───────────────────────────────────────────────────────
    st.subheader(f"Score {version_label} vs QQQ (Nasdaq-100)")
    fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
    _add_qe_bg(fig_comp)
    _add_crisis_bg(fig_comp)
    fig_comp.add_trace(
        go.Scatter(x=score_s.index, y=score_s, mode="lines",
                   name=f"Score {version_label}",
                   line=dict(color="#3b82f6", width=1.8)),
        secondary_y=False)
    if not qqq.empty:
        fig_comp.add_trace(
            go.Scatter(x=qqq.index, y=qqq, mode="lines", name="QQQ (USD)",
                       line=dict(color="#10b981", width=1.5), opacity=0.85),
            secondary_y=True)
        crit_dates = ind[ind[alert_crit_col]].index
        qqq_crit   = qqq.reindex(crit_dates).dropna()
        if not qqq_crit.empty:
            fig_comp.add_trace(
                go.Scatter(x=qqq_crit.index, y=qqq_crit,
                           mode="markers", name="🔴 Alerta crítico",
                           marker=dict(color="rgba(239,68,68,0.6)", size=5, symbol="circle")),
                secondary_y=True)
    _add_events(fig_comp, score_s.index.min())
    fig_comp.add_hline(y=warn, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"Atenção ({warn:.2f})",
                       annotation_position="top right", annotation_font=dict(size=9))
    fig_comp.add_hline(y=crit, line_dash="dash", line_color="#ef4444",
                       annotation_text=f"Crítico ({crit:.2f})",
                       annotation_position="top right", annotation_font=dict(size=9))
    fig_comp.update_layout(
        height=400, margin=dict(l=8, r=60, t=44, b=8),
        paper_bgcolor="white", plot_bgcolor="white", showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0, xanchor="left", font=dict(size=10)),
        xaxis=_XAXIS,
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title=f"Score {version_label}"),
        yaxis2=dict(showgrid=False, title="QQQ (USD)", side="right"))
    st.plotly_chart(fig_comp, width="stretch")


def _component_panel(components: list):
    """components: list of (key, label, color, fill_color, weight_label)"""
    st.subheader("Componentes — contribuição positiva acima do deadzone")
    grid = st.columns(3)
    for i, (key, label, color, fill_color, weight) in enumerate(components):
        s   = ind[key].dropna()
        s_c = s.clip(lower=0)
        fc  = go.Figure()
        fc.add_scatter(x=s.index, y=s, mode="lines",
                       line=dict(color="#e2e8f0", width=1.0), showlegend=False)
        fc.add_scatter(x=s_c.index, y=s_c,
                       fill="tozeroy", fillcolor=fill_color,
                       line=dict(color=color, width=1.2), showlegend=False)
        fc.add_hline(y=0, line_color="#d1d5db", line_width=0.8)
        fc.update_layout(
            height=190, margin=dict(l=6, r=6, t=36, b=6),
            paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
            title=dict(text=f"{label} — <b>{weight}</b>",
                       font=dict(size=11, color="#374151"), x=0, xanchor="left"),
            xaxis=_XAXIS, yaxis=_YAXIS)
        with grid[i % 3]:
            st.plotly_chart(fc, width="stretch")


def _backtest_section(bt_rows_raw: list, score_col: str, label: str):
    data_start = ind.index.min()
    bt_rows = []
    for r in bt_rows_raw:
        ts_start   = pd.Timestamp(next(w[0] for w in CRISIS_WINDOWS if w[2] == r["evento"]))
        ts_end     = pd.Timestamp(next(w[1] for w in CRISIS_WINDOWS if w[2] == r["evento"]))
        pre_from   = max(ts_start - pd.Timedelta(days=LOOKBACK_DAYS), data_start)
        pre_score  = ind.loc[pre_from:ts_start, score_col]
        cris_score = ind.loc[ts_start:ts_end,   score_col]
        vel_str  = (f"✅ {r['vel_lead']}d antes" if r["vel_lead"] is not None else "❌")
        lead_str = (f"{r['lead']}d" if r["lead"] is not None else "—")
        bt_rows.append({
            "Evento":            r["evento"],
            "⚡ Vel. pré-crise":  vel_str,
            "P90 confirmação":   f"{r['det']} ({lead_str})",
            "Score máx. pré":    f"{pre_score.max():.3f}"  if not pre_score.empty  else "—",
            "Score máx. crise":  f"{cris_score.max():.3f}" if not cris_score.empty else "—",
        })
    st.subheader(
        f"Backtest {label} · janela {LOOKBACK_DAYS}d · "
        "⚡ velocidade = alerta precoce · P90 = confirmação"
    )
    st.dataframe(pd.DataFrame(bt_rows), width="stretch", hide_index=True,
                 column_config={
                     "Evento":           st.column_config.TextColumn(width="medium"),
                     "⚡ Vel. pré-crise": st.column_config.TextColumn(width="medium"),
                     "P90 confirmação":  st.column_config.TextColumn(width="medium"),
                     "Score máx. pré":   st.column_config.TextColumn(width="small"),
                     "Score máx. crise": st.column_config.TextColumn(width="small"),
                 })


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_v6, tab_v7, tab_comp = st.tabs([
    "📊 v6 — Baseline (7 indicadores)",
    "🔬 v7 — Framework MOVE (10 indicadores)",
    "⚖️ Comparação v6 → v7",
])

data_start = ind.index.min()
bt_v6 = _run_backtest(ind["v6_alert_crit"], data_start, ind["v6_alert_velocity"])
bt_v7 = _run_backtest(ind["v7_alert_crit"], data_start, ind["v7_alert_velocity"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB v6
# ─────────────────────────────────────────────────────────────────────────────
with tab_v6:
    v6_s     = ind["v6_score"].dropna()
    v6_latest = float(v6_s.iloc[-1])
    v6_vel    = float(ind["v6_velocity"].dropna().iloc[-1])
    v6_crit   = bool(ind["v6_alert_crit"].iloc[-1])
    v6_warn   = bool(ind["v6_alert_warn"].iloc[-1])
    v6_vel_a  = bool(ind["v6_alert_velocity"].iloc[-1])
    v6_status = ("🔴 CRÍTICO"      if v6_crit
                 else "⚡ ACELERAÇÃO" if v6_vel_a
                 else "🟡 ATENÇÃO"   if v6_warn
                 else "🟢 Normal")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score v6 EMA-21",      f"{v6_latest:.3f}",           v6_status)
    c2.metric("Threshold Atenção",    f"{v6_warn_thresh:.3f}",
              f"P{int(WARN_PCT*100)} {CALIB_START[:4]}–{CALIB_END[:4]}")
    c3.metric("Threshold Crítico",    f"{v6_crit_thresh:.3f}",
              f"P{int(CRIT_PCT*100)} {CALIB_START[:4]}–{CALIB_END[:4]}")
    c4.metric("Velocidade (5d)",      f"{v6_vel:+.3f}",
              "⚡ Alerta" if v6_vel_a else "—")

    _score_charts("v6", "v6_score", "v6_score_raw", "v6_velocity",
                  "v6_alert_velocity", "v6_alert_crit",
                  v6_warn_thresh, v6_crit_thresh, "v6")

    V6_COMPONENTS = [
        ("hy_z",      "HY Spread de Crédito",  "#a78bfa", "rgba(167,139,250,0.18)", "25%"),
        ("tbill_z",   "T-Bill 3M Stress",       "#06b6d4", "rgba(6,182,212,0.18)",  "20%"),
        ("kre_z",     "KRE drop-from-peak",     "#10b981", "rgba(16,185,129,0.18)", "20%"),
        ("curve_z",   "Curva 10Y-2Y (inv.)",    "#f43f5e", "rgba(244,63,94,0.18)",  "15%"),
        ("vix_z",     "VIX",                    "#f97316", "rgba(249,115,22,0.18)", "10%"),
        ("t10y_z",    "T-Note 10Y",             "#3b82f6", "rgba(59,130,246,0.18)",  "5%"),
        ("funding_z", "SOFR − Fed Funds",       "#8b5cf6", "rgba(139,92,246,0.18)",  "5%"),
    ]
    _component_panel(V6_COMPONENTS)
    _backtest_section(bt_v6, "v6_score", "v6")

# ─────────────────────────────────────────────────────────────────────────────
# TAB v7
# ─────────────────────────────────────────────────────────────────────────────
with tab_v7:
    v7_s      = ind["v7_score"].dropna()
    v7_latest = float(v7_s.iloc[-1])
    v7_vel    = float(ind["v7_velocity"].dropna().iloc[-1])
    v7_crit_a = bool(ind["v7_alert_crit"].iloc[-1])
    v7_warn_a = bool(ind["v7_alert_warn"].iloc[-1])
    v7_vel_a  = bool(ind["v7_alert_velocity"].iloc[-1])
    v7_status = ("🔴 CRÍTICO"      if v7_crit_a
                 else "⚡ ACELERAÇÃO" if v7_vel_a
                 else "🟡 ATENÇÃO"   if v7_warn_a
                 else "🟢 Normal")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score v7 EMA-21",      f"{v7_latest:.3f}",           v7_status)
    c2.metric("Threshold Atenção",    f"{v7_warn_thresh:.3f}",
              f"P{int(WARN_PCT*100)} {CALIB_START[:4]}–{CALIB_END[:4]}")
    c3.metric("Threshold Crítico",    f"{v7_crit_thresh:.3f}",
              f"P{int(CRIT_PCT*100)} {CALIB_START[:4]}–{CALIB_END[:4]}")
    c4.metric("Velocidade (5d)",      f"{v7_vel:+.3f}",
              "⚡ Alerta" if v7_vel_a else "—")

    _score_charts("v7", "v7_score", "v7_score_raw", "v7_velocity",
                  "v7_alert_velocity", "v7_alert_crit",
                  v7_warn_thresh, v7_crit_thresh, "v7")

    # ── Sinais Avançados — Framework MOVE ────────────────────────────────────
    st.subheader("Sinais Avançados — Framework MOVE")
    st.caption(
        "Leitura em três camadas: nível absoluto · vetor (direção) · "
        "divergência com VIX e DXY. Quadrante atual destacado."
    )

    latest_mz  = float(ind["move_z"].dropna().iloc[-1])  if ind["move_z"].notna().any()  else 0.0
    latest_vz  = float(ind["vix_z"].dropna().iloc[-1])   if ind["vix_z"].notna().any()   else 0.0
    latest_dz  = float(ind["dxy_z"].dropna().iloc[-1])   if ind["dxy_z"].notna().any()   else 0.0
    latest_igz = float(ind["ig_z"].dropna().iloc[-1])    if ind["ig_z"].notna().any()    else 0.0
    latest_hyz = float(ind["hy_z"].dropna().iloc[-1])    if ind["hy_z"].notna().any()    else 0.0

    current_q = (
        "Q1" if latest_mz >= 0 and latest_vz >= 0 else
        "Q2" if latest_mz >= 0 and latest_vz < 0  else
        "Q3" if latest_mz <  0 and latest_vz >= 0 else "Q4"
    )
    quad_stats = _quadrant_stats(ind["move_z"], ind["vix_z"], qqq)

    def _quad_card(q_id, emoji, title, subtitle, stats):
        active = (q_id == current_q)
        bdr    = "2px solid #ef4444" if active else "1px solid #e5e7eb"
        bg     = "#fff7f7" if active else "#fafafa"
        tag    = " &nbsp;<b style='color:#ef4444'>← ATUAL</b>" if active else ""
        freq   = f"{stats['freq']:.0f}% dos dias"     if stats else "—"
        ret    = (f"{stats['mean']:+.1f}% ± {stats['std']:.1f}%"
                  if stats and stats["mean"] is not None else "dados insuf.")
        st.markdown(
            f"""<div style="border:{bdr};border-radius:8px;padding:12px 14px;
                background:{bg};margin-bottom:8px">
            <b>{emoji} {title}</b>{tag}<br>
            <span style="color:#6b7280;font-size:0.82em">{subtitle}</span><br>
            <span style="font-size:0.85em">
                Freq histórica: <b>{freq}</b> &nbsp;|&nbsp;
                QQQ próx. 21d: <b>{ret}</b>
            </span></div>""",
            unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        _quad_card("Q2", "⚠️", "MOVE ↑  VIX ↓",
                   "Bonds estressados, equities calmas — sinal de antecipação mais valioso",
                   quad_stats.get("q2"))
        _quad_card("Q4", "🟢", "MOVE ↓  VIX ↓",
                   "Ambos calmos — expansão limpa, melhor regime para ativos de risco",
                   quad_stats.get("q4"))
    with col_r:
        _quad_card("Q1", "🔴", "MOVE ↑  VIX ↑",
                   "Ambos estressados — stress sistêmico confirmado",
                   quad_stats.get("q1"))
        _quad_card("Q3", "🟡", "MOVE ↓  VIX ↑",
                   "Equities nervosas, mercado de bonds estável — stress isolado em equities",
                   quad_stats.get("q3"))

    move_dxy_active = (latest_mz > 0) and (latest_dz > 0)
    dxy_label = (
        f"🚨 **Ativa** (MOVE z={latest_mz:+.2f} · DXY z={latest_dz:+.2f}) "
        "— fuga para caixa em dólar + compressão simultânea de colateral"
        if move_dxy_active else
        f"✅ **Não ativa** (MOVE z={latest_mz:+.2f} · DXY z={latest_dz:+.2f})"
    )
    st.markdown(f"**MOVE + DXY (combinação mais perigosa):** {dxy_label}")

    ig_hy_div = latest_hyz - latest_igz
    if ig_hy_div > 1.0:
        st.markdown(
            f"**HY/IG divergência:** Stress concentrado em crédito especulativo "
            f"(z_HY={latest_hyz:.2f} vs z_IG={latest_igz:.2f})")
    elif latest_igz > latest_hyz + 0.5 and latest_igz > 0:
        st.markdown(
            f"**HY/IG divergência:** IG mais estressado que HY "
            f"(z_IG={latest_igz:.2f} vs z_HY={latest_hyz:.2f})")
    else:
        st.markdown(
            f"**HY/IG:** Sem divergência relevante "
            f"(z_HY={latest_hyz:.2f} · z_IG={latest_igz:.2f})")

    # MOVE histórico
    st.subheader(f"{move_label} (z-score rolante)")
    fig_move = go.Figure()
    _add_qe_bg(fig_move)
    _add_crisis_bg(fig_move)
    mz_s = ind["move_z"].dropna()
    fig_move.add_scatter(x=mz_s.index, y=mz_s, mode="lines",
                         name="MOVE z-score",
                         line=dict(color="#0ea5e9", width=1.5))
    fig_move.add_hline(y=0,   line_color="#d1d5db", line_width=0.8)
    fig_move.add_hline(y=1.0, line_dash="dash", line_color="#0ea5e9",
                       annotation_text="z=1σ", annotation_font=dict(size=8))
    _add_events(fig_move, mz_s.index.min())
    fig_move.update_layout(height=200, margin=dict(l=8, r=8, t=32, b=8),
                           paper_bgcolor="white", plot_bgcolor="white",
                           showlegend=False, xaxis=_XAXIS, yaxis=_YAXIS)
    st.plotly_chart(fig_move, width="stretch")

    V7_COMPONENTS = [
        ("hy_z",       "HY Spread",                  "#a78bfa", "rgba(167,139,250,0.18)", "21%"),
        ("ig_z",       "IG OAS (J)",                 "#c084fc", "rgba(192,132,252,0.18)",  "7%"),
        ("move_z",     f"MOVE / Vol T10Y (G)",        "#0ea5e9", "rgba(14,165,233,0.18)",  "15%"),
        ("move_vix_z", "Divergência MOVE/VIX (H)",   "#6366f1", "rgba(99,102,241,0.18)",   "5%"),
        ("tbill_z",    "T-Bill 3M Stress",            "#06b6d4", "rgba(6,182,212,0.18)",   "15%"),
        ("kre_z",      "KRE drop-from-peak",          "#10b981", "rgba(16,185,129,0.18)",  "17%"),
        ("curve_z",    "Curva 10Y-2Y (inv.)",         "#f43f5e", "rgba(244,63,94,0.18)",   "12%"),
        ("dxy_z",      "DXY pct-21d (I) — obs. only","#f59e0b", "rgba(245,158,11,0.18)",   "—"),
        ("vix_z",      "VIX",                         "#f97316", "rgba(249,115,22,0.18)",   "5%"),
        ("t10y_z",     "T-Note 10Y",                  "#3b82f6", "rgba(59,130,246,0.18)",   "2%"),
        ("funding_z",  "SOFR − Fed Funds",            "#8b5cf6", "rgba(139,92,246,0.18)",   "1%"),
    ]
    _component_panel(V7_COMPONENTS)
    _backtest_section(bt_v7, "v7_score", "v7")

# ─────────────────────────────────────────────────────────────────────────────
# TAB Comparação
# ─────────────────────────────────────────────────────────────────────────────
with tab_comp:
    st.subheader("Comparação Estatística v6 → v7")
    st.caption(
        f"⚡ = alerta de velocidade (precoce) · P90 = confirmação crítica · "
        f"Janela: {LOOKBACK_DAYS}d · mesmo período histórico para ambas as versões."
    )

    comp_rows = []
    for r6, r7 in zip(bt_v6, bt_v7):
        v6_vel = f"✅ {r6['vel_lead']}d" if r6["vel_lead"] is not None else "❌"
        v7_vel = f"✅ {r7['vel_lead']}d" if r7["vel_lead"] is not None else "❌"
        v6_p90 = f"{r6['det']} {r6['lead']}d" if r6["lead"] is not None else r6["det"]
        v7_p90 = f"{r7['det']} {r7['lead']}d" if r7["lead"] is not None else r7["det"]

        l6, l7 = r6["lead"], r7["lead"]
        if l6 is not None and l7 is not None:
            delta_p90 = f"{l7 - l6:+d}d"
        elif l6 is None and l7 is not None and l7 > 0:
            delta_p90 = f"novo ✅ +{l7}d"
        elif l6 is not None and l7 is None:
            delta_p90 = "perdeu ❌"
        else:
            delta_p90 = "—"

        vl6, vl7 = r6["vel_lead"], r7["vel_lead"]
        if vl6 is not None and vl7 is not None:
            delta_vel = f"{vl7 - vl6:+d}d"
        elif vl6 is None and vl7 is not None:
            delta_vel = f"novo ✅ +{vl7}d"
        elif vl6 is not None and vl7 is None:
            delta_vel = "perdeu ❌"
        else:
            delta_vel = "—"

        comp_rows.append({
            "Evento":     r6["evento"],
            "v6 ⚡ vel.": v6_vel,
            "v6 P90":     v6_p90,
            "v7 ⚡ vel.": v7_vel,
            "v7 P90":     v7_p90,
            "Δ ⚡ vel.":  delta_vel,
            "Δ P90":      delta_p90,
        })

    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True,
                 column_config={
                     "Evento":     st.column_config.TextColumn(width="medium"),
                     "v6 ⚡ vel.": st.column_config.TextColumn(width="small"),
                     "v6 P90":     st.column_config.TextColumn(width="small"),
                     "v7 ⚡ vel.": st.column_config.TextColumn(width="small"),
                     "v7 P90":     st.column_config.TextColumn(width="small"),
                     "Δ ⚡ vel.":  st.column_config.TextColumn(width="small"),
                     "Δ P90":      st.column_config.TextColumn(width="small"),
                 })

    # Métricas de resumo
    n_ev        = len(CRISIS_WINDOWS)
    v6_vel_det  = sum(1 for r in bt_v6 if r["vel_lead"] is not None)
    v7_vel_det  = sum(1 for r in bt_v7 if r["vel_lead"] is not None)
    v6_p90_det  = sum(1 for r in bt_v6 if r["det"] == "✅")
    v7_p90_det  = sum(1 for r in bt_v7 if r["det"] == "✅")
    v6_vel_lds  = [r["vel_lead"] for r in bt_v6 if r["vel_lead"] is not None]
    v7_vel_lds  = [r["vel_lead"] for r in bt_v7 if r["vel_lead"] is not None]

    st.subheader("Resumo")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("⚡ Detecção veloc. v6", f"{v6_vel_det}/{n_ev}",
                 f"{v6_vel_det/n_ev*100:.0f}%")
    col_b.metric("⚡ Detecção veloc. v7", f"{v7_vel_det}/{n_ev}",
                 f"{v7_vel_det - v6_vel_det:+d} eventos", delta_color="normal")
    col_c.metric("⚡ Antecip. média v6",
                 f"{sum(v6_vel_lds)/len(v6_vel_lds):.0f}d" if v6_vel_lds else "—")
    col_d.metric("⚡ Antecip. média v7",
                 f"{sum(v7_vel_lds)/len(v7_vel_lds):.0f}d" if v7_vel_lds else "—",
                 (f"{sum(v7_vel_lds)/len(v7_vel_lds) - sum(v6_vel_lds)/len(v6_vel_lds):+.0f}d"
                  if v7_vel_lds and v6_vel_lds else "—"),
                 delta_color="normal")

    # Falsos positivos
    buffer      = pd.Timedelta(days=30)
    crisis_mask = pd.Series(False, index=ind.index)
    for c_start, c_end, _ in CRISIS_WINDOWS:
        crisis_mask.loc[pd.Timestamp(c_start) - buffer : pd.Timestamp(c_end) + buffer] = True
    avail      = ind["v7_alert_crit"].notna()
    total_days = int(avail.sum())

    if total_days > 0:
        v6_fp_crit = int((ind["v6_alert_crit"]     & ~crisis_mask & avail).sum())
        v7_fp_crit = int((ind["v7_alert_crit"]     & ~crisis_mask & avail).sum())
        v6_fp_vel  = int((ind["v6_alert_velocity"] & ~crisis_mask & avail).sum())
        v7_fp_vel  = int((ind["v7_alert_velocity"] & ~crisis_mask & avail).sum())
        fp_delta   = v7_fp_crit - v6_fp_crit

        st.markdown(
            f"**Falsos positivos P90:** v6={v6_fp_crit}d ({v6_fp_crit/total_days*100:.1f}%) · "
            f"v7={v7_fp_crit}d ({v7_fp_crit/total_days*100:.1f}%) · "
            f"Δ = **{fp_delta:+d}d** "
            f"({'✅ menos FP' if fp_delta < 0 else '⚠️ mais FP' if fp_delta > 0 else '= igual'})  \n"
            f"**Falsos positivos ⚡ vel.:** v6={v6_fp_vel}d ({v6_fp_vel/total_days*100:.1f}%) · "
            f"v7={v7_fp_vel}d ({v7_fp_vel/total_days*100:.1f}%)"
        )

    # Tabela de composição comparativa
    st.subheader("Composição dos scores")
    st.dataframe(
        pd.DataFrame([
            {"Indicador": "HY Spread de Crédito",
             "v6": "25%", "v7": "21%", "Tipo": "Leading",
             "Fundamento": "Principal preditor cross-asset de recessão"},
            {"Indicador": "T-Bill 3M Stress",
             "v6": "20%", "v7": "15%", "Tipo": "Leading",
             "Fundamento": "Fuga para T-Bills; substituto TED spread pós-LIBOR"},
            {"Indicador": "KRE — queda do pico",
             "v6": "20%", "v7": "17%", "Tipo": "Leading",
             "Fundamento": "Deterioração bancária gradual; capturou SVB"},
            {"Indicador": "Curva 10Y-2Y (invertida)",
             "v6": "15%", "v7": "12%", "Tipo": "Leading",
             "Fundamento": "Preditor clássico de recessão 6-18 meses"},
            {"Indicador": "VIX",
             "v6": "10%", "v7":  "5%", "Tipo": "Coincidente",
             "Fundamento": "Confirmação; sinal coincidente, não antecipado"},
            {"Indicador": "T-Note 10 anos",
             "v6":  "5%", "v7":  "2%", "Tipo": "Contexto",
             "Fundamento": "Nível de juros longos; redundante com curva"},
            {"Indicador": "SOFR − Fed Funds",
             "v6":  "5%", "v7":  "1%", "Tipo": "Leading",
             "Fundamento": "Stress overnight; histórico curto (abr/2018)"},
            {"Indicador": "MOVE / Vol T10Y (novo G)",
             "v6":   "—", "v7": "15%", "Tipo": "Leading",
             "Fundamento": "Vol implícita bonds > VIX para risco de funding e liquidez"},
            {"Indicador": "IG OAS (novo J)",
             "v6":   "—", "v7":  "7%", "Tipo": "Leading",
             "Fundamento": "IG stress precede HY; divergência IG/HY sinaliza contágio"},
            {"Indicador": "Divergência MOVE/VIX (novo H)",
             "v6":   "—", "v7":  "5%", "Tipo": "Antecipado",
             "Fundamento": "MOVE ↑ + VIX ↓ = bonds stress antes de equities reagirem"},
            {"Indicador": "DXY pct-21d (novo I)",
             "v6":   "—", "v7": "obs.", "Tipo": "Leading",
             "Fundamento": "Removido do score (penaliza crises de confiança no dólar); mantido em Sinais Avançados"},
        ]),
        width="stretch", hide_index=True,
        column_config={
            "Indicador":  st.column_config.TextColumn(width="medium"),
            "v6":         st.column_config.TextColumn(width="small"),
            "v7":         st.column_config.TextColumn(width="small"),
            "Tipo":       st.column_config.TextColumn(width="small"),
            "Fundamento": st.column_config.TextColumn(width="large"),
        },
    )
