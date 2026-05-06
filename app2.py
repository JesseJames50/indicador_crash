"""
Monitor de Liquidez Sistêmica v6
Melhorias sobre v5:
  E. Deadzoning por componente (z > 0.25) — score zero em regime neutro
  F. Thresholds fixos calibrados no período 2018-2024 — sem deriva temporal
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

# ─── Parâmetros ───────────────────────────────────────────────────────────────
ROLL_WINDOW  = 252
EMA_SPAN     = 21
PERSIST_DAYS = 3
VEL_PCT      = 0.90    # percentil de velocidade
VEL_MIN_ABS  = 0.04    # variação mínima absoluta em 5 dias para alerta velocidade
DEADZONE     = 0.25    # E: z-score mínimo para contribuição por componente
WARN_PCT     = 0.70    # F: percentil atenção no período de calibração
CRIT_PCT     = 0.90    # F: percentil crítico no período de calibração
CALIB_START  = "2018-01-01"  # F: início do período de referência para thresholds
CALIB_END    = "2024-12-31"  # F: fim do período de referência

# Pesos mantidos da v3
WEIGHTS = {
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

LOOKBACK_DAYS = 45

# ─── Funções de indicadores ───────────────────────────────────────────────────
def zscore_rolling(series: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    # 1: min_periods = window completo — sem z-scores instáveis no warmup
    roll = series.rolling(window, min_periods=window)
    std  = roll.std().where(lambda s: s > 0)
    return (series - roll.mean()) / std


def kre_stress(kre: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    peak = kre.rolling(window, min_periods=window).max()
    return -(kre - peak) / peak * 100


def persistence_signal(above: pd.Series, days: int = PERSIST_DAYS) -> pd.Series:
    """True quando `above` foi verdadeiro por `days` dias consecutivos."""
    return above.fillna(False).astype(int).rolling(days, min_periods=days).sum() >= days


def safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns and df[name].notna().any():
        return df[name]
    return pd.Series(dtype=float, index=df.index, name=name)


def detect_qe_periods(walcl: pd.Series,
                      pct_thresh: float = 0.05,
                      window: int = 90) -> list:
    """Detecta períodos de expansão do balanço do Fed > pct_thresh em `window` dias."""
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


# ─── Carregamento e cálculo ───────────────────────────────────────────────────
# TTL 12h: garante dados frescos tanto na abertura (~9h30 ET) quanto no
# fechamento (~16h ET) — os dois janelas de maior estresse intraday.
@st.cache_data(ttl=43200)
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list]:
    df   = merge_data()
    fred = Fred(api_key=FRED_API_KEY)

    # T-Bill 3M
    try:
        df = df.join(
            fred.get_series("DTB3", observation_start=START_DATE).rename("dtb3"),
            how="left")
    except Exception:
        df["dtb3"] = float("nan")

    # Curva 10Y-2Y
    try:
        df = df.join(
            fred.get_series("T10Y2Y", observation_start=START_DATE).rename("t10y2y"),
            how="left")
    except Exception:
        df["t10y2y"] = float("nan")

    # Balanço do Fed (D: regime QE)
    qe_periods = []
    try:
        walcl = fred.get_series("WALCL", observation_start=START_DATE)
        walcl = walcl.resample("D").ffill()
        qe_periods = detect_qe_periods(walcl)
    except Exception:
        pass

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

    # ── E: Score com deadzoning — cada componente só contribui se z > DEADZONE ─
    score, total_w = pd.Series(0.0, index=df.index), 0.0
    for comp_key, w in WEIGHTS.items():
        s = ind[comp_key]
        if s.notna().any():
            score   += w * (s - DEADZONE).clip(lower=0).fillna(0)
            total_w += w
    ind["score_raw"] = score / total_w if total_w > 0 else score

    # EMA-21
    ind["score"] = ind["score_raw"].ewm(span=EMA_SPAN, adjust=False).mean()

    # ── F: Thresholds fixos calibrados no período de referência ──────────────
    calib_s = ind["score"].loc[pd.Timestamp(CALIB_START):pd.Timestamp(CALIB_END)].dropna()
    if len(calib_s) >= ROLL_WINDOW:
        warn_fixed = float(calib_s.quantile(WARN_PCT))
        crit_fixed = float(calib_s.quantile(CRIT_PCT))
    else:
        warn_fixed, crit_fixed = 0.20, 0.45   # fallback conservador

    ind["thresh_warn"] = warn_fixed
    ind["thresh_crit"] = crit_fixed

    ind["alert_warn"] = persistence_signal(ind["score"] >= warn_fixed)
    ind["alert_crit"] = persistence_signal(ind["score"] >= crit_fixed)

    # ── C: Velocidade — P90 rolling 3a + filtro mínimo absoluto ─────────────
    ind["velocity"] = ind["score"].diff(5)
    vel_thresh = (ind["velocity"]
                  .rolling(756, min_periods=504)
                  .quantile(VEL_PCT))
    ind["alert_velocity"] = (
        (ind["velocity"] >= vel_thresh) &
        (ind["velocity"] >= VEL_MIN_ABS) &
        (ind["score"]    >= warn_fixed * 0.7)
    )

    # ── QQQ ───────────────────────────────────────────────────────────────────
    try:
        qqq_df = yf.download("QQQ", start=START_DATE, auto_adjust=True, progress=False)
        if qqq_df is not None and not qqq_df.empty:
            arr = qqq_df.filter(like="Close").to_numpy()
            col = arr[:, 0] if arr.ndim == 2 else arr
            qqq = pd.Series(col, index=qqq_df.index, name="QQQ", dtype=float)
        else:
            qqq = pd.Series(dtype=float, name="QQQ")
    except Exception:
        qqq = pd.Series(dtype=float, name="QQQ")

    return df, ind, qqq, qe_periods


# ─── Layout ───────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Liquidity Monitor v6")
st.title("📊 Monitor de Liquidez Sistêmica v6")
st.caption(
    f"Z-score rolante {ROLL_WINDOW}d · EMA-{EMA_SPAN} · "
    f"Deadzone z>{DEADZONE} · Thresholds P{int(WARN_PCT*100)}/P{int(CRIT_PCT*100)} "
    f"fixos ({CALIB_START[:4]}–{CALIB_END[:4]}) · "
    f"Velocidade P{int(VEL_PCT*100)}+mín {VEL_MIN_ABS} · QE (WALCL)"
)

with st.sidebar:
    if st.button("🔄 Forçar atualização", use_container_width=True,
                 help="Recarrega todos os dados agora (TTL normal: 12h)"):
        load_all.clear()
        st.rerun()

df, ind, qqq, qe_periods = load_all()
st.caption(
    f"Dados carregados em: **{datetime.now().strftime('%d/%m/%Y %H:%M')}** · "
    "atualização automática a cada 12h (abertura ~9h30 ET e fechamento ~16h ET)"
)

score_s      = ind["score"].dropna()
score_raw_s  = ind["score_raw"].dropna()
latest       = float(score_s.iloc[-1])
latest_raw   = float(score_raw_s.iloc[-1])
latest_vel   = float(ind["velocity"].dropna().iloc[-1])
# thresholds fixos (escalares únicos para toda a série)
warn_thresh  = float(ind["thresh_warn"].iloc[0])
crit_thresh  = float(ind["thresh_crit"].iloc[0])
in_crit      = bool(ind["alert_crit"].iloc[-1])
in_warn      = bool(ind["alert_warn"].iloc[-1])
in_vel       = bool(ind["alert_velocity"].iloc[-1])
status       = ("🔴 CRÍTICO"      if in_crit
                else "⚡ ACELERAÇÃO" if in_vel
                else "🟡 ATENÇÃO"   if in_warn
                else "🟢 Normal")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Score EMA-21",       f"{latest:.3f}",       status)
c2.metric("Threshold Atenção",  f"{warn_thresh:.3f}",  f"P{int(WARN_PCT*100)} {CALIB_START[:4]}–{CALIB_END[:4]}")
c3.metric("Threshold Crítico",  f"{crit_thresh:.3f}",  f"P{int(CRIT_PCT*100)} {CALIB_START[:4]}–{CALIB_END[:4]}")
c4.metric("Velocidade (5d)",    f"{latest_vel:+.3f}",  "⚡ Alerta" if in_vel else "—")

# ─── Helpers visuais ──────────────────────────────────────────────────────────
_XAXIS  = dict(tickangle=-45, tickformat="%b/%y", showgrid=False)
_YAXIS  = dict(showgrid=True, gridcolor="#e5e7eb")
_MARGIN = dict(l=8, r=8, t=44, b=8)


def _add_crisis_bg(fig):
    for c_start, c_end, c_label in CRISIS_WINDOWS:
        fig.add_vrect(
            x0=c_start, x1=c_end,
            fillcolor="rgba(239,68,68,0.08)",
            layer="below", line_width=0,
            annotation_text=c_label,
            annotation_position="top left",
            annotation_font=dict(size=9, color="#b91c1c"),
        )


def _add_qe_bg(fig):
    """D: shading cinza nos períodos de QE ativo."""
    for qe_s, qe_e in qe_periods:
        fig.add_vrect(
            x0=qe_s, x1=qe_e,
            fillcolor="rgba(107,114,128,0.07)",
            layer="below", line_width=0,
        )


def _add_events(fig, min_date):
    for ev_date, ev_label in CRISIS_EVENTS:
        if pd.Timestamp(ev_date) < min_date:
            continue
        fig.add_vline(
            x=pd.Timestamp(ev_date).timestamp() * 1000,
            line_dash="dot", line_color="#dc2626", line_width=1,
            annotation_text=ev_label,
            annotation_position="top",
            annotation_font=dict(size=8, color="#dc2626"),
        )


def _add_thresholds(fig):
    """F: thresholds fixos calibrados — linhas horizontais estáveis."""
    fig.add_hline(
        y=warn_thresh, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Atenção P{int(WARN_PCT*100)} ({warn_thresh:.2f})",
        annotation_position="top right",
        annotation_font=dict(size=9),
    )
    fig.add_hline(
        y=crit_thresh, line_dash="dash", line_color="#ef4444",
        annotation_text=f"Crítico P{int(CRIT_PCT*100)} ({crit_thresh:.2f})",
        annotation_position="top right",
        annotation_font=dict(size=9),
    )


# ─── Gráfico principal ────────────────────────────────────────────────────────
st.subheader("Score histórico com eventos de crise")

fig_main = go.Figure()
_add_qe_bg(fig_main)
_add_crisis_bg(fig_main)

# Score bruto (fundo pontilhado)
fig_main.add_scatter(
    x=score_raw_s.index, y=score_raw_s,
    mode="lines", name="Score bruto",
    line=dict(color="#cbd5e1", width=0.9, dash="dot"),
    opacity=0.7,
)

# Score suavizado
fig_main.add_scatter(
    x=score_s.index, y=score_s,
    mode="lines", name=f"Score EMA-{EMA_SPAN}",
    line=dict(color="#3b82f6", width=2.0),
)

# Marcadores de alerta de velocidade (C)
vel_idx = ind[ind["alert_velocity"]].index
vel_vals = score_s.reindex(vel_idx).dropna()
if not vel_vals.empty:
    fig_main.add_scatter(
        x=vel_vals.index, y=vel_vals,
        mode="markers", name="⚡ Aceleração",
        marker=dict(color="#f97316", size=6, symbol="triangle-up"),
    )

# Thresholds fixos calibrados
_add_thresholds(fig_main)
_add_events(fig_main, score_s.index.min())

fig_main.update_layout(
    height=380, margin=_MARGIN,
    paper_bgcolor="white", plot_bgcolor="white",
    showlegend=True,
    legend=dict(orientation="h", y=1.06, x=0, xanchor="left", font=dict(size=10)),
    xaxis=_XAXIS, yaxis=_YAXIS,
)
st.plotly_chart(fig_main, width="stretch")

if qe_periods:
    st.caption(
        f"🔘 Áreas cinzas = {len(qe_periods)} períodos de QE ativo (WALCL +5% em 90d) "
        "— spreads artificialmente comprimidos pelo Fed reduzem confiabilidade dos componentes."
    )

# ─── Velocidade do score ──────────────────────────────────────────────────────
st.subheader(f"C — Velocidade do Score (variação 5 dias) · alerta quando > P{int(VEL_PCT*100)}")

fig_vel = go.Figure()
_add_qe_bg(fig_vel)
_add_crisis_bg(fig_vel)

vel_s    = ind["velocity"].dropna()
vel_pos  = vel_s.where(vel_s > 0, 0)
vel_neg  = vel_s.where(vel_s < 0, 0)
vel_thr  = ind["velocity"].expanding(min_periods=ROLL_WINDOW).quantile(VEL_PCT).dropna()

fig_vel.add_scatter(
    x=vel_pos.index, y=vel_pos,
    fill="tozeroy", fillcolor="rgba(59,130,246,0.20)",
    line=dict(color="#3b82f6", width=0.8), name="Velocidade ↑",
)
fig_vel.add_scatter(
    x=vel_neg.index, y=vel_neg,
    fill="tozeroy", fillcolor="rgba(156,163,175,0.15)",
    line=dict(color="#9ca3af", width=0.8), name="Velocidade ↓",
)
fig_vel.add_scatter(
    x=vel_thr.index, y=vel_thr,
    mode="lines", name=f"P{int(VEL_PCT*100)} (alerta)",
    line=dict(color="#f97316", width=1.2, dash="dot"),
)

# Marcadores de alerta (C)
va_dates = ind[ind["alert_velocity"]].index
va_vals  = vel_s.reindex(va_dates).dropna()
if not va_vals.empty:
    fig_vel.add_scatter(
        x=va_vals.index, y=va_vals,
        mode="markers", name="⚡ Alerta",
        marker=dict(color="#f97316", size=7, symbol="triangle-up"),
    )

fig_vel.add_hline(y=0, line_color="#d1d5db", line_width=0.8)
fig_vel.update_layout(
    height=200, margin=dict(l=8, r=8, t=32, b=8),
    paper_bgcolor="white", plot_bgcolor="white",
    showlegend=True,
    legend=dict(orientation="h", y=1.10, x=0, xanchor="left", font=dict(size=10)),
    xaxis=_XAXIS, yaxis=_YAXIS,
)
st.plotly_chart(fig_vel, width="stretch")

# ─── Score vs QQQ ─────────────────────────────────────────────────────────────
st.subheader("Score de Liquidez vs QQQ — Invesco QQQ Trust (Nasdaq-100)")

fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
_add_qe_bg(fig_comp)
_add_crisis_bg(fig_comp)

fig_comp.add_trace(
    go.Scatter(
        x=score_s.index, y=score_s,
        mode="lines", name=f"Score EMA-{EMA_SPAN}",
        line=dict(color="#3b82f6", width=1.8),
    ),
    secondary_y=False,
)

if not qqq.empty:
    fig_comp.add_trace(
        go.Scatter(
            x=qqq.index, y=qqq,
            mode="lines", name="QQQ (USD)",
            line=dict(color="#10b981", width=1.5),
            opacity=0.85,
        ),
        secondary_y=True,
    )

# Alertas críticos como marcadores no gráfico QQQ
if not qqq.empty:
    crit_dates = ind[ind["alert_crit"]].index
    qqq_crit   = qqq.reindex(crit_dates).dropna()
    if not qqq_crit.empty:
        fig_comp.add_trace(
            go.Scatter(
                x=qqq_crit.index, y=qqq_crit,
                mode="markers", name="🔴 Alerta crítico",
                marker=dict(color="rgba(239,68,68,0.6)", size=5, symbol="circle"),
            ),
            secondary_y=True,
        )

_add_events(fig_comp, score_s.index.min())

fig_comp.add_hline(
    y=warn_thresh, line_dash="dash", line_color="#f59e0b",
    annotation_text=f"Atenção ({warn_thresh:.2f})",
    annotation_position="top right", annotation_font=dict(size=9),
)
fig_comp.add_hline(
    y=crit_thresh, line_dash="dash", line_color="#ef4444",
    annotation_text=f"Crítico ({crit_thresh:.2f})",
    annotation_position="top right", annotation_font=dict(size=9),
)

fig_comp.update_layout(
    height=400,
    margin=dict(l=8, r=60, t=44, b=8),
    paper_bgcolor="white", plot_bgcolor="white",
    showlegend=True,
    legend=dict(orientation="h", y=1.06, x=0, xanchor="left", font=dict(size=10)),
    xaxis=_XAXIS,
    yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="Score"),
    yaxis2=dict(showgrid=False, title="QQQ (USD)", side="right"),
)
st.plotly_chart(fig_comp, width="stretch")

# ─── Componentes do score ─────────────────────────────────────────────────────
st.subheader("Componentes do score (z-score rolante 252d — contribuição positiva apenas)")

COMPONENTS = [
    ("hy_z",      "HY Spread",               "#a78bfa", "rgba(167,139,250,0.18)", "25%"),
    ("tbill_z",   "T-Bill 3M Stress",        "#06b6d4", "rgba(6,182,212,0.18)",   "20%"),
    ("kre_z",     "KRE drop-from-peak",      "#10b981", "rgba(16,185,129,0.18)",  "20%"),
    ("curve_z",   "Curva 10Y-2Y (inv.)",     "#f43f5e", "rgba(244,63,94,0.18)",   "15%"),
    ("vix_z",     "VIX",                     "#f97316", "rgba(249,115,22,0.18)",  "10%"),
    ("t10y_z",    "T-Note 10Y",              "#3b82f6", "rgba(59,130,246,0.18)",   "5%"),
    ("funding_z", "SOFR − Fed Funds",        "#8b5cf6", "rgba(139,92,246,0.18)",   "5%"),
]

grid = st.columns(3)
for i, (key, label, color, fill_color, weight) in enumerate(COMPONENTS):
    s   = ind[key].dropna()
    s_c = s.clip(lower=0)   # mostra versão clipada (o que efetivamente contribui)
    fc  = go.Figure()
    fc.add_scatter(x=s.index, y=s, mode="lines",
                   line=dict(color="#e2e8f0", width=1.0), name="bruto",
                   showlegend=False)
    fc.add_scatter(x=s_c.index, y=s_c,
                   fill="tozeroy", fillcolor=fill_color,
                   line=dict(color=color, width=1.2), name="clipado",
                   showlegend=False)
    fc.add_hline(y=0, line_color="#d1d5db", line_width=0.8)
    fc.update_layout(
        height=190, margin=dict(l=6, r=6, t=36, b=6),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        title=dict(text=f"{label} — <b>{weight}</b>",
                   font=dict(size=11, color="#374151"), x=0, xanchor="left"),
        xaxis=_XAXIS, yaxis=_YAXIS,
    )
    with grid[i % 3]:
        st.plotly_chart(fc, width="stretch")

# ─── Backtest ─────────────────────────────────────────────────────────────────
st.subheader(f"Backtest — threshold fixo P{int(CRIT_PCT*100)}={crit_thresh:.2f} · antecipação até {LOOKBACK_DAYS}d")

data_start = ind.index.min()
bt_rows    = []

for c_start, c_end, c_label in CRISIS_WINDOWS:
    ts_start = pd.Timestamp(c_start)
    ts_end   = pd.Timestamp(c_end)
    pre_from = max(ts_start - pd.Timedelta(days=LOOKBACK_DAYS), data_start)

    if ts_end < data_start:
        bt_rows.append({
            "Evento": c_label, "Detectado?": "⬜ Sem dados",
            "Antecipação (dias)": "—", "Score máx. pré": "—", "Score máx. crise": "—",
            "⚡ Velocidade antes?": "—",
        })
        continue

    pre_score  = ind.loc[pre_from:ts_start,  "score"]
    cris_score = ind.loc[ts_start:ts_end,    "score"]
    pre_crit   = ind.loc[pre_from:ts_start,  "alert_crit"]
    cris_crit  = ind.loc[ts_start:ts_end,    "alert_crit"]
    pre_vel    = ind.loc[pre_from:ts_start,  "alert_velocity"]

    detected_before = bool(pre_crit.any())  if not pre_crit.empty  else False
    detected_during = bool(cris_crit.any()) if not cris_crit.empty else False
    vel_before      = bool(pre_vel.any())   if not pre_vel.empty   else False

    if detected_before:
        first_alert = pre_crit[pre_crit].index[0]
        lead_days   = int((ts_start - first_alert).days)
        detection   = "✅ Antecipado"
    elif detected_during:
        first_alert = cris_crit[cris_crit].index[0]
        lead_days   = -int((first_alert - ts_start).days)
        detection   = "⚠️ Durante"
    else:
        lead_days = None
        detection = "❌ Não detectado"

    bt_rows.append({
        "Evento":              c_label,
        "Detectado?":          detection,
        "Antecipação (dias)":  str(lead_days) if lead_days is not None else "—",
        "Score máx. pré":      f"{pre_score.max():.3f}"  if not pre_score.empty  else "—",
        "Score máx. crise":    f"{cris_score.max():.3f}" if not cris_score.empty else "—",
        "⚡ Velocidade antes?": "✅" if vel_before else "❌",
    })

buffer      = pd.Timedelta(days=30)
crisis_mask = pd.Series(False, index=ind.index)
for c_start, c_end, _ in CRISIS_WINDOWS:
    crisis_mask.loc[pd.Timestamp(c_start) - buffer : pd.Timestamp(c_end) + buffer] = True

available  = ind["alert_crit"].notna()
fp_days    = int((ind["alert_crit"] & ~crisis_mask & available).sum())
total_days = int(available.sum())
vel_fp     = int((ind["alert_velocity"] & ~crisis_mask & available).sum())

st.dataframe(
    pd.DataFrame(bt_rows),
    width="stretch", hide_index=True,
    column_config={
        "Evento":              st.column_config.TextColumn(width="medium"),
        "Detectado?":          st.column_config.TextColumn(width="small"),
        "Antecipação (dias)":  st.column_config.TextColumn(width="small"),
        "Score máx. pré":      st.column_config.TextColumn(width="small"),
        "Score máx. crise":    st.column_config.TextColumn(width="small"),
        "⚡ Velocidade antes?": st.column_config.TextColumn(width="small"),
    },
)

if total_days > 0:
    st.markdown(
        f"**Falsos positivos (score crítico):** {fp_days} dias ({fp_days/total_days*100:.1f}% do histórico)  \n"
        f"**Falsos positivos (velocidade):** {vel_fp} dias ({vel_fp/total_days*100:.1f}% do histórico)"
    )

# ─── Composição do score ──────────────────────────────────────────────────────
st.subheader("Composição do score v5")

st.dataframe(
    pd.DataFrame([
        {"Indicador": "HY Spread de Crédito",     "Peso": "25%", "Tipo": "Leading",
         "Fonte": "FRED — BAMLH0A0HYM2",
         "Fundamento": "G&Z (2012): melhor preditor de recessão; clip elimina sinal falso QE"},
        {"Indicador": "T-Bill 3M Stress (FF−DTB3)", "Peso": "20%", "Tipo": "Leading",
         "Fonte": "FRED — DTB3 + FEDFUNDS",
         "Fundamento": "Substituto do TED pós-LIBOR; voo para T-Bills precede crises de funding"},
        {"Indicador": "KRE — queda do pico",       "Peso": "20%", "Tipo": "Leading",
         "Fonte": "Yahoo Finance — KRE",
         "Fundamento": "Deterioração bancária gradual; captura SVB sem depender de spreads"},
        {"Indicador": "Curva 10Y-2Y (invertida)",  "Peso": "15%", "Tipo": "Leading",
         "Fonte": "FRED — T10Y2Y",
         "Fundamento": "Inversão precede recessões; captura aperto monetário 2022-2023"},
        {"Indicador": "VIX",                        "Peso": "10%", "Tipo": "Coincidente",
         "Fonte": "Yahoo Finance — ^VIX",
         "Fundamento": "Medo do mercado; sinal coincidente, não leading"},
        {"Indicador": "T-Note 10 anos",             "Peso":  "5%", "Tipo": "Contexto",
         "Fonte": "Yahoo Finance — ^TNX",
         "Fundamento": "Nível de juros longos; redundante com curva, peso mínimo"},
        {"Indicador": "SOFR − Fed Funds",           "Peso":  "5%", "Tipo": "Leading",
         "Fonte": "FRED — SOFR + FEDFUNDS",
         "Fundamento": "Stress overnight; histórico curto (abr/2018), peso mínimo"},
    ]),
    width="stretch", hide_index=True,
    column_config={
        "Indicador":  st.column_config.TextColumn(width="medium"),
        "Peso":       st.column_config.TextColumn(width="small"),
        "Tipo":       st.column_config.TextColumn(width="small"),
        "Fonte":      st.column_config.TextColumn(width="medium"),
        "Fundamento": st.column_config.TextColumn(width="large"),
    },
)
