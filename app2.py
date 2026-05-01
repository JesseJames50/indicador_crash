"""
Monitor de Liquidez Sistêmica v2
Melhorias metodológicas sobre app.py:
  1. Z-score rolante (252 dias) — elimina look-ahead bias
  2. KRE: queda acumulada do pico rolante em vez de retorno diário
  3. TED Spread (TEDRATE) como 6º componente
  4. Pesos empíricos diferenciados por tipo de indicador
  5. Filtro de persistência: alerta só após N dias consecutivos
  6. Backtest com marcação de crises históricas e métricas de detecção
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
from fredapi import Fred

from data_loader import merge_data
from config import FRED_API_KEY, START_DATE

# ─── Parâmetros ───────────────────────────────────────────────────────────────
ROLL_WINDOW  = 252   # janela do z-score rolante (dias úteis ≈ 1 ano)
PERSIST_DAYS = 3     # dias consecutivos para confirmar alerta
WARN_THRESH  = 0.75
CRIT_THRESH  = 1.50

# Pesos empíricos: leading indicators recebem maior peso
WEIGHTS = {
    "hy_z":      0.30,  # HY Spread   — melhor leading (Gilchrist & Zakrajsek 2012)
    "ted_z":     0.25,  # TED Spread  — stress interbancário clássico
    "kre_z":     0.20,  # KRE peak    — deterioração bancária gradual
    "vix_z":     0.10,  # VIX         — coincidente
    "t10y_z":    0.10,  # T-Note 10Y  — contexto macro
    "funding_z": 0.05,  # SOFR−FF     — histórico curto (desde 2018)
}

# Eventos pontuais para marcar no gráfico
CRISIS_EVENTS = [
    ("2020-02-20", "COVID início"),
    ("2020-03-16", "COVID fundo"),
    ("2023-03-10", "SVB colapso"),
    ("2018-12-24", "Selloff Q4/2018"),
]

# Janelas de crise para backtest (início, fim, nome)
CRISIS_WINDOWS = [
    ("2018-10-01", "2019-01-31", "Selloff Q4/2018"),
    ("2020-02-15", "2020-05-31", "COVID 2020"),
    ("2023-03-01", "2023-05-31", "SVB 2023"),
]

LOOKBACK_DAYS = 30   # janela de antecipação esperada no backtest

# ─── Funções de indicadores ───────────────────────────────────────────────────
def zscore_rolling(series: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    """Z-score com janela rolante — sem look-ahead bias."""
    roll = series.rolling(window, min_periods=window // 2)
    std  = roll.std().where(lambda s: s > 0)
    return (series - roll.mean()) / std


def kre_stress(kre: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    """Queda % do pico rolante (valor positivo = maior stress)."""
    peak = kre.rolling(window, min_periods=window // 2).max()
    return -(kre - peak) / peak * 100


def persistence_signal(score: pd.Series, threshold: float,
                       days: int = PERSIST_DAYS) -> pd.Series:
    """True quando score esteve >= threshold por `days` dias consecutivos."""
    above = (score >= threshold).astype(int)
    return above.rolling(days, min_periods=days).sum() >= days


def safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns and df[name].notna().any():
        return df[name]
    return pd.Series(dtype=float, index=df.index, name=name)


# ─── Carregamento e cálculo ───────────────────────────────────────────────────
@st.cache_data
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = merge_data()

    # TED Spread (disponível no FRED até ~2023; pós-2023 usa FRA-OIS como proxy)
    try:
        fred = Fred(api_key=FRED_API_KEY)
        ted  = fred.get_series("TEDRATE", observation_start=START_DATE)
        df   = df.join(ted.rename("ted_spread"), how="left")
    except Exception:
        df["ted_spread"] = float("nan")

    # Indicadores v2
    ind = pd.DataFrame(index=df.index)
    ind["t10y_z"]    = zscore_rolling(safe_col(df, "t10y"))
    ind["kre_z"]     = zscore_rolling(kre_stress(safe_col(df, "kre")))
    ind["hy_z"]      = zscore_rolling(safe_col(df, "hy_spread"))
    ind["vix_z"]     = zscore_rolling(safe_col(df, "vix"))
    ind["ted_z"]     = zscore_rolling(safe_col(df, "ted_spread"))

    sofr_ff = safe_col(df, "sofr") - safe_col(df, "fed_funds")
    ind["funding_z"] = zscore_rolling(sofr_ff)

    # Score ponderado com renormalização pelos componentes disponíveis
    score   = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in WEIGHTS.items():
        s = ind[col]
        if s.notna().any():
            score   += w * s.fillna(0)
            total_w += w
    ind["score"] = score / total_w if total_w > 0 else score

    # Alertas com filtro de persistência
    ind["alert_warn"] = persistence_signal(ind["score"], WARN_THRESH)
    ind["alert_crit"] = persistence_signal(ind["score"], CRIT_THRESH)

    # QQQ para gráfico comparativo
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

    return df, ind, qqq


# ─── Layout ───────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Liquidity Monitor v2")
st.title("📊 Monitor de Liquidez Sistêmica v2")
st.caption(
    "Z-score rolante 252d · TED Spread · KRE drop-from-peak · "
    f"Persistência {PERSIST_DAYS}d · Pesos empíricos"
)

df, ind, qqq = load_all()

latest   = float(ind["score"].dropna().iloc[-1])
in_crit  = bool(ind["alert_crit"].iloc[-1])
in_warn  = bool(ind["alert_warn"].iloc[-1])
status   = ("🔴 ALERTA CRÍTICO" if in_crit
            else "🟡 ATENÇÃO" if in_warn
            else "🟢 Normal")

st.metric("Score Sistêmico v2", f"{latest:.3f}", status)

# ─── Gráfico principal ────────────────────────────────────────────────────────
st.subheader("Score histórico com eventos de crise")

fig_main = go.Figure()

# Zonas sombreadas de crise
for c_start, c_end, c_label in CRISIS_WINDOWS:
    fig_main.add_vrect(
        x0=c_start, x1=c_end,
        fillcolor="rgba(239,68,68,0.09)",
        layer="below", line_width=0,
        annotation_text=c_label,
        annotation_position="top left",
        annotation_font=dict(size=9, color="#b91c1c"),
    )

# Score
score_s = ind["score"].dropna()
fig_main.add_scatter(
    x=score_s.index, y=score_s,
    mode="lines", name="Score v2",
    line=dict(color="#3b82f6", width=1.8),
)

# Zonas de alerta coloridas (fill acima do threshold)
fig_main.add_scatter(
    x=score_s.index,
    y=score_s.clip(lower=WARN_THRESH, upper=CRIT_THRESH),
    fill="tozeroy", fillcolor="rgba(245,158,11,0.12)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
)
fig_main.add_scatter(
    x=score_s.index,
    y=score_s.clip(lower=CRIT_THRESH),
    fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
)

# Thresholds
fig_main.add_hline(y=WARN_THRESH, line_dash="dash", line_color="#f59e0b",
                   annotation_text=f"Atenção {WARN_THRESH}",
                   annotation_position="top right",
                   annotation_font=dict(size=10))
fig_main.add_hline(y=CRIT_THRESH, line_dash="dash", line_color="#ef4444",
                   annotation_text=f"Crítico {CRIT_THRESH}",
                   annotation_position="top right",
                   annotation_font=dict(size=10))

# Eventos pontuais
for ev_date, ev_label in CRISIS_EVENTS:
    if pd.Timestamp(ev_date) < score_s.index.min():
        continue
    fig_main.add_vline(
        x=pd.Timestamp(ev_date).timestamp() * 1000,
        line_dash="dot",
        line_color="#dc2626", line_width=1,
        annotation_text=ev_label,
        annotation_position="top",
        annotation_font=dict(size=8, color="#dc2626"),
    )

fig_main.update_layout(
    height=340,
    margin=dict(l=8, r=8, t=44, b=8),
    paper_bgcolor="white", plot_bgcolor="white",
    showlegend=False,
    xaxis=dict(tickangle=-45, tickformat="%b/%y", showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
)
st.plotly_chart(fig_main, width="stretch")

# ─── Score vs QQQ ─────────────────────────────────────────────────────────────
st.subheader("Score de Liquidez vs QQQ — Invesco QQQ Trust (Nasdaq-100)")

fig_comp = make_subplots(specs=[[{"secondary_y": True}]])

# Zonas sombreadas de crise
for c_start, c_end, c_label in CRISIS_WINDOWS:
    fig_comp.add_vrect(
        x0=c_start, x1=c_end,
        fillcolor="rgba(239,68,68,0.09)",
        layer="below", line_width=0,
        annotation_text=c_label,
        annotation_position="top left",
        annotation_font=dict(size=9, color="#b91c1c"),
    )

# Score (eixo esquerdo)
score_s = ind["score"].dropna()
fig_comp.add_trace(
    go.Scatter(
        x=score_s.index, y=score_s,
        mode="lines", name="Score de Liquidez",
        line=dict(color="#3b82f6", width=1.8),
    ),
    secondary_y=False,
)

# QQQ (eixo direito)
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

# Linhas de threshold (eixo esquerdo)
fig_comp.add_hline(
    y=WARN_THRESH, line_dash="dash", line_color="#f59e0b",
    annotation_text=f"Atenção {WARN_THRESH}",
    annotation_position="top right",
    annotation_font=dict(size=9),
)
fig_comp.add_hline(
    y=CRIT_THRESH, line_dash="dash", line_color="#ef4444",
    annotation_text=f"Crítico {CRIT_THRESH}",
    annotation_position="top right",
    annotation_font=dict(size=9),
)

fig_comp.update_layout(
    height=360,
    margin=dict(l=8, r=60, t=44, b=8),
    paper_bgcolor="white", plot_bgcolor="white",
    showlegend=True,
    legend=dict(orientation="h", y=1.06, x=0, xanchor="left", font=dict(size=11)),
    xaxis=dict(tickangle=-45, tickformat="%b/%y", showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="Score"),
    yaxis2=dict(showgrid=False, title="QQQ (USD)", side="right"),
)
st.plotly_chart(fig_comp, width="stretch")

# ─── Componentes do score ─────────────────────────────────────────────────────
st.subheader("Componentes do score (z-score rolante 252d)")

COMPONENTS = [
    ("hy_z",      "HY Spread",          "#a78bfa", "30%"),
    ("ted_z",     "TED Spread",         "#06b6d4", "25%"),
    ("kre_z",     "KRE drop-from-peak", "#10b981", "20%"),
    ("vix_z",     "VIX",                "#f97316", "10%"),
    ("t10y_z",    "T-Note 10Y",         "#3b82f6", "10%"),
    ("funding_z", "SOFR − Fed Funds",   "#8b5cf6",  "5%"),
]

grid = st.columns(3)
for i, (key, label, color, weight) in enumerate(COMPONENTS):
    s = ind[key].dropna()
    fc = go.Figure()
    fc.add_scatter(x=s.index, y=s, mode="lines",
                   line=dict(color=color, width=1.4))
    fc.add_hline(y=0, line_color="#d1d5db", line_width=0.8)
    fc.add_hline(y=CRIT_THRESH, line_dash="dot", line_color="#ef4444", line_width=0.8)
    fc.update_layout(
        height=190,
        margin=dict(l=6, r=6, t=36, b=6),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        title=dict(
            text=f"{label} — <b>{weight}</b>",
            font=dict(size=11, color="#374151"),
            x=0, xanchor="left",
        ),
        xaxis=dict(tickangle=-45, tickformat="%b/%y", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
    )
    with grid[i % 3]:
        st.plotly_chart(fc, width="stretch")

# ─── Backtest ─────────────────────────────────────────────────────────────────
st.subheader(f"Backtest — detecção com antecipação de {LOOKBACK_DAYS} dias")

data_start = ind.index.min()
bt_rows    = []

for c_start, c_end, c_label in CRISIS_WINDOWS:
    ts_start = pd.Timestamp(c_start)
    ts_end   = pd.Timestamp(c_end)
    pre_from = ts_start - pd.Timedelta(days=LOOKBACK_DAYS)

    if ts_end < data_start:
        bt_rows.append({
            "Evento": c_label, "Detectado antes?": "⬜ Sem dados históricos",
            "Antecipação (dias)": "—", "Score máx. pré-crise": "—",
            "Score máx. na crise": "—",
        })
        continue

    pre_score  = ind.loc[pre_from:ts_start, "score"] if pre_from >= data_start else pd.Series(dtype=float)
    cris_score = ind.loc[ts_start:ts_end,   "score"]
    pre_alert  = ind.loc[pre_from:ts_start, "alert_crit"] if pre_from >= data_start else pd.Series(dtype=bool)
    cris_alert = ind.loc[ts_start:ts_end,   "alert_crit"]

    detected_before = bool(pre_alert.any()) if not pre_alert.empty else False
    detected_during = bool(cris_alert.any())

    if detected_before and not pre_alert.empty:
        first_alert = pre_alert[pre_alert].index[0]
        lead_days   = int((ts_start - first_alert).days)
        detection   = "✅ Antecipado"
    elif detected_during:
        first_alert = cris_alert[cris_alert].index[0]
        lead_days   = -int((first_alert - ts_start).days)
        detection   = "⚠️ Durante a crise"
    else:
        lead_days = None
        detection = "❌ Não detectado"

    bt_rows.append({
        "Evento":               c_label,
        "Detectado antes?":     detection,
        "Antecipação (dias)":   str(lead_days) if lead_days is not None else "—",
        "Score máx. pré-crise": f"{pre_score.max():.3f}"  if not pre_score.empty  else "—",
        "Score máx. na crise":  f"{cris_score.max():.3f}" if not cris_score.empty else "—",
    })

# Falsos positivos: dias em alerta crítico fora de qualquer janela de crise (±buffer)
buffer = pd.Timedelta(days=30)
crisis_mask = pd.Series(False, index=ind.index)
for c_start, c_end, _ in CRISIS_WINDOWS:
    s = pd.Timestamp(c_start) - buffer
    e = pd.Timestamp(c_end)   + buffer
    crisis_mask.loc[s:e] = True

available   = ind["alert_crit"].notna()
fp_days     = int((ind["alert_crit"] & ~crisis_mask & available).sum())
total_days  = int(available.sum())

st.dataframe(
    pd.DataFrame(bt_rows),
    width="stretch",
    hide_index=True,
    column_config={
        "Evento":               st.column_config.TextColumn(width="medium"),
        "Detectado antes?":     st.column_config.TextColumn(width="small"),
        "Antecipação (dias)":   st.column_config.TextColumn(width="small"),
        "Score máx. pré-crise": st.column_config.TextColumn(width="small"),
        "Score máx. na crise":  st.column_config.TextColumn(width="small"),
    },
)

if total_days > 0:
    fp_pct = fp_days / total_days * 100
    st.markdown(
        f"**Falsos positivos:** {fp_days} dias em alerta crítico fora das janelas de crise "
        f"({fp_pct:.1f}% do histórico disponível)"
    )

# ─── Composição do score ──────────────────────────────────────────────────────
st.subheader("Composição do score v2")

st.dataframe(
    pd.DataFrame([
        {
            "Indicador":   "HY Spread de Crédito",
            "Peso":        "30%",
            "Tipo":        "Leading",
            "Fonte":       "FRED — BAMLH0A0HYM2",
            "Fundamento":  "Gilchrist & Zakrajsek (2012): melhor preditor individual de recessão",
        },
        {
            "Indicador":   "TED Spread",
            "Peso":        "25%",
            "Tipo":        "Leading",
            "Fonte":       "FRED — TEDRATE",
            "Fundamento":  "Desconfiança interbancária clássica; precede crises de funding",
        },
        {
            "Indicador":   "KRE — queda do pico",
            "Peso":        "20%",
            "Tipo":        "Leading",
            "Fonte":       "Yahoo Finance — KRE",
            "Fundamento":  "Deterioração bancária gradual; mais robusto que retorno diário",
        },
        {
            "Indicador":   "VIX",
            "Peso":        "10%",
            "Tipo":        "Coincidente",
            "Fonte":       "Yahoo Finance — ^VIX",
            "Fundamento":  "Medo do mercado; tende a disparar junto com, não antes do crash",
        },
        {
            "Indicador":   "T-Note 10 anos",
            "Peso":        "10%",
            "Tipo":        "Contexto",
            "Fonte":       "Yahoo Finance — ^TNX",
            "Fundamento":  "Custo do funding soberano; nem sempre leading em crises de liquidez",
        },
        {
            "Indicador":   "SOFR − Fed Funds",
            "Peso":         "5%",
            "Tipo":        "Leading",
            "Fonte":       "FRED — SOFR + FEDFUNDS",
            "Fundamento":  "Stress overnight; histórico curto (desde abr/2018), peso reduzido",
        },
    ]),
    width="stretch",
    hide_index=True,
    column_config={
        "Indicador":  st.column_config.TextColumn(width="medium"),
        "Peso":       st.column_config.TextColumn(width="small"),
        "Tipo":       st.column_config.TextColumn(width="small"),
        "Fonte":      st.column_config.TextColumn(width="medium"),
        "Fundamento": st.column_config.TextColumn(width="large"),
    },
)
