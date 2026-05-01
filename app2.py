"""
Monitor de Liquidez Sistêmica v3
Melhorias sobre v2:
  1. Curva de juros 10Y-2Y (T10Y2Y) — inversão precede recessões
  2. T-Bill 3M stress (FEDFUNDS - DTB3) substitui TED descontinuado
  3. Pesos empíricos rebalanceados (7 componentes)
  4. Suavização EMA-21 sobre o score bruto — reduz falsos positivos
  5. Janelas de crise expandidas: Bear 2022 e Tarifas 2025
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
ROLL_WINDOW  = 252   # janela z-score rolante (≈ 1 ano)
EMA_SPAN     = 21    # suavização EMA do score final
PERSIST_DAYS = 3     # dias consecutivos para confirmar alerta
WARN_THRESH  = 0.60
CRIT_THRESH  = 1.20

# Pesos rebalanceados — 7 componentes com cobertura histórica contínua
WEIGHTS = {
    "hy_z":      0.25,  # HY Spread        — leading, G&Z 2012
    "tbill_z":   0.20,  # T-Bill 3M stress — substituto TED pós-LIBOR
    "kre_z":     0.20,  # KRE drop-from-peak — deterioração bancária
    "curve_z":   0.15,  # Curva 10Y-2Y inv.  — preditor de recessão
    "vix_z":     0.10,  # VIX               — coincidente
    "t10y_z":    0.05,  # T-Note 10Y nível  — contexto macro
    "funding_z": 0.05,  # SOFR − Fed Funds  — stress overnight
}

# Eventos pontuais para marcar no gráfico
CRISIS_EVENTS = [
    ("2018-12-24", "Q4/2018"),
    ("2020-02-20", "COVID início"),
    ("2020-03-16", "COVID fundo"),
    ("2022-01-03", "Bear 2022"),
    ("2023-03-10", "SVB colapso"),
    ("2025-04-02", "Tarifas"),
]

# Janelas de crise para backtest
CRISIS_WINDOWS = [
    ("2018-10-01", "2019-01-31", "Selloff Q4/2018"),
    ("2020-02-15", "2020-05-31", "COVID 2020"),
    ("2022-01-01", "2022-12-31", "Bear Market 2022"),
    ("2023-03-01", "2023-05-31", "SVB 2023"),
    ("2025-02-01", "2025-04-30", "Tarifas 2025"),
]

LOOKBACK_DAYS = 45  # janela de antecipação no backtest (aumentada para 45d)

# ─── Funções de indicadores ───────────────────────────────────────────────────
def zscore_rolling(series: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    roll = series.rolling(window, min_periods=window // 2)
    std  = roll.std().where(lambda s: s > 0)
    return (series - roll.mean()) / std


def kre_stress(kre: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    peak = kre.rolling(window, min_periods=window // 2).max()
    return -(kre - peak) / peak * 100


def persistence_signal(score: pd.Series, threshold: float,
                       days: int = PERSIST_DAYS) -> pd.Series:
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

    fred = Fred(api_key=FRED_API_KEY)

    # T-Bill 3M (voo para segurança: quando cai abaixo do Fed Funds = stress)
    try:
        dtb3 = fred.get_series("DTB3", observation_start=START_DATE)
        df   = df.join(dtb3.rename("dtb3"), how="left")
    except Exception:
        df["dtb3"] = float("nan")

    # Curva 10Y − 2Y (negativa = invertida = stress)
    try:
        t10y2y = fred.get_series("T10Y2Y", observation_start=START_DATE)
        df     = df.join(t10y2y.rename("t10y2y"), how="left")
    except Exception:
        df["t10y2y"] = float("nan")

    df = df.ffill()

    # ── Componentes do score ──────────────────────────────────────────────────
    ind = pd.DataFrame(index=df.index)

    ind["t10y_z"]  = zscore_rolling(safe_col(df, "t10y"))
    ind["kre_z"]   = zscore_rolling(kre_stress(safe_col(df, "kre")))
    ind["hy_z"]    = zscore_rolling(safe_col(df, "hy_spread"))
    ind["vix_z"]   = zscore_rolling(safe_col(df, "vix"))

    # T-Bill stress: Fed Funds - DTB3  → positivo quando T-Bill cai (flight to safety)
    tbill_stress   = safe_col(df, "fed_funds") - safe_col(df, "dtb3")
    ind["tbill_z"] = zscore_rolling(tbill_stress)

    # Curva invertida: negamos T10Y2Y → positivo quando curva inverte = stress
    ind["curve_z"] = zscore_rolling(-safe_col(df, "t10y2y"))

    # SOFR − Fed Funds: stress de funding overnight
    sofr_ff          = safe_col(df, "sofr") - safe_col(df, "fed_funds")
    ind["funding_z"] = zscore_rolling(sofr_ff)

    # ── Score ponderado (renormalizado pelos componentes disponíveis) ──────────
    score   = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for comp_key, w in WEIGHTS.items():
        s = ind[comp_key]
        if s.notna().any():
            score   += w * s.fillna(0)
            total_w += w
    ind["score_raw"] = score / total_w if total_w > 0 else score

    # Suavização EMA — reduz ruído sem atraso material
    ind["score"] = ind["score_raw"].ewm(span=EMA_SPAN, adjust=False).mean()

    # Alertas com filtro de persistência (sobre score suavizado)
    ind["alert_warn"] = persistence_signal(ind["score"], WARN_THRESH)
    ind["alert_crit"] = persistence_signal(ind["score"], CRIT_THRESH)

    # ── QQQ para gráfico comparativo ──────────────────────────────────────────
    try:
        qqq_df = yf.download("QQQ", start=START_DATE, auto_adjust=True, progress=False)
        if qqq_df is not None and not qqq_df.empty:
            arr = qqq_df.filter(like="Close").to_numpy()
            qqq_col = arr[:, 0] if arr.ndim == 2 else arr
            qqq = pd.Series(qqq_col, index=qqq_df.index, name="QQQ", dtype=float)
        else:
            qqq = pd.Series(dtype=float, name="QQQ")
    except Exception:
        qqq = pd.Series(dtype=float, name="QQQ")

    return df, ind, qqq


# ─── Layout ───────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Liquidity Monitor v3")
st.title("📊 Monitor de Liquidez Sistêmica v3")
st.caption(
    f"Z-score rolante {ROLL_WINDOW}d · EMA-{EMA_SPAN} · "
    "Curva 10Y-2Y · T-Bill Stress · KRE drop-from-peak · "
    f"Persistência {PERSIST_DAYS}d · 7 componentes"
)

df, ind, qqq = load_all()

latest  = float(ind["score"].dropna().iloc[-1])
raw_lat = float(ind["score_raw"].dropna().iloc[-1])
in_crit = bool(ind["alert_crit"].iloc[-1])
in_warn = bool(ind["alert_warn"].iloc[-1])
status  = ("🔴 ALERTA CRÍTICO" if in_crit
           else "🟡 ATENÇÃO"    if in_warn
           else "🟢 Normal")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Score Suavizado (EMA-21)", f"{latest:.3f}",  status)
col_m2.metric("Score Bruto",              f"{raw_lat:.3f}", None)
col_m3.metric("Alerta",                   status,           None)

# ─── Helpers de layout ────────────────────────────────────────────────────────
_XAXIS = dict(tickangle=-45, tickformat="%b/%y", showgrid=False)
_YAXIS = dict(showgrid=True, gridcolor="#e5e7eb")
_MARGIN = dict(l=8, r=8, t=44, b=8)


def _add_crisis_bg(fig, windows=CRISIS_WINDOWS):
    for c_start, c_end, c_label in windows:
        fig.add_vrect(
            x0=c_start, x1=c_end,
            fillcolor="rgba(239,68,68,0.08)",
            layer="below", line_width=0,
            annotation_text=c_label,
            annotation_position="top left",
            annotation_font=dict(size=9, color="#b91c1c"),
        )


def _add_events(fig, score_min):
    for ev_date, ev_label in CRISIS_EVENTS:
        if pd.Timestamp(ev_date) < score_min:
            continue
        fig.add_vline(
            x=pd.Timestamp(ev_date).timestamp() * 1000,
            line_dash="dot", line_color="#dc2626", line_width=1,
            annotation_text=ev_label,
            annotation_position="top",
            annotation_font=dict(size=8, color="#dc2626"),
        )


def _add_thresholds(fig):
    fig.add_hline(y=WARN_THRESH, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Atenção {WARN_THRESH}",
                  annotation_position="top right",
                  annotation_font=dict(size=9))
    fig.add_hline(y=CRIT_THRESH, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Crítico {CRIT_THRESH}",
                  annotation_position="top right",
                  annotation_font=dict(size=9))


# ─── Gráfico principal — Score histórico ─────────────────────────────────────
st.subheader("Score histórico com eventos de crise")

fig_main = go.Figure()
_add_crisis_bg(fig_main)

score_s     = ind["score"].dropna()
score_raw_s = ind["score_raw"].dropna()

# Score bruto (fundo, tracejado leve)
fig_main.add_scatter(
    x=score_raw_s.index, y=score_raw_s,
    mode="lines", name="Score bruto",
    line=dict(color="#cbd5e1", width=1.0, dash="dot"),
    opacity=0.6,
)

# Score suavizado (linha principal)
fig_main.add_scatter(
    x=score_s.index, y=score_s,
    mode="lines", name=f"Score EMA-{EMA_SPAN}",
    line=dict(color="#3b82f6", width=2.0),
)

# Faixas coloridas de alerta
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

_add_thresholds(fig_main)
_add_events(fig_main, score_s.index.min())

fig_main.update_layout(
    height=360, margin=_MARGIN,
    paper_bgcolor="white", plot_bgcolor="white",
    showlegend=True,
    legend=dict(orientation="h", y=1.06, x=0, xanchor="left", font=dict(size=10)),
    xaxis=_XAXIS, yaxis=_YAXIS,
)
st.plotly_chart(fig_main, width="stretch")

# ─── Score vs QQQ ─────────────────────────────────────────────────────────────
st.subheader("Score de Liquidez vs QQQ — Invesco QQQ Trust (Nasdaq-100)")

fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
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

_add_thresholds(fig_comp)
_add_events(fig_comp, score_s.index.min())

fig_comp.update_layout(
    height=380,
    margin=dict(l=8, r=60, t=44, b=8),
    paper_bgcolor="white", plot_bgcolor="white",
    showlegend=True,
    legend=dict(orientation="h", y=1.06, x=0, xanchor="left", font=dict(size=11)),
    xaxis=_XAXIS,
    yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="Score"),
    yaxis2=dict(showgrid=False, title="QQQ (USD)", side="right"),
)
st.plotly_chart(fig_comp, width="stretch")

# ─── Componentes do score ─────────────────────────────────────────────────────
st.subheader("Componentes do score (z-score rolante 252d)")

COMPONENTS = [
    ("hy_z",      "HY Spread",                "#a78bfa", "25%"),
    ("tbill_z",   "T-Bill 3M Stress",         "#06b6d4", "20%"),
    ("kre_z",     "KRE drop-from-peak",       "#10b981", "20%"),
    ("curve_z",   "Curva 10Y-2Y (inv.)",      "#f43f5e", "15%"),
    ("vix_z",     "VIX",                      "#f97316", "10%"),
    ("t10y_z",    "T-Note 10Y",               "#3b82f6",  "5%"),
    ("funding_z", "SOFR − Fed Funds",         "#8b5cf6",  "5%"),
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
        xaxis=_XAXIS, yaxis=_YAXIS,
    )
    with grid[i % 3]:
        st.plotly_chart(fc, width="stretch")

# ─── Backtest ─────────────────────────────────────────────────────────────────
st.subheader(f"Backtest — detecção com antecipação de até {LOOKBACK_DAYS} dias")

data_start = ind.index.min()
bt_rows    = []

for c_start, c_end, c_label in CRISIS_WINDOWS:
    ts_start = pd.Timestamp(c_start)
    ts_end   = pd.Timestamp(c_end)
    pre_from = ts_start - pd.Timedelta(days=LOOKBACK_DAYS)

    if ts_end < data_start:
        bt_rows.append({
            "Evento": c_label, "Detectado antes?": "⬜ Sem dados",
            "Antecipação (dias)": "—", "Score máx. pré-crise": "—",
            "Score máx. na crise": "—",
        })
        continue

    pre_from_eff = max(pre_from, data_start)
    pre_score    = ind.loc[pre_from_eff:ts_start, "score"]
    cris_score   = ind.loc[ts_start:ts_end,       "score"]
    pre_alert    = ind.loc[pre_from_eff:ts_start,  "alert_crit"]
    cris_alert   = ind.loc[ts_start:ts_end,        "alert_crit"]

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

buffer      = pd.Timedelta(days=30)
crisis_mask = pd.Series(False, index=ind.index)
for c_start, c_end, _ in CRISIS_WINDOWS:
    crisis_mask.loc[pd.Timestamp(c_start) - buffer : pd.Timestamp(c_end) + buffer] = True

available  = ind["alert_crit"].notna()
fp_days    = int((ind["alert_crit"] & ~crisis_mask & available).sum())
total_days = int(available.sum())

st.dataframe(
    pd.DataFrame(bt_rows),
    width="stretch", hide_index=True,
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
st.subheader("Composição do score v3")

st.dataframe(
    pd.DataFrame([
        {
            "Indicador":  "HY Spread de Crédito",
            "Peso":       "25%",
            "Tipo":       "Leading",
            "Fonte":      "FRED — BAMLH0A0HYM2",
            "Fundamento": "Gilchrist & Zakrajsek (2012): melhor preditor individual de recessão",
        },
        {
            "Indicador":  "T-Bill 3M Stress (FF − DTB3)",
            "Peso":       "20%",
            "Tipo":       "Leading",
            "Fonte":      "FRED — DTB3 + FEDFUNDS",
            "Fundamento": "Substituto do TED pós-LIBOR; voo para T-Bills precede crises de funding",
        },
        {
            "Indicador":  "KRE — queda do pico",
            "Peso":       "20%",
            "Tipo":       "Leading",
            "Fonte":      "Yahoo Finance — KRE",
            "Fundamento": "Deterioração bancária gradual; mais robusto que retorno diário",
        },
        {
            "Indicador":  "Curva 10Y-2Y (invertida)",
            "Peso":       "15%",
            "Tipo":       "Leading",
            "Fonte":      "FRED — T10Y2Y",
            "Fundamento": "Inversão precede recessões; captura bear market 2022 e ciclo de aperto",
        },
        {
            "Indicador":  "VIX",
            "Peso":       "10%",
            "Tipo":       "Coincidente",
            "Fonte":      "Yahoo Finance — ^VIX",
            "Fundamento": "Medo do mercado; tende a disparar junto com, não antes do crash",
        },
        {
            "Indicador":  "T-Note 10 anos",
            "Peso":        "5%",
            "Tipo":       "Contexto",
            "Fonte":      "Yahoo Finance — ^TNX",
            "Fundamento": "Nível de juros longos; redundante com a curva, peso reduzido",
        },
        {
            "Indicador":  "SOFR − Fed Funds",
            "Peso":        "5%",
            "Tipo":       "Leading",
            "Fonte":      "FRED — SOFR + FEDFUNDS",
            "Fundamento": "Stress overnight; histórico curto (desde abr/2018), peso mínimo",
        },
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
