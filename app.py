import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import merge_data
from indicators import compute_indicators

st.set_page_config(layout="wide")
st.title("📊 Monitor de Liquidez Sistêmica")

CHART_H      = 220   # altura dos gráficos auxiliares (px)
MAIN_CHART_H = 260   # altura do gráfico principal (px)

XAXIS_COMMON = dict(
    tickangle=-45,
    tickformat="%b/%y",
    showgrid=False,
)
YAXIS_COMMON = dict(showgrid=True, gridcolor="#e5e7eb")
MARGIN = dict(l=8, r=8, t=40, b=8)


def make_fig(height: int, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=MARGIN,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        title=dict(
            text=title,
            font=dict(size=13, color="#374151"),
            x=0,
            xanchor="left",
            pad=dict(l=4),
        ),
        xaxis=XAXIS_COMMON,
        yaxis=YAXIS_COMMON,
    )
    return fig


@st.cache_data
def load():
    df = merge_data()
    indicators = compute_indicators(df)
    return df, indicators


df, ind = load()

latest_score = ind["systemic_score"].iloc[-1]


def classify(score: float) -> str:
    if score < 0.5:
        return "🟢 Normal"
    elif score < 1.5:
        return "🟡 Atenção"
    return "🔴 Crítico"


st.metric("Score Sistêmico", round(latest_score, 2), classify(latest_score))

# ── Gráfico principal ────────────────────────────────────────────────────────
fig_main = make_fig(MAIN_CHART_H, title="Score Sistêmico")
fig_main.add_scatter(
    x=ind.index, y=ind["systemic_score"],
    mode="lines", name="Score sistêmico",
    line=dict(color="#3b82f6", width=2),
)
fig_main.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b",
                   annotation_text="Atenção 0.5", annotation_position="top left")
fig_main.add_hline(y=1.5, line_dash="dash", line_color="#ef4444",
                   annotation_text="Crítico 1.5", annotation_position="top left")
st.plotly_chart(fig_main, width="stretch")

# ── Gráficos auxiliares ──────────────────────────────────────────────────────
st.subheader("Indicadores")

INDICATORS = [
    ("t10y",      "T-Note 10 anos (%)",         "#3b82f6"),
    ("kre",       "KRE — Bancos Regionais ($)",  "#10b981"),
    ("hy_spread", "HY Spread de Crédito (%)",    "#a78bfa"),
    ("vix",       "VIX — Volatilidade",          "#f97316"),
]

col1, col2 = st.columns(2)
cols = [col1, col2, col1, col2]

for (col_key, label, color), col in zip(INDICATORS, cols):
    fig = make_fig(CHART_H, title=label)
    fig.add_scatter(
        x=df.index, y=df[col_key],
        mode="lines", name=label,
        line=dict(color=color, width=1.6),
    )
    with col:
        st.plotly_chart(fig, width="stretch")

# ── Tabela de referência ─────────────────────────────────────────────────────
st.subheader("Guia de níveis de referência")

REF = pd.DataFrame([
    {
        "Indicador":      "① T-Note 10 anos",
        "Peso no score":  "20%",
        "🟢 Normal":      "< 4,0%",
        "🟡 Atenção":     "4,0% – 4,5%",
        "🔴 Crítico":     "> 4,5%",
        "O que sinaliza": "Custo do crédito americano; juros altos comprimem ativos de risco.",
    },
    {
        "Indicador":      "② KRE — Bancos Regionais",
        "Peso no score":  "20%",
        "🟢 Normal":      "Queda < 15% do pico",
        "🟡 Atenção":     "Queda 15% – 30%",
        "🔴 Crítico":     "Queda > 30%",
        "O que sinaliza": "Estresse no sistema bancário regional; risco de corrida bancária.",
    },
    {
        "Indicador":      "③ HY Spread de Crédito",
        "Peso no score":  "20%",
        "🟢 Normal":      "< 4,0%",
        "🟡 Atenção":     "4,0% – 6,0%",
        "🔴 Crítico":     "> 6,0%",
        "O que sinaliza": "Prêmio de risco corporativo; spreads altos indicam fuga de capital.",
    },
    {
        "Indicador":      "④ VIX — Volatilidade",
        "Peso no score":  "20%",
        "🟢 Normal":      "< 20",
        "🟡 Atenção":     "20 – 30",
        "🔴 Crítico":     "> 30",
        "O que sinaliza": "Medo do mercado acionário americano; picos refletem liquidações.",
    },
    {
        "Indicador":      "⑤ Spread SOFR – Fed Funds",
        "Peso no score":  "20%",
        "🟢 Normal":      "≈ 0%",
        "🟡 Atenção":     "0,1% – 0,3%",
        "🔴 Crítico":     "> 0,3%",
        "O que sinaliza": "Estresse no mercado interbancário overnight; funding sob pressão.",
    },
])

st.dataframe(
    REF,
    width="stretch",
    hide_index=True,
    column_config={
        "Indicador":      st.column_config.TextColumn(width="medium"),
        "Peso no score":  st.column_config.TextColumn(width="small"),
        "🟢 Normal":      st.column_config.TextColumn(width="small"),
        "🟡 Atenção":     st.column_config.TextColumn(width="small"),
        "🔴 Crítico":     st.column_config.TextColumn(width="small"),
        "O que sinaliza": st.column_config.TextColumn(width="large"),
    },
)
