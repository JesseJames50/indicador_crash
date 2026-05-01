# ============================================================
# COMPARADOR B3 v5 — RS Multiativos (Benchmark = Último)
# Adaptado do Google Colab para VS Code com Streamlit
# ============================================================
# Como usar:
#   pip install streamlit yfinance pandas numpy plotly
#   streamlit run b3_comparador.py
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Configurações ───────────────────────────────────────────
PERIOD_MAP = {
    "1M": 30, "3M": 90, "6M": 180,
    "1A": 365, "2A": 730, "5A": 1825, "10A": 3650,
}

COLORS = ["#F5C518","#00E676","#FF6B6B","#40C4FF",
          "#CE93D8","#FFAB40","#80CBC4","#EF9A9A","#B0BEC5","#A5D6A7"]

DARK_BG   = "#131722"
GRID_CLR  = "#2A2E39"
TEXT_CLR  = "#D1D4DC"
PANEL_CLR = "#1E222D"


# ── Download (idêntico ao Colab) ───────────────────────────
@st.cache_data(ttl=300)
def get_close(tickers: tuple, start: str, end: str) -> pd.DataFrame:

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = [tickers[0]]

    close.columns = [str(c).upper() for c in close.columns]
    return close.dropna(how="all")


# ── Função principal (idêntica ao Colab) ───────────────────
def plotar(tickers_raw: str, periodo: str, filtro: str):

    # ── Ajuste automático B3 (.SA) ───────────────────────────
    tickers_input = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers_input:
        st.warning("⚠️ Nenhum ticker informado.")
        return

    ticker_map = {}   # yahoo_ticker -> nome original
    tickers = []

    for t in tickers_input:
        yahoo_ticker = t if "." in t else t + ".SA"
        ticker_map[yahoo_ticker] = t
        tickers.append(yahoo_ticker)

    # ── Período ──────────────────────────────────────────────
    days  = PERIOD_MAP.get(periodo, 365)
    end   = datetime.today()
    start = end - timedelta(days=days)

    with st.spinner("Baixando dados da B3..."):
        close = get_close(
            tuple(tickers),
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )

    if close.empty:
        st.error("❌ Nenhum dado retornado. Verifique os tickers e sua conexão.")
        return

    found = [t for t in tickers if t in close.columns]

    if not found:
        st.error("❌ Nenhum ativo disponível.")
        return

    missing = [ticker_map[t] for t in tickers if t not in close.columns]
    if missing:
        st.warning(f"⚠️ Não encontrados (ignorados): {', '.join(missing)}")

    close = close[found].dropna()

    # ── Filtro principal ─────────────────────────────────────
    if filtro == "Preço":
        df_plot = close.copy()
        y_title = "Preço (R$)"
    elif filtro == "Base 100":
        base    = close.iloc[0]
        df_plot = (close / base) * 100
        y_title = "Base 100"
    else:
        base    = close.iloc[0]
        df_plot = ((close / base) - 1) * 100
        y_title = "Variação (%)"

    # ── Criar subplots ───────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    # ── Helper: label inline no final da linha ──────────────
    annotations = []

    def _label(s, text, color, yref):
        return dict(
            x=s.index[-1],
            y=float(s.iloc[-1]),
            xref="x", yref=yref,
            text=f"<b>{text}</b>",
            showarrow=False,
            xanchor="left", yanchor="middle",
            xshift=6,
            font=dict(color=color, size=10),
            bgcolor="rgba(19,23,34,0.80)",
            borderpad=2,
        )

    # ── Painel 1: Comparação ─────────────────────────────────
    for i, ticker in enumerate(found):
        s     = df_plot[ticker]
        color = COLORS[i % len(COLORS)]
        nome  = ticker_map[ticker]

        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values,
                mode="lines",
                name=nome,
                line=dict(color=color, width=1.8),
            ),
            row=1, col=1,
        )

        # Valor final para o label
        val = float(s.iloc[-1])
        if filtro == "Preço":
            val_str = f"R${val:.2f}"
        elif filtro == "Base 100":
            val_str = f"{val - 100:+.1f}"
        else:
            val_str = f"{val:+.1f}%"

        annotations.append(_label(s, f"{nome}  {val_str}", color, "y"))

    if filtro != "Preço":
        ref_y = 100 if filtro == "Base 100" else 0
        fig.add_hline(y=ref_y, line_dash="dot",
                      line_color=GRID_CLR, row=1, col=1)

    # ── Painel 2: RS (Benchmark = Último) ────────────────────
    if len(found) >= 2:
        benchmark = found[-1]
        ativos_rs = found[:-1]

        for i, ativo in enumerate(ativos_rs):
            rs      = close[ativo] / close[benchmark]
            rs_norm = (rs / rs.iloc[0]) * 100
            color   = COLORS[i % len(COLORS)]
            nome_a  = ticker_map[ativo]
            nome_b  = ticker_map[benchmark]
            rs_chg  = float(rs_norm.iloc[-1]) - 100

            fig.add_trace(
                go.Scatter(
                    x=rs_norm.index, y=rs_norm.values,
                    mode="lines",
                    name=f"RS ({nome_a}/{nome_b})",
                    line=dict(color=color, width=2),
                ),
                row=2, col=1,
            )

            annotations.append(_label(rs_norm,
                                       f"RS {nome_a}/{nome_b}  {rs_chg:+.1f}",
                                       color, "y2"))

        fig.add_hline(y=100, line_dash="dot",
                      line_color=GRID_CLR, row=2, col=1)

    # ── Painel 3: Retorno diário (primeiro ativo) ────────────
    ref      = found[0]
    ret_d    = close[ref].pct_change().dropna() * 100
    bar_cols = ["#00E676" if v >= 0 else "#FF6B6B" for v in ret_d]

    fig.add_trace(
        go.Bar(
            x=ret_d.index, y=ret_d.values,
            marker_color=bar_cols,
            showlegend=False,
        ),
        row=3, col=1,
    )

    # ── Layout ───────────────────────────────────────────────
    fig.update_layout(
        title=f"B3: {', '.join([ticker_map[t] for t in found])} | {periodo} | {filtro}",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        hovermode="x unified",
        height=850,
        font=dict(color=TEXT_CLR),
        legend=dict(
            orientation="h",
            y=1.02, x=0,
            bgcolor=PANEL_CLR,
            font=dict(color="#FFFFFF", size=12),
        ),
        # Margem direita extra para os labels não serem cortados
        margin=dict(r=130),
        annotations=annotations,
    )

    ax = dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)

    fig.update_yaxes(title_text=y_title,           row=1, col=1)
    fig.update_yaxes(title_text="RS Base 100",     row=2, col=1)
    fig.update_yaxes(title_text="Var. Diária (%)", row=3, col=1)

    st.plotly_chart(fig, width='stretch')

    # ── Tabela de resumo ─────────────────────────────────────
    st.subheader("📊 Resumo do Período")
    rows = []
    for ticker in found:
        s       = close[ticker]
        retorno = (s.iloc[-1] / s.iloc[0] - 1) * 100
        vol     = s.pct_change().std() * np.sqrt(252) * 100
        rows.append({
            "Ticker":       ticker_map[ticker],
            "Retorno (%)":  f"{retorno:+.2f}%",
            "Volatilidade": f"{vol:.2f}%",
            "Máximo":       f"R$ {s.max():.2f}",
            "Mínimo":       f"R$ {s.min():.2f}",
            "Último Preço": f"R$ {s.iloc[-1]:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ── Interface Streamlit ─────────────────────────────────────
def main():
    st.set_page_config(page_title="B3 Comparador", page_icon="🇧🇷", layout="wide")

    if "params" not in st.session_state:
        st.session_state.params = None

    st.title("🇧🇷 Comparador B3 — Força Relativa")
    st.caption("Compare ações da B3 com análise de Força Relativa (RS) e retorno diário.")

    with st.sidebar:
        st.header("⚙️ Configurações")

        tickers_raw = st.text_input(
            "Tickers B3 (separados por vírgula)",
            value="PETR4, VALE3, BOVA11",
            help="Digite os códigos da B3 normalmente. Ex: PETR4, VALE3, BOVA11\nO sufixo .SA é adicionado automaticamente.",
        )

        periodo = st.selectbox("Período", list(PERIOD_MAP.keys()), index=3)
        filtro  = st.selectbox("Visualização", ["Base 100", "Percentual", "Preço"])

        st.divider()
        st.info(
            "💡 **Força Relativa:** O **último ticker** é usado como benchmark.\n\n"
            "📌 O sufixo **.SA** é adicionado automaticamente."
        )

        if st.button("🔄 Atualizar", width='stretch', type="primary"):
            get_close.clear()
            st.session_state.params = (tickers_raw, periodo, filtro)

    if st.session_state.params is None:
        st.session_state.params = (tickers_raw, periodo, filtro)

    plotar(*st.session_state.params)


if __name__ == "__main__":
    main()

