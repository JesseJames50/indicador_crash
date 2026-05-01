# ============================================================
# ETF COMPARADOR NASDAQ v4 — Com Força Relativa (RS)
# Versão adaptada para VS Code com Streamlit
# ============================================================
# Como usar:
#   1. Instale as dependências:
#      pip install streamlit yfinance pandas numpy plotly
#
#   2. Execute no terminal:
#      streamlit run etf_comparador.py
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

COLORS = [
    "#F5C518", "#00E676", "#FF6B6B", "#40C4FF",
    "#CE93D8", "#FFAB40", "#80CBC4", "#EF9A9A", "#B0BEC5", "#A5D6A7"
]

DARK_BG   = "#131722"
GRID_CLR  = "#2A2E39"
TEXT_CLR  = "#D1D4DC"
PANEL_CLR = "#1E222D"


# ── Download robusto ───────────────────────────────────────
@st.cache_data(ttl=300)  # cache de 5 minutos
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


# ── Função de plotagem ──────────────────────────────────────
def plotar(tickers_raw: str, periodo: str, filtro: str):

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.warning("⚠️ Nenhum ticker informado.")
        return

    days  = PERIOD_MAP.get(periodo, 365)
    end   = datetime.today()
    start = end - timedelta(days=days)

    with st.spinner("Baixando dados..."):
        close = get_close(
            tuple(tickers),
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )

    if close.empty:
        st.error("❌ Nenhum dado retornado. Verifique os tickers.")
        return

    found = [t for t in tickers if t in close.columns]
    if not found:
        st.error("❌ Nenhum ativo disponível.")
        return

    # ffill/bfill resolve dias de pregão diferentes entre ativos de bolsas distintas
    # (ex: SPY vs TSLA vs ativos europeus). dropna(how="any") zeraria o DataFrame.
    close = close[found].ffill().bfill().dropna(how="all")

    if close.empty:
        st.error("❌ Dados insuficientes após alinhamento dos calendários. Tente outro período.")
        return

    # ── Filtro principal ─────────────────────────────────────
    if filtro == "Preço":
        df_plot   = close.copy()
        y_title   = "Preço (USD)"

    elif filtro == "Base 100":
        base    = close.iloc[0]
        df_plot = (close / base) * 100
        y_title = "Base 100"

    else:  # Percentual
        base    = close.iloc[0]
        df_plot = ((close / base) - 1) * 100
        y_title = "Variação (%)"

    # ── Criar subplots (3 painéis) ───────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    # ── Painel 1: Comparação ─────────────────────────────────
    # Benchmark (último ticker) é plotado primeiro (camada de baixo)
    benchmark_idx = len(found) - 1
    benchmark     = found[benchmark_idx]
    plot_order    = [benchmark_idx] + [i for i in range(len(found)) if i != benchmark_idx]

    annotations = []   # labels inline ao final de cada linha

    def _end_label(s, label, color, yref="y"):
        """Cria anotação no ponto final da série."""
        last_x = s.index[-1]
        last_y = float(s.iloc[-1])
        return dict(
            x=last_x,
            y=last_y,
            xref="x",
            yref=yref,
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=6,
            font=dict(color=color, size=10),
            bgcolor="rgba(19,23,34,0.75)",
        )

    for plot_i, i in enumerate(plot_order):
        ticker       = found[i]
        s            = df_plot[ticker]
        color        = COLORS[i % len(COLORS)]
        is_benchmark = (i == benchmark_idx)
        width        = 1.5 if is_benchmark else 2.2
        dash_style   = "dot" if is_benchmark else "solid"

        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values,
                mode="lines",
                name=ticker,
                line=dict(color=color, width=width, dash=dash_style),
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Label inline: nome + valor final
        val = float(df_plot[ticker].iloc[-1])
        if filtro == "Preço":
            val_str = f"${val:.2f}"
        elif filtro == "Base 100":
            chg = val - 100
            val_str = f"{chg:+.1f}"
        else:
            val_str = f"{val:+.1f}%"

        label = f"{ticker}  {val_str}"
        annotations.append(_end_label(df_plot[ticker], label, color, yref="y"))

    if filtro != "Preço":
        ref_y = 100 if filtro == "Base 100" else 0
        fig.add_hline(y=ref_y, line_dash="dot",
                      line_color=GRID_CLR, row=1, col=1)

    # ── Painel 2: Força Relativa (Benchmark = último ticker) ──
    if len(found) >= 2:
        ativos_rs = found[:-1]

        for i, ativo in enumerate(ativos_rs):
            rs      = close[ativo] / close[benchmark]
            rs_norm = (rs / rs.iloc[0]) * 100
            color   = COLORS[i % len(COLORS)]
            rs_chg  = float(rs_norm.iloc[-1]) - 100

            fig.add_trace(
                go.Scatter(
                    x=rs_norm.index,
                    y=rs_norm.values,
                    mode="lines",
                    name=f"RS {ativo}/{benchmark}",
                    line=dict(color=color, width=2),
                    showlegend=True,
                ),
                row=2, col=1,
            )

            label_rs = f"RS {ativo}/{benchmark}  {rs_chg:+.1f}"
            annotations.append(_end_label(rs_norm, label_rs, color, yref="y2"))

        fig.add_hline(y=100, line_dash="dot",
                      line_color=GRID_CLR, row=2, col=1)

    # ── Painel 3: Retorno diário ─────────────────────────────
    ref     = found[0]
    ret_d   = close[ref].pct_change().dropna() * 100
    bar_cols = ["#00E676" if v >= 0 else "#FF6B6B" for v in ret_d]

    fig.add_trace(
        go.Bar(
            x=ret_d.index,
            y=ret_d.values,
            marker_color=bar_cols,
            name="Var. Diária",
            showlegend=False,
        ),
        row=3, col=1,
    )

    # ── Layout ───────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"ETFs: {', '.join(found)} | {periodo} | {filtro}",
            font=dict(color=TEXT_CLR, size=14),
            x=0,
            xanchor="left",
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        hovermode="x unified",
        height=950,
        font=dict(color=TEXT_CLR, size=12),
        # Legenda vertical à esquerda — sem sobreposição com o gráfico
        legend=dict(
            orientation="v",
            x=-0.01,
            y=1.0,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(19,23,34,0.92)",
            bordercolor="#555966",
            borderwidth=1,
            font=dict(color="#FFFFFF", size=11),
            tracegroupgap=6,
            itemsizing="constant",
        ),
        margin=dict(l=160, r=100, t=80, b=60),
        annotations=annotations,
    )

    ax = dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)

    fig.update_yaxes(title_text=y_title,         row=1, col=1)
    fig.update_yaxes(title_text="RS Base 100",   row=2, col=1)
    fig.update_yaxes(title_text="Var. Diária (%)", row=3, col=1)

    st.plotly_chart(fig, width='stretch')

    # ── Tabela de resumo ─────────────────────────────────────
    st.subheader("📊 Resumo do Período")
    rows = []
    for ticker in found:
        s      = close[ticker]
        retorno = (s.iloc[-1] / s.iloc[0] - 1) * 100
        vol     = s.pct_change().std() * np.sqrt(252) * 100
        maximo  = s.max()
        minimo  = s.min()
        rows.append({
            "Ticker":       ticker,
            "Retorno (%)":  f"{retorno:+.2f}%",
            "Volatilidade": f"{vol:.2f}%",
            "Máximo":       f"${maximo:.2f}",
            "Mínimo":       f"${minimo:.2f}",
            "Último Preço": f"${s.iloc[-1]:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ── Interface Streamlit ─────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ETF Comparador",
        page_icon="📈",
        layout="wide",
    )

    # ── Inicializa session_state ─────────────────────────────
    if "params" not in st.session_state:
        st.session_state.params = None   # None = ainda não plotou

    # Cabeçalho
    st.title("📈 ETF Comparador — Força Relativa")
    st.caption("Compare ETFs com análise de Força Relativa (RS) e retorno diário.")

    # Sidebar com controles
    with st.sidebar:
        st.header("⚙️ Configurações")

        tickers_raw = st.text_input(
            "Tickers (separados por vírgula)",
            value="AAXJ, SPY",
            help="Ex: AAXJ, QQQ, SPY, VTI",
        )

        periodo = st.selectbox(
            "Período",
            options=list(PERIOD_MAP.keys()),
            index=3,  # padrão: 1A
        )

        filtro = st.selectbox(
            "Visualização",
            options=["Base 100", "Percentual", "Preço"],
            index=0,
        )

        st.divider()
        st.info(
            "💡 **Força Relativa:** O **último ticker** informado é usado "
            "como benchmark. Os demais são comparados a ele no Painel 2."
        )

        atualizar = st.button("🔄 Atualizar", width='stretch', type="primary")

    # ── Lógica do botão ──────────────────────────────────────
    if atualizar:
        # Limpa o cache para forçar novo download de dados
        get_close.clear()
        st.session_state.params = (tickers_raw, periodo, filtro)

    # Primeira carga: plota automaticamente com os valores padrão
    if st.session_state.params is None:
        st.session_state.params = (tickers_raw, periodo, filtro)

    # Plota usando os parâmetros salvos no estado
    plotar(*st.session_state.params)


if __name__ == "__main__":
    main()
