# ============================================================
# COMPARADOR MULTIMERCADO — B3 & ETF/Internacional
# Streamlit + yfinance + Plotly
# ============================================================
# Como usar:
#   pip install streamlit yfinance pandas numpy plotly
#   streamlit run comparador.py
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
    "#CE93D8", "#FFAB40", "#80CBC4", "#EF9A9A", "#B0BEC5", "#A5D6A7",
]

DARK_BG   = "#131722"
GRID_CLR  = "#2A2E39"
TEXT_CLR  = "#D1D4DC"
PANEL_CLR = "#1E222D"

DEFAULTS = {
    "B3":                "PETR4, VALE3, BOVA11",
    "ETF / Internacional": "AAXJ, SPY",
}


# ── Download ────────────────────────────────────────────────
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


# ── Plotagem ────────────────────────────────────────────────
def plotar(tickers_raw: str, periodo: str, filtro: str, modo: str):
    is_b3    = (modo == "B3")
    currency = "R$" if is_b3 else "$"

    # ── Parsing de tickers ────────────────────────────────
    tickers_input = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers_input:
        st.warning("⚠️ Nenhum ticker informado.")
        return

    if is_b3:
        ticker_map = {}
        tickers    = []
        for t in tickers_input:
            yahoo = t if "." in t else t + ".SA"
            ticker_map[yahoo] = t
            tickers.append(yahoo)
    else:
        tickers    = tickers_input
        ticker_map = {t: t for t in tickers}

    # ── Período ───────────────────────────────────────────
    days  = PERIOD_MAP.get(periodo, 365)
    end   = datetime.today()
    start = end - timedelta(days=days)

    spinner_msg = "Baixando dados da B3..." if is_b3 else "Baixando dados..."
    with st.spinner(spinner_msg):
        close = get_close(
            tuple(tickers),
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )

    if close.empty:
        st.error("❌ Nenhum dado retornado. Verifique os tickers e sua conexão.")
        return

    found   = [t for t in tickers if t in close.columns]
    missing = [ticker_map[t] for t in tickers if t not in close.columns]

    if not found:
        st.error("❌ Nenhum ativo disponível.")
        return
    if missing:
        st.warning(f"⚠️ Não encontrados (ignorados): {', '.join(missing)}")

    close = close[found]

    # ETF: alinha calendários de pregões distintos (ex: SPY vs ativos europeus)
    if not is_b3:
        close = close.ffill().bfill()

    close = close.dropna(how="all")

    if close.empty:
        st.error("❌ Dados insuficientes após alinhamento. Tente outro período.")
        return

    # ── Filtro ────────────────────────────────────────────
    if filtro == "Preço":
        df_plot = close.copy()
        y_title = f"Preço ({currency})"
    elif filtro == "Base 100":
        base    = close.iloc[0]
        df_plot = (close / base) * 100
        y_title = "Base 100"
    else:
        base    = close.iloc[0]
        df_plot = ((close / base) - 1) * 100
        y_title = "Variação (%)"

    # ── Subplots ──────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    annotations   = []
    benchmark_idx = len(found) - 1
    benchmark     = found[benchmark_idx]

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

    # ── Painel 1: Comparação ──────────────────────────────
    # No modo ETF, benchmark é plotado primeiro (camada de baixo)
    if is_b3:
        plot_order = list(range(len(found)))
    else:
        plot_order = [benchmark_idx] + [i for i in range(len(found)) if i != benchmark_idx]

    for i in plot_order:
        ticker = found[i]
        s      = df_plot[ticker]
        color  = COLORS[i % len(COLORS)]
        nome   = ticker_map[ticker]
        is_bm  = (i == benchmark_idx)

        # Benchmark aparece tracejado/fino só no modo ETF
        width      = 1.5 if (not is_b3 and is_bm) else (1.8 if is_b3 else 2.2)
        dash_style = "dot" if (not is_b3 and is_bm) else "solid"

        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values,
                mode="lines",
                name=nome,
                line=dict(color=color, width=width, dash=dash_style),
            ),
            row=1, col=1,
        )

        val = float(s.iloc[-1])
        if filtro == "Preço":
            val_str = f"{currency}{val:.2f}"
        elif filtro == "Base 100":
            val_str = f"{val - 100:+.1f}"
        else:
            val_str = f"{val:+.1f}%"

        annotations.append(_label(s, f"{nome}  {val_str}", color, "y"))

    if filtro != "Preço":
        ref_y = 100 if filtro == "Base 100" else 0
        fig.add_hline(y=ref_y, line_dash="dot", line_color=GRID_CLR, row=1, col=1)

    # ── Painel 2: Força Relativa ───────────────────────────
    if len(found) >= 2:
        for i, ativo in enumerate(found[:-1]):
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
            annotations.append(
                _label(rs_norm, f"RS {nome_a}/{nome_b}  {rs_chg:+.1f}", color, "y2")
            )

        fig.add_hline(y=100, line_dash="dot", line_color=GRID_CLR, row=2, col=1)

    # ── Painel 3: Correlação móvel vs benchmark ───────────
    # Janela de 21 pregões (~1 mês); ajustada se o período for curto
    CORR_WINDOW = min(21, max(5, len(close) // 5))
    rets = close.pct_change()

    if len(found) >= 2:
        for i, ativo in enumerate(found[:-1]):
            # Sem dropna(): o índice permanece igual ao de close, garantindo
            # alinhamento perfeito do eixo X. Os primeiros CORR_WINDOW-1 valores
            # ficam NaN e Plotly simplesmente não os traça (connectgaps=False).
            corr       = rets[ativo].rolling(CORR_WINDOW).corr(rets[benchmark])
            valid_corr = corr.dropna()

            # Pula o par se não houver dados suficientes para a janela
            if valid_corr.empty:
                continue

            color  = COLORS[i % len(COLORS)]
            nome_a = ticker_map[ativo]
            nome_b = ticker_map[benchmark]
            last_c = float(valid_corr.iloc[-1])

            fig.add_trace(
                go.Scatter(
                    x=corr.index, y=corr.values,
                    mode="lines",
                    name=f"Corr ({nome_a}/{nome_b})",
                    line=dict(color=color, width=1.5),
                    connectgaps=False,
                    showlegend=True,
                ),
                row=3, col=1,
            )
            # Label posicionado no último ponto válido da série
            annotations.append(
                _label(valid_corr, f"Corr {nome_a}/{nome_b}  {last_c:+.2f}", color, "y3")
            )

    fig.add_hline(y=0, line_dash="dot", line_color=GRID_CLR, row=3, col=1)

    # ── Layout ────────────────────────────────────────────
    nomes_display = [ticker_map[t] for t in found]
    title_prefix  = "🇧🇷 B3" if is_b3 else "📈 ETF"

    legend_cfg = (
        dict(
            orientation="h",
            y=1.02, x=0,
            bgcolor=PANEL_CLR,
            font=dict(color="#FFFFFF", size=12),
        )
        if is_b3
        else dict(
            orientation="v",
            x=-0.01, y=1.0,
            xanchor="right", yanchor="top",
            bgcolor="rgba(19,23,34,0.92)",
            bordercolor="#555966",
            borderwidth=1,
            font=dict(color="#FFFFFF", size=11),
            tracegroupgap=6,
            itemsizing="constant",
        )
    )

    margin_cfg = dict(r=130) if is_b3 else dict(l=160, r=100, t=80, b=60)

    fig.update_layout(
        title=f"{title_prefix}: {', '.join(nomes_display)} | {periodo} | {filtro}",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        hovermode="x unified",
        height=900,
        font=dict(color=TEXT_CLR),
        legend=legend_cfg,
        margin=margin_cfg,
        annotations=annotations,
    )

    # Força o mesmo intervalo de datas em todos os painéis (evita desalinhamento
    # causado por séries com índices de tamanhos diferentes, ex: correlação móvel)
    x_range = [close.index[0], close.index[-1]]

    ax = dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig.update_xaxes(**ax, range=x_range)
    fig.update_yaxes(**ax)

    fig.update_yaxes(title_text=y_title,                      row=1, col=1)
    fig.update_yaxes(title_text="RS Base 100",                row=2, col=1)
    fig.update_yaxes(title_text=f"Corr {CORR_WINDOW}d",
                     range=[-1, 1],                           row=3, col=1)

    st.plotly_chart(fig, width='stretch')

    # ── Tabela de resumo ──────────────────────────────────
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
            "Máximo":       f"{currency} {s.max():.2f}",
            "Mínimo":       f"{currency} {s.min():.2f}",
            "Último Preço": f"{currency} {s.iloc[-1]:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ── Interface Streamlit ─────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Comparador Multimercado",
        page_icon="📊",
        layout="wide",
    )

    if "params"        not in st.session_state:
        st.session_state.params        = None
    if "modo_anterior" not in st.session_state:
        st.session_state.modo_anterior = None

    st.title("📊 Comparador Multimercado — Força Relativa")
    st.caption(
        "Compare ativos da B3 ou ETFs/ações internacionais "
        "com análise de Força Relativa (RS) e retorno diário."
    )

    with st.sidebar:
        st.header("⚙️ Configurações")

        modo = st.radio(
            "Mercado",
            options=["B3", "ETF / Internacional"],
            horizontal=True,
        )

        # Reseta parâmetros ao trocar de mercado
        if modo != st.session_state.modo_anterior:
            st.session_state.params        = None
            st.session_state.modo_anterior = modo

        tickers_raw = st.text_input(
            "Tickers (separados por vírgula)",
            value=DEFAULTS[modo],
            help=(
                "B3: ex. PETR4, VALE3, BOVA11  (sufixo .SA adicionado automaticamente)\n"
                "ETF: ex. AAXJ, QQQ, SPY  (formato Yahoo Finance)"
            ),
        )

        periodo = st.selectbox("Período", list(PERIOD_MAP.keys()), index=3)
        filtro  = st.selectbox("Visualização", ["Base 100", "Percentual", "Preço"])

        st.divider()

        if modo == "B3":
            st.info(
                "💡 **Força Relativa:** O **último ticker** é o benchmark.\n\n"
                "📌 O sufixo **.SA** é adicionado automaticamente."
            )
        else:
            st.info(
                "💡 **Força Relativa:** O **último ticker** é o benchmark.\n\n"
                "📌 Use tickers no formato Yahoo Finance (ex: AAXJ, SPY, QQQ)."
            )

        if st.button("🔄 Atualizar", width='stretch', type="primary"):
            get_close.clear()
            st.session_state.params = (tickers_raw, periodo, filtro, modo)

    if st.session_state.params is None:
        st.session_state.params = (tickers_raw, periodo, filtro, modo)

    plotar(*st.session_state.params)


if __name__ == "__main__":
    main()
