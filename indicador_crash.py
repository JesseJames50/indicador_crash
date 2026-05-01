import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ─── FRED API KEY ───────────────────────────────────────────────────────────
FRED_API_KEY = "263fca8c17f12a1ec97764bd4a6fdc42"

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor de Risco Sistêmico",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── STYLING ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0c10;
    color: #e2e8f0;
}

.main { background-color: #0a0c10; }
.block-container { padding: 2rem 2rem 2rem 2rem; max-width: 1400px; }

/* Header */
.dashboard-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 2rem;
}
.dashboard-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.02em;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}
.dashboard-header p {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #111520;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.card-safe::before   { background: linear-gradient(90deg, #10b981, #34d399); }
.card-warn::before   { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.card-danger::before { background: linear-gradient(90deg, #ef4444, #f87171); }

.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.8rem;
}
.card-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.value-safe   { color: #34d399; }
.value-warn   { color: #fbbf24; }
.value-danger { color: #f87171; }

.card-status {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
}
.status-safe   { background: rgba(16,185,129,0.12); color: #34d399; }
.status-warn   { background: rgba(245,158,11,0.12); color: #fbbf24; }
.status-danger { background: rgba(239,68,68,0.12);  color: #f87171; }

.card-threshold {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    margin-top: 0.7rem;
}

/* Alert Banner */
.alert-banner {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-ok      { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); color: #34d399; }
.alert-warning { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.30); color: #fbbf24; }
.alert-critical{ background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.35); color: #f87171; }

/* Section titles */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2535;
}

/* Update info */
.update-info {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #334155;
    text-align: right;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e2535;
}

/* Plotly chart background override */
.js-plotly-plot .plotly { background: transparent !important; }

/* Divider */
hr { border-color: #1e2535; margin: 1.5rem 0; }

/* Streamlit overrides */
div[data-testid="stMetric"] { display: none; }
.stSpinner > div { border-color: #334155 !important; }
button[kind="primary"] {
    background: #1e2535 !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── THRESHOLDS ─────────────────────────────────────────────────────────────
THRESH = {
    "TNX":  {"warn": 4.5,  "danger": 5.0,  "unit": "%",  "label": "Rendimento T-10Y"},
    "KRE":  {"peak": None, "drop_warn": 15, "drop_danger": 30, "unit": "$", "label": "KRE — Bancos Regionais"},
    "SPREAD": {"warn": 4.5, "danger": 6.0,  "unit": "%",  "label": "Spread de Crédito HY"},
}

PERIOD_OPTIONS = {"1 mês": "1mo", "3 meses": "3mo", "6 meses": "6mo", "1 ano": "1y", "2 anos": "2y"}
CHART_TEMPLATE = "plotly_dark"
CHART_PAPER_BG = "#111520"
CHART_PLOT_BG  = "#0d1018"


# ─── DATA FETCHING ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_yf(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"].to_frame("value")
            else:
                df = df[["Close"]].rename(columns={"Close": "value"})
            df = df.dropna()
            if not df.empty:
                return df
    except Exception:
        pass
    # Fallback para Ticker.history() — necessário para índices como ^TNX
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        df = df[["Close"]].rename(columns={"Close": "value"}).dropna()
        return df
    except Exception:
        return pd.DataFrame()


# ── Credit spread source label (set by fetch_fred_spread) ──
_SPREAD_SOURCE: str = ""


@st.cache_data(ttl=300)
def fetch_fred_spread(period: str, fred_api_key: str = FRED_API_KEY) -> pd.DataFrame:
    """
    Fetch US HY credit spread with 4 fallback strategies:
      1. FRED CSV direct (browser User-Agent)
      2. FRED JSON API (requires free API key from fred.stlouisfed.org)
      3. pandas_datareader FRED
      4. yfinance proxy: HYG 30-day SEC yield minus ^TNX
    """
    from io import StringIO

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fred.stlouisfed.org/",
    }

    period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    days = period_map.get(period, 365)
    cutoff = datetime.now() - timedelta(days=days)

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df["value"] != "."]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df = df[df.index >= cutoff]
        return df

    # ── Strategy 1: FRED CSV com headers de navegador ──────────────────────
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        if "DATE" in resp.text[:200]:
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            df.columns = ["value"]
            df = _clean(df)
            if not df.empty:
                st.session_state["spread_source"] = "FRED (dados oficiais)"
                return df
    except Exception:
        pass

    # ── Strategy 2: FRED JSON API com chave do usuário ─────────────────────
    if fred_api_key and len(fred_api_key) == 32:
        try:
            obs_start = cutoff.strftime("%Y-%m-%d")
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id=BAMLH0A0HYM2&observation_start={obs_start}"
                f"&file_type=json&api_key={fred_api_key}"
            )
            resp = requests.get(url, headers=HEADERS, timeout=15)
            data = resp.json()
            obs = data.get("observations", [])
            if obs:
                df = pd.DataFrame(obs)[["date", "value"]]
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df = _clean(df)
                if not df.empty:
                    st.session_state["spread_source"] = "FRED API (chave fornecida)"
                    return df
        except Exception:
            pass

    # ── Strategy 3: pandas_datareader ──────────────────────────────────────
    try:
        import pandas_datareader.data as web
        df = web.get_data_fred("BAMLH0A0HYM2", start=cutoff, end=datetime.now())
        if df is not None and not df.empty:
            df.columns = ["value"]
            df = df.dropna()
            df = df[df.index >= cutoff]
            if not df.empty:
                st.session_state["spread_source"] = "FRED (pandas-datareader)"
                return df
    except Exception:
        pass

    # ── Strategy 4: yfinance proxy — HYG yield vs ^TNX ────────────────────
    # HYG = iShares HY Bond ETF. trailingAnnualDividendYield ≈ HY yield.
    # Spread proxy = HYG_yield - TNX_yield
    try:
        hyg_info  = yf.Ticker("HYG").fast_info
        tnx_info  = yf.Ticker("^TNX").fast_info
        hyg_price = hyg_info.last_price
        tnx_yield = tnx_info.last_price / 100  # ^TNX is quoted as percentage

        # HYG annual dividend ~ $4.80 on ~$77 ≈ 6.2% gross yield
        # We approximate current HY yield from price deviation
        hyg_hist = yf.download("HYG", period="2y", auto_adjust=True, progress=False)
        if isinstance(hyg_hist.columns, pd.MultiIndex):
            hyg_hist = hyg_hist["Close"]
            if isinstance(hyg_hist, pd.DataFrame):
                hyg_hist = hyg_hist.iloc[:, 0]
        else:
            hyg_hist = hyg_hist["Close"]

        # HYG yield estimate: fixed coupon stream / current price
        # HYG has ~6.5% coupon at par (~$100 face), trades ~$77 → yield ~8.4%
        HYG_COUPON_RATE = 0.065
        HYG_PAR         = 100.0
        hyg_yield_series = (HYG_COUPON_RATE * HYG_PAR) / hyg_hist.values
        spread_series    = hyg_yield_series - tnx_yield
        spread_series    = pd.Series(
            spread_series,
            index=hyg_hist.index,
            name="value",
        ).clip(lower=0)

        df = pd.DataFrame({"value": spread_series * 100})  # to percentage points
        df = df[df.index >= cutoff].dropna()
        if not df.empty:
            st.session_state["spread_source"] = "⚠ Proxy estimado (HYG/^TNX) — instale chave FRED para dados reais"
            return df
    except Exception:
        pass

    st.session_state["spread_source"] = "erro"
    return pd.DataFrame()


def get_current(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(df["value"].iloc[-1])


def get_kre_peak(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(df["value"].max())


# ─── STATUS HELPERS ──────────────────────────────────────────────────────────
def tnx_status(val):
    if val is None:
        return "neutral", "Sem dados"
    if val >= THRESH["TNX"]["danger"]:
        return "danger", f"⚠ ALERTA CRÍTICO — acima de {THRESH['TNX']['danger']}%"
    if val >= THRESH["TNX"]["warn"]:
        return "warn", f"⚡ ATENÇÃO — acima de {THRESH['TNX']['warn']}%"
    return "safe", "✓ Normal"


def kre_status(val, peak):
    if val is None or peak is None:
        return "neutral", "Sem dados"
    drop = ((peak - val) / peak) * 100
    if drop >= THRESH["KRE"]["drop_danger"]:
        return "danger", f"⚠ ALERTA CRÍTICO — queda de {drop:.1f}% do pico"
    if drop >= THRESH["KRE"]["drop_warn"]:
        return "warn", f"⚡ ATENÇÃO — queda de {drop:.1f}% do pico"
    return "safe", f"✓ Normal (−{drop:.1f}% do pico)"


def spread_status(val):
    if val is None:
        return "neutral", "Sem dados"
    if val >= THRESH["SPREAD"]["danger"]:
        return "danger", f"⚠ ALERTA CRÍTICO — acima de {THRESH['SPREAD']['danger']}%"
    if val >= THRESH["SPREAD"]["warn"]:
        return "warn", f"⚡ ATENÇÃO — acima de {THRESH['SPREAD']['warn']}%"
    return "safe", "✓ Normal"


def overall_alert(s1, s2, s3):
    statuses = [s1, s2, s3]
    n_danger = statuses.count("danger")
    n_warn   = statuses.count("warn")
    if n_danger >= 2:
        return "critical", "🚨 SINAL SISTÊMICO — 2 ou mais indicadores em nível crítico. Avalie sua exposição imediatamente."
    if n_danger == 1:
        return "warning", "⚡ 1 indicador em nível crítico. Monitore os demais com atenção redobrada."
    if n_warn >= 2:
        return "warning", "⚡ 2 ou mais indicadores em zona de atenção. Acompanhe a evolução diariamente."
    if n_warn == 1:
        return "warning", "📡 1 indicador em zona de atenção. Mantenha monitoramento ativo."
    return "ok", "✓ Todos os indicadores dentro dos parâmetros normais."


# ─── CHART BUILDER ───────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.07) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha) for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_chart(df: pd.DataFrame, title: str, unit: str,
                thresholds: list[tuple[float, str, str]],
                color: str = "#60a5fa") -> go.Figure:
    fig = go.Figure()

    if not df.empty:
        fill_color = hex_to_rgba(color, 0.07) if color.startswith("#") else color
        # Area fill
        fig.add_trace(go.Scatter(
            x=df.index, y=df["value"],
            mode="lines",
            line=dict(color=color, width=1.8),
            fill="tozeroy",
            fillcolor=fill_color,
            name=title,
            hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>%{{y:.3f}}{unit}</b><extra></extra>",
        ))

        # Threshold lines
        for thresh_val, thresh_label, thresh_color in thresholds:
            fig.add_hline(
                y=thresh_val,
                line=dict(color=thresh_color, width=1.2, dash="dash"),
                annotation_text=thresh_label,
                annotation_position="top right",
                annotation=dict(
                    font=dict(size=10, color=thresh_color, family="Space Mono"),
                    bgcolor="#0a0c10",
                ),
            )

        # Last value annotation
        last_val = df["value"].iloc[-1]
        last_date = df.index[-1]
        fig.add_annotation(
            x=last_date, y=last_val,
            text=f" {last_val:.2f}{unit}",
            showarrow=False,
            font=dict(color=color, size=12, family="Space Mono"),
            xanchor="left",
        )

    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_PLOT_BG,
        height=260,
        margin=dict(l=10, r=20, t=20, b=10),
        showlegend=False,
        xaxis=dict(
            gridcolor="#1a2030",
            showgrid=True,
            tickfont=dict(family="Space Mono", size=9, color="#475569"),
            tickformat="%b/%y",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#1a2030",
            showgrid=True,
            tickfont=dict(family="Space Mono", size=9, color="#475569"),
            ticksuffix=unit,
            zeroline=False,
            side="right",
        ),
        hoverlabel=dict(
            bgcolor="#111520",
            bordercolor="#334155",
            font=dict(family="Space Mono", size=11),
        ),
    )
    return fig


def build_kre_chart(df: pd.DataFrame, peak: float | None, period: str) -> go.Figure:
    fig = build_chart(
        df, "KRE", "$",
        thresholds=[
            (peak * 0.85, "−15% do pico", "#f59e0b") if peak else (0, "", "#f59e0b"),
            (peak * 0.70, "−30% ALERTA", "#ef4444") if peak else (0, "", "#ef4444"),
        ],
        color="#818cf8",
    )
    if peak and not df.empty:
        fig.add_hline(
            y=peak,
            line=dict(color="#334155", width=1, dash="dot"),
            annotation_text=f"Pico: ${peak:.2f}",
            annotation_position="top left",
            annotation=dict(font=dict(size=9, color="#64748b", family="Space Mono"), bgcolor="#0a0c10"),
        )
    return fig


# ─── METRIC CARD ─────────────────────────────────────────────────────────────
def metric_card(label, value_str, status_level, status_text, threshold_text):
    card_cls = f"metric-card card-{status_level}"
    val_cls  = f"value-{status_level}"
    st_cls   = f"card-status status-{status_level}"
    dot = {"safe": "●", "warn": "◆", "danger": "▲"}.get(status_level, "●")
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="card-label">{label}</div>
        <div class="card-value {val_cls}">{value_str}</div>
        <div class="{st_cls}">{dot} {status_text}</div>
        <div class="card-threshold">{threshold_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN APP ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>Monitor de Risco Sistêmico 🇺🇸</h1>
        <p>Três indicadores de alerta precoce para instabilidade financeira americana</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configurações")
        st.markdown("""
**Spread de Crédito HY**

Os dados vêm do FRED (Federal Reserve).
Se houver bloqueio, insira sua chave gratuita:

👉 [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
        """)
        fred_key = st.text_input(
            "Chave FRED API (opcional)",
            type="password",
            placeholder="32 caracteres alfanuméricos",
            help="Gratuita em fred.stlouisfed.org. Necessária se o FRED bloquear acesso direto.",
        )
        st.markdown("---")
        st.markdown("""
**Limiares de alerta**

| Indicador | Atenção | Crítico |
|-----------|---------|---------|
| T-Note 10Y | > 4,5% | > 5,0% |
| KRE queda | > 15% | > 30% |
| Spread HY | > 4,5% | > 6,0% |
        """)

    # Controls row
    col_period, col_refresh, col_spacer = st.columns([2, 1, 5])
    with col_period:
        period_label = st.selectbox(
            "Período do gráfico",
            list(PERIOD_OPTIONS.keys()),
            index=3,
            label_visibility="collapsed",
        )
        period = PERIOD_OPTIONS[period_label]
    with col_refresh:
        refresh = st.button("⟳ Atualizar", width='stretch')

    if refresh:
        st.cache_data.clear()
        st.session_state.pop("spread_source", None)

    # Initialize spread source tracking
    if "spread_source" not in st.session_state:
        st.session_state["spread_source"] = ""

    # Fetch data
    with st.spinner("Carregando dados..."):
        df_tnx    = fetch_yf("^TNX", period)
        df_kre    = fetch_yf("KRE", period)
        df_spread = fetch_fred_spread(period, fred_api_key=fred_key or FRED_API_KEY)

    tnx_val    = get_current(df_tnx)
    kre_val    = get_current(df_kre)
    kre_peak   = get_kre_peak(fetch_yf("KRE", "2y"))  # peak always from 2y
    spread_val = get_current(df_spread)

    s1, msg1 = tnx_status(tnx_val)
    s2, msg2 = kre_status(kre_val, kre_peak)
    s3, msg3 = spread_status(spread_val)

    # Overall alert banner
    alert_level, alert_msg = overall_alert(s1, s2, s3)
    alert_cls = {"ok": "alert-ok", "warning": "alert-warning", "critical": "alert-critical"}[alert_level]
    st.markdown(f'<div class="alert-banner {alert_cls}">{alert_msg}</div>', unsafe_allow_html=True)

    # ── KRE drop calc ──
    kre_drop = ((kre_peak - kre_val) / kre_peak * 100) if (kre_val and kre_peak) else 0

    # ── Metric cards ──
    st.markdown('<div class="section-title">Valores atuais</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "① Rendimento T-Note 10 anos",
            f"{tnx_val:.2f}%" if tnx_val else "N/D",
            s1, msg1,
            f"Atenção: 4,5% | Crítico: 5,0% | Atual: {tnx_val:.3f}%" if tnx_val else "—",
        )
    with c2:
        metric_card(
            "② KRE — Bancos Regionais EUA",
            f"${kre_val:.2f}" if kre_val else "N/D",
            s2, msg2,
            f"Pico 2 anos: ${kre_peak:.2f} | Queda atual: {kre_drop:.1f}% | Crítico: −30%" if kre_val else "—",
        )
    with c3:
        metric_card(
            "③ Spread de Crédito HY (OAS)",
            f"{spread_val:.2f}%" if spread_val else "N/D",
            s3, msg3,
            f"Atenção: 4,5% | Crítico: 6,0% | Atual: {spread_val:.3f}%" if spread_val else "—",
        )
        src = st.session_state.get("spread_source", "")
        if src:
            color = "#64748b" if "FRED" in src else "#f59e0b"
            icon  = "📡" if "Proxy" in src else "✓"
            st.markdown(
                f'<div style="font-family:Space Mono,monospace;font-size:0.62rem;'
                f'color:{color};margin-top:6px;">{icon} Fonte: {src}</div>',
                unsafe_allow_html=True,
            )
        if not spread_val:
            st.markdown(
                '<div style="font-family:Space Mono,monospace;font-size:0.62rem;'
                'color:#f59e0b;margin-top:6px;">💡 Adicione chave FRED na barra lateral</div>',
                unsafe_allow_html=True,
            )

    # ── Charts ──
    st.markdown('<div class="section-title">Histórico — ' + period_label + '</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("**T-Note 10 anos** `^TNX`", help="Rendimento do título do Tesouro americano de 10 anos. Alerta quando > 5% por 3 dias consecutivos.")
        fig_tnx = build_chart(
            df_tnx, "T-Note 10Y", "%",
            thresholds=[
                (4.5, "Atenção 4,5%", "#f59e0b"),
                (5.0, "Crítico 5,0%", "#ef4444"),
            ],
            color="#38bdf8",
        )
        st.plotly_chart(fig_tnx, width='stretch', config={"displayModeBar": False})

    with g2:
        st.markdown("**Bancos Regionais** `KRE`", help="ETF de bancos regionais americanos. Alerta quando cai 30% do pico recente.")
        fig_kre = build_kre_chart(df_kre, kre_peak, period)
        st.plotly_chart(fig_kre, width='stretch', config={"displayModeBar": False})

    with g3:
        st.markdown("**Spread HY** `BAMLH0A0HYM2`", help="Diferencial de juros entre títulos de alto risco e títulos seguros. Alerta quando > 6%.")
        fig_spread = build_chart(
            df_spread, "HY Spread", "%",
            thresholds=[
                (4.5, "Atenção 4,5%", "#f59e0b"),
                (6.0, "Crítico 6,0%", "#ef4444"),
            ],
            color="#a78bfa",
        )
        st.plotly_chart(fig_spread, width='stretch', config={"displayModeBar": False})

    # ── Reference table ──
    st.markdown('<div class="section-title">Guia de referência rápida</div>', unsafe_allow_html=True)

    ref_data = {
        "Indicador": [
            "① T-Note 10 anos",
            "② KRE (bancos regionais)",
            "③ Spread HY (crédito)",
        ],
        "Zona Normal": ["< 4,5%", "Queda < 15% do pico", "< 4,5%"],
        "Zona de Atenção ⚡": ["4,5% – 5,0%", "Queda 15–30% do pico", "4,5% – 6,0%"],
        "Zona Crítica ⚠": ["> 5,0% por 3 dias", "Queda > 30% do pico", "> 6,0%"],
        "O que sinaliza": [
            "Desconfiança na dívida americana / pressão sobre bancos",
            "Estresse nos bancos regionais / risco de corrida bancária",
            "Crédito corporativo sob pressão / risco de inadimplência em cascata",
        ],
    }

    df_ref = pd.DataFrame(ref_data)
    st.dataframe(
        df_ref,
        width='stretch',
        hide_index=True,
        column_config={
            "Indicador":          st.column_config.TextColumn(width="medium"),
            "Zona Normal":        st.column_config.TextColumn(width="small"),
            "Zona de Atenção ⚡": st.column_config.TextColumn(width="small"),
            "Zona Crítica ⚠":    st.column_config.TextColumn(width="small"),
            "O que sinaliza":     st.column_config.TextColumn(width="large"),
        },
    )

    # ── Correlação: alerta combinado ──
    st.markdown('<div class="section-title">Lógica de alerta sistêmico</div>', unsafe_allow_html=True)
    triggers_active = sum([s1 == "danger", s2 == "danger", s3 == "danger"])
    warns_active    = sum([s1 == "warn",   s2 == "warn",   s3 == "warn"])

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
            <div class="card-label">Alertas críticos ativos</div>
            <div class="card-value {'value-danger' if triggers_active >= 2 else 'value-warn' if triggers_active == 1 else 'value-safe'}">
                {triggers_active}/3
            </div>
            <div class="card-label" style="margin-top:0.5rem;">Zonas de atenção ativas</div>
            <div class="card-value value-warn" style="font-size:1.8rem;">{warns_active}/3</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="metric-card">
            <div class="card-label">Como interpretar a combinação dos sinais</div>
            <div style="font-size:0.85rem; color:#94a3b8; line-height:1.7; font-family:'Syne',sans-serif;">
                <b style="color:#34d399;">0 alertas:</b> Sistema operando normalmente. Revisão semanal suficiente.<br>
                <b style="color:#fbbf24;">1 atenção:</b> Monitoramento diário recomendado. Revise sua liquidez.<br>
                <b style="color:#fbbf24;">2+ atenções:</b> Avalie se sua reserva de emergência está adequada.<br>
                <b style="color:#f87171;">1 crítico:</b> Verifique os outros dois indicadores imediatamente.<br>
                <b style="color:#f87171;">2+ críticos:</b> Sinal sistêmico histórico. Considere consultar um assessor financeiro.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    now = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
    st.markdown(f"""
    <div class="update-info">
        Dados atualizados em: {now} &nbsp;·&nbsp;
        Fontes: Yahoo Finance (^TNX, KRE) · FRED / ICE BofA (BAMLH0A0HYM2) &nbsp;·&nbsp;
        Cache: 5 minutos &nbsp;·&nbsp;
        <i>Não constitui recomendação de investimento.</i>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh a cada 5 minutos
    time.sleep(0)
    st.markdown("""
    <script>
    setTimeout(function(){ window.location.reload(); }, 300000);
    </script>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
