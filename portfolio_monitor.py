"""
portfolio_monitor.py — Monitor de Risco Sistêmico + Rebalanceamento de Carteira
Score v7 integrado com alocação dinâmica: mostra o que comprar/vender conforme o regime atual.
Atualização automática a cada 6h (alinhada ao horário de mercado americano).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, urllib3
from datetime import datetime, timedelta
from fredapi import Fred

from data_loader import merge_data
from config import FRED_API_KEY, START_DATE

urllib3.disable_warnings()

# ─── Parâmetros score v7 ──────────────────────────────────────────────────────
ROLL_WINDOW       = 252
EMA_SPAN          = 21
PERSIST_DAYS      = 3
VEL_PCT           = 0.90
VEL_MIN_ABS       = 0.04
DEADZONE          = 0.25
DEADZONE_PER_COMP = {"move_z": 0.35, "ig_z": 0.35, "move_vix_z": 0.30}
WARN_PCT          = 0.70
CRIT_PCT          = 0.92
CALIB_START       = "2018-01-01"
CALIB_END         = "2024-12-31"

V7_WEIGHTS = {
    "hy_z": 0.21, "ig_z": 0.07, "move_z": 0.15, "move_vix_z": 0.05,
    "tbill_z": 0.15, "kre_z": 0.17, "curve_z": 0.12,
    "vix_z": 0.05, "t10y_z": 0.02, "funding_z": 0.01,
}

# ─── Carteira e alocações alvo por regime ─────────────────────────────────────
ETF_META = {
    "SPY":  {"label": "S&P 500",         "bloco": "Crescimento Core",   "cor": "#3b82f6"},
    "QQQ":  {"label": "Nasdaq-100",       "bloco": "Crescimento Core",   "cor": "#60a5fa"},
    "VTV":  {"label": "Vanguard Value",   "bloco": "Crescimento Core",   "cor": "#93c5fd"},
    "XLP":  {"label": "Consumer Staples", "bloco": "Setor Defensivo",    "cor": "#10b981"},
    "XLV":  {"label": "Healthcare",       "bloco": "Setor Defensivo",    "cor": "#34d399"},
    "XLU":  {"label": "Utilities",        "bloco": "Setor Defensivo",    "cor": "#6ee7b7"},
    "IEF":  {"label": "Treasuries 7-10a", "bloco": "Renda Fixa",         "cor": "#f59e0b"},
    "SCHP": {"label": "TIPS",             "bloco": "Renda Fixa",         "cor": "#fbbf24"},
    "GLD":  {"label": "Ouro",             "bloco": "Proteção Sistêmica", "cor": "#ef4444"},
    "BIL":  {"label": "T-Bills 1-3m",    "bloco": "Proteção Sistêmica", "cor": "#f87171"},
    "PDBC": {"label": "Commodities",      "bloco": "Proteção Sistêmica", "cor": "#fca5a5"},
}

TARGET_BY_REGIME = {
    "normal": {
        "SPY": 0.25, "QQQ": 0.15, "VTV": 0.10,
        "XLP": 0.06, "XLV": 0.05, "XLU": 0.04,
        "IEF": 0.03, "SCHP": 0.02,
        "GLD": 0.17, "BIL": 0.06, "PDBC": 0.07,
    },
    "atencao": {
        "SPY": 0.20, "QQQ": 0.12, "VTV": 0.08,
        "XLP": 0.08, "XLV": 0.07, "XLU": 0.05,
        "IEF": 0.10, "SCHP": 0.08,
        "GLD": 0.12, "BIL": 0.07, "PDBC": 0.03,
    },
    "aceleracao": {
        "SPY": 0.15, "QQQ": 0.08, "VTV": 0.07,
        "XLP": 0.10, "XLV": 0.08, "XLU": 0.06,
        "IEF": 0.12, "SCHP": 0.08,
        "GLD": 0.15, "BIL": 0.08, "PDBC": 0.03,
    },
    "critico": {
        "SPY": 0.10, "QQQ": 0.05, "VTV": 0.05,
        "XLP": 0.12, "XLV": 0.08, "XLU": 0.06,
        "IEF": 0.15, "SCHP": 0.09,
        "GLD": 0.20, "BIL": 0.07, "PDBC": 0.03,
    },
}

REGIME_CFG = {
    "normal": {
        "emoji": "🟢", "titulo": "NORMAL",
        "cor": "#16a34a", "bg": "#f0fdf4", "borda": "#86efac",
        "desc": (
            "Score abaixo do threshold de atenção (P70). Mercado em regime tranquilo. "
            "Estratégia: maximizar crescimento, reduzir renda fixa e manter proteção em ouro."
        ),
    },
    "atencao": {
        "emoji": "🟡", "titulo": "ATENÇÃO",
        "cor": "#d97706", "bg": "#fffbeb", "borda": "#fcd34d",
        "desc": (
            "Score acima do P70 por 3+ dias consecutivos. Sinais de deterioração incipiente. "
            "Estratégia: posição balanceada — aumentar defensivos e renda fixa gradualmente."
        ),
    },
    "aceleracao": {
        "emoji": "⚡", "titulo": "ACELERAÇÃO",
        "cor": "#ea580c", "bg": "#fff7ed", "borda": "#fed7aa",
        "desc": (
            "Score acelerando rapidamente (P90 de velocidade). Deterioração em curso antes do threshold crítico. "
            "Estratégia: rotacionar para proteção — reduzir crescimento, aumentar ouro e defensivos."
        ),
    },
    "critico": {
        "emoji": "🔴", "titulo": "CRÍTICO",
        "cor": "#dc2626", "bg": "#fef2f2", "borda": "#fca5a5",
        "desc": (
            "Score acima do P92 por 3+ dias. Historicamente associado a eventos de crise sistêmica. "
            "Estratégia: máxima proteção — mínimo em growth, máximo em ouro, BIL e defensivos."
        ),
    },
}

# ─── Funções auxiliares de score ──────────────────────────────────────────────
def _zscore(s: pd.Series, w: int = ROLL_WINDOW) -> pd.Series:
    r = s.rolling(w, min_periods=w)
    return (s - r.mean()) / r.std().where(lambda x: x > 0)


def _kre_stress(kre: pd.Series, w: int = ROLL_WINDOW) -> pd.Series:
    return -(kre - kre.rolling(w, min_periods=w).max()) / kre.rolling(w, min_periods=w).max() * 100


def _persist(above: pd.Series, days: int = PERSIST_DAYS) -> pd.Series:
    return above.fillna(False).astype(int).rolling(days, min_periods=days).sum() >= days


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns and df[col].notna().any():
        return df[col]
    return pd.Series(dtype=float, index=df.index)


def _build_score(ind: pd.DataFrame) -> pd.Series:
    score, tw = pd.Series(0.0, index=ind.index), 0.0
    for k, w in V7_WEIGHTS.items():
        if k not in ind.columns:
            continue
        s = ind[k]
        if s.notna().any():
            dz = DEADZONE_PER_COMP.get(k, DEADZONE)
            score += w * (s - dz).clip(lower=0).fillna(0)
            tw    += w
    return score / tw if tw > 0 else score


def _detect_qe(walcl: pd.Series) -> list:
    if walcl.empty:
        return []
    g = walcl.pct_change(90).dropna()
    periods, in_qe, t0 = [], False, None
    for d, v in g.items():
        if v > 0.05 and not in_qe:
            t0, in_qe = d, True
        elif v <= 0.05 and in_qe:
            periods.append((t0, d)); in_qe = False
    if in_qe:
        periods.append((t0, g.index[-1]))
    return periods


def _dl_yahoo(ticker: str, start: datetime, end: datetime) -> pd.Series:
    """Download via Yahoo Finance API direto (contorna SSL corporativo)."""
    p1 = int(start.timestamp())
    p2 = int(end.timestamp())
    yf_t = ticker.replace("^", "%5E")
    url  = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_t}"
            f"?interval=1d&period1={p1}&period2={p2}&events=div,splits")
    try:
        r = requests.get(url, verify=False, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        d = r.json()["chart"]["result"][0]
        ts = pd.to_datetime(d["timestamp"], unit="s", utc=True)
        ts = ts.tz_convert("America/New_York").normalize().tz_localize(None)
        ind = d["indicators"]
        vals = (ind["adjclose"][0]["adjclose"] if "adjclose" in ind and ind["adjclose"]
                else ind["quote"][0]["close"])
        return pd.Series(vals, index=ts, name=ticker, dtype=float).dropna()
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def _dl_yahoo_latest(ticker: str) -> float:
    """Retorna o preço mais recente de um ticker (último fechamento disponível)."""
    end   = datetime.now()
    start = end - timedelta(days=10)
    s = _dl_yahoo(ticker, start, end)
    return float(s.dropna().iloc[-1]) if s.notna().any() else float("nan")


# ─── Carregamento do score v7 (TTL 6h) ───────────────────────────────────────
@st.cache_data(ttl=21600)
def load_score():
    df   = merge_data()
    fred = Fred(api_key=FRED_API_KEY)

    for sid, col in [("DTB3","dtb3"), ("T10Y2Y","t10y2y"), ("BAMLC0A0CM","ig_oas")]:
        try:
            df = df.join(fred.get_series(sid, observation_start=START_DATE).rename(col), how="left")
        except Exception:
            df[col] = float("nan")

    qe_periods = []
    try:
        walcl = fred.get_series("WALCL", observation_start=START_DATE).resample("D").ffill()
        qe_periods = _detect_qe(walcl)
    except Exception:
        pass

    # MOVE
    end_dt  = datetime.now()
    start_dt= end_dt - timedelta(days=4000)
    move_raw = _dl_yahoo("^MOVE", start_dt, end_dt)
    if move_raw.notna().sum() >= ROLL_WINDOW:
        df = df.join(move_raw.rename("move_raw"), how="left")
    else:
        df["move_raw"] = _safe(df,"t10y").diff().rolling(21, min_periods=15).std() * (252**0.5)

    df = df.ffill()

    ind = pd.DataFrame(index=df.index)
    ind["t10y_z"]     = _zscore(_safe(df,"t10y"))
    ind["kre_z"]      = _zscore(_kre_stress(_safe(df,"kre")))
    ind["hy_z"]       = _zscore(_safe(df,"hy_spread"))
    ind["vix_z"]      = _zscore(_safe(df,"vix"))
    ind["tbill_z"]    = _zscore(_safe(df,"fed_funds") - _safe(df,"dtb3"))
    ind["curve_z"]    = _zscore(-_safe(df,"t10y2y"))
    ind["funding_z"]  = _zscore(_safe(df,"sofr") - _safe(df,"fed_funds"))
    ind["ig_z"]       = _zscore(_safe(df,"ig_oas"))
    ind["move_z"]     = _zscore(_safe(df,"move_raw"))
    ind["move_vix_z"] = (ind["move_z"] - ind["vix_z"]).clip(lower=0)

    ind["score_raw"] = _build_score(ind)
    ind["score"]     = ind["score_raw"].ewm(span=EMA_SPAN, adjust=False).mean()

    calib = ind["score"].loc[pd.Timestamp(CALIB_START):pd.Timestamp(CALIB_END)].dropna()
    warn  = float(calib.quantile(WARN_PCT)) if len(calib) >= ROLL_WINDOW else 0.20
    crit  = float(calib.quantile(CRIT_PCT)) if len(calib) >= ROLL_WINDOW else 0.45

    ind["alert_warn"] = _persist(ind["score"] >= warn)
    ind["alert_crit"] = _persist(ind["score"] >= crit)

    ind["velocity"]   = ind["score"].diff(5)
    vel_thr = ind["velocity"].rolling(756, min_periods=504).quantile(VEL_PCT)
    ind["alert_vel"]  = (
        (ind["velocity"] >= vel_thr) &
        (ind["velocity"] >= VEL_MIN_ABS) &
        (ind["score"]    >= warn * 0.7)
    )

    loaded_at = datetime.now()
    return ind, warn, crit, qe_periods, loaded_at


# ─── Preços atuais dos ETFs (TTL 1h) ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_prices():
    end   = datetime.now()
    start = end - timedelta(days=5)
    prices = {}
    for t in list(ETF_META.keys()) + ["USDBRL=X"]:
        s = _dl_yahoo(t, start, end)
        prices[t] = float(s.dropna().iloc[-1]) if s.notna().any() else float("nan")
    return prices, datetime.now()


# ─── Horário de mercado ───────────────────────────────────────────────────────
def _market_info():
    from datetime import timezone
    # Estimativa simples: ET = UTC-4 (horário de verão) ou UTC-5
    utc_now  = datetime.utcnow()
    et_now   = utc_now - timedelta(hours=4)   # ajuste fixo EST/EDT ≈ UTC-4
    brt_now  = utc_now - timedelta(hours=3)
    is_open  = (
        et_now.weekday() < 5 and
        (et_now.hour > 9 or (et_now.hour == 9 and et_now.minute >= 30)) and
        et_now.hour < 16
    )
    return is_open, et_now, brt_now


# ─── Layout ───────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Monitor de Risco + Carteira")
st.title("📊 Monitor de Risco Sistêmico — Rebalanceamento de Carteira")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Configuração")
    capital_total = st.number_input(
        "Capital total (R$)", min_value=1000, value=300_000, step=5000,
        format="%d", help="Valor total investido na carteira")

    st.markdown("---")
    st.subheader("📋 Posições atuais (R$)")
    st.caption("Informe o valor atual de cada ETF ou deixe em branco para usar a alocação base (Atenção).")

    base_alloc = TARGET_BY_REGIME["atencao"]
    atual_vals = {}
    for t, meta in ETF_META.items():
        default_val = int(capital_total * base_alloc[t])
        atual_vals[t] = st.number_input(
            f"{t} — {meta['label']}", min_value=0,
            value=default_val, step=100, format="%d", key=f"pos_{t}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Score", use_container_width=True, help="Recarregar score v7"):
            load_score.clear()
            st.rerun()
    with col_b:
        if st.button("💱 Preços", use_container_width=True, help="Recarregar preços ETFs"):
            load_prices.clear()
            st.rerun()

# Carrega dados
ind, warn_thresh, crit_thresh, qe_periods, loaded_at = load_score()
prices, prices_at = load_prices()

# Estado atual
score_latest = float(ind["score"].dropna().iloc[-1])
vel_latest   = float(ind["velocity"].dropna().iloc[-1])
in_crit      = bool(ind["alert_crit"].iloc[-1])
in_warn      = bool(ind["alert_warn"].iloc[-1])
in_vel       = bool(ind["alert_vel"].iloc[-1])

if in_crit:
    regime = "critico"
elif in_vel:
    regime = "aceleracao"
elif in_warn:
    regime = "atencao"
else:
    regime = "normal"

cfg = REGIME_CFG[regime]
is_open, et_now, brt_now = _market_info()

# ── Cabeçalho de status ───────────────────────────────────────────────────────
mkt_badge = (
    "🟢 **Mercado aberto**" if is_open
    else "🔴 **Mercado fechado**"
)
usdbrl = prices.get("USDBRL=X", float("nan"))
st.caption(
    f"Score atualizado: **{loaded_at.strftime('%d/%m/%Y %H:%M')}** · "
    f"Preços: **{prices_at.strftime('%H:%M')}** · "
    f"USD/BRL: **{usdbrl:.2f}** · "
    f"{mkt_badge} (ET: {et_now.strftime('%H:%M')} · BRT: {brt_now.strftime('%H:%M')})"
)

# ── Banner de risco ───────────────────────────────────────────────────────────
st.markdown(
    f"""<div style="
        background:{cfg['bg']};
        border: 2px solid {cfg['borda']};
        border-radius: 12px;
        padding: 20px 28px;
        margin: 12px 0 20px 0;
    ">
    <div style="display:flex; align-items:center; gap:16px">
        <span style="font-size:2.8rem; line-height:1">{cfg['emoji']}</span>
        <div>
            <div style="font-size:1.5rem; font-weight:700; color:{cfg['cor']}">
                REGIME ATUAL: {cfg['titulo']}
            </div>
            <div style="color:#374151; margin-top:4px; font-size:0.95rem">
                {cfg['desc']}
            </div>
        </div>
    </div>
    </div>""",
    unsafe_allow_html=True
)

# ── Métricas do score ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Score v7 EMA-21", f"{score_latest:.3f}",
          f"{cfg['emoji']} {cfg['titulo']}")
c2.metric("Threshold Atenção", f"{warn_thresh:.3f}",
          f"P{int(WARN_PCT*100)} cal. 2018-2024")
c3.metric("Threshold Crítico", f"{crit_thresh:.3f}",
          f"P{int(CRIT_PCT*100)} cal. 2018-2024")
c4.metric("Velocidade (5d)", f"{vel_latest:+.3f}",
          "⚡ Alerta ativo" if in_vel else "Normal")
c5.metric("USD/BRL", f"R$ {usdbrl:.2f}" if not np.isnan(usdbrl) else "—",
          "Câmbio atual")

st.markdown("---")

# ── Tabela de rebalanceamento ─────────────────────────────────────────────────
st.subheader(f"Tabela de Rebalanceamento — Regime {cfg['emoji']} {cfg['titulo']}")
st.caption(
    "Posição atual (sidebar) vs alocação alvo para o regime de risco vigente. "
    "Ação indica o movimento necessário para rebalancear."
)

target = TARGET_BY_REGIME[regime]
capital_atual = sum(atual_vals.values())

rows = []
for t, meta in ETF_META.items():
    val_atual   = atual_vals[t]
    pct_atual   = val_atual / capital_total if capital_total > 0 else 0
    pct_alvo    = target[t]
    val_alvo    = capital_total * pct_alvo
    diferenca   = val_alvo - val_atual
    price_usd   = prices.get(t, float("nan"))
    val_atual_usd = val_atual / usdbrl if not np.isnan(usdbrl) and usdbrl > 0 else float("nan")
    val_alvo_usd  = val_alvo  / usdbrl if not np.isnan(usdbrl) and usdbrl > 0 else float("nan")

    if abs(diferenca) < capital_total * 0.005:   # diferença < 0.5% → manter
        acao = "✅ Manter"
    elif diferenca > 0:
        acao = f"🟢 Comprar"
    else:
        acao = f"🔴 Vender"

    rows.append({
        "ETF":           t,
        "Ativo":         meta["label"],
        "Bloco":         meta["bloco"],
        "Atual %":       pct_atual,
        "Alvo %":        pct_alvo,
        "Δ pp":          pct_alvo - pct_atual,
        "R$ Atual":      val_atual,
        "R$ Alvo":       val_alvo,
        "R$ Mover":      diferenca,
        "USD Atual":     val_atual_usd,
        "USD Alvo":      val_alvo_usd,
        "Preço USD":     price_usd,
        "Ação":          acao,
    })

df_rebal = pd.DataFrame(rows)

# Formata para exibição
df_show = pd.DataFrame({
    "ETF":       df_rebal["ETF"],
    "Ativo":     df_rebal["Ativo"],
    "Bloco":     df_rebal["Bloco"],
    "Atual %":   df_rebal["Atual %"].apply(lambda x: f"{x*100:.1f}%"),
    "Alvo %":    df_rebal["Alvo %"].apply(lambda x: f"{x*100:.1f}%"),
    "Δ pp":      df_rebal["Δ pp"].apply(lambda x: f"{x*100:+.1f}pp"),
    "R$ Atual":  df_rebal["R$ Atual"].apply(lambda x: f"R$ {x:,.0f}"),
    "R$ Alvo":   df_rebal["R$ Alvo"].apply(lambda x: f"R$ {x:,.0f}"),
    "R$ Mover":  df_rebal["R$ Mover"].apply(lambda x: f"R$ {x:+,.0f}"),
    "Preço USD": df_rebal["Preço USD"].apply(
                    lambda x: f"${x:.2f}" if not np.isnan(x) else "—"),
    "Ação":      df_rebal["Ação"],
})

st.dataframe(
    df_show, width="stretch", hide_index=True,
    column_config={
        "ETF":       st.column_config.TextColumn(width="small"),
        "Ativo":     st.column_config.TextColumn(width="medium"),
        "Bloco":     st.column_config.TextColumn(width="medium"),
        "Atual %":   st.column_config.TextColumn(width="small"),
        "Alvo %":    st.column_config.TextColumn(width="small"),
        "Δ pp":      st.column_config.TextColumn(width="small"),
        "R$ Atual":  st.column_config.TextColumn(width="medium"),
        "R$ Alvo":   st.column_config.TextColumn(width="medium"),
        "R$ Mover":  st.column_config.TextColumn(width="medium"),
        "Preço USD": st.column_config.TextColumn(width="small"),
        "Ação":      st.column_config.TextColumn(width="medium"),
    }
)

# Resumo de movimentações
to_buy  = df_rebal[df_rebal["R$ Mover"] >  capital_total * 0.005]
to_sell = df_rebal[df_rebal["R$ Mover"] < -capital_total * 0.005]
total_move = df_rebal["R$ Mover"].abs().sum() / 2  # divide por 2 pois compra = venda

col_s, col_b2, col_t = st.columns(3)
col_s.metric("Vender (total)", f"R$ {to_sell['R$ Mover'].abs().sum():,.0f}",
             f"{len(to_sell)} ETF(s)")
col_b2.metric("Comprar (total)", f"R$ {to_buy['R$ Mover'].sum():,.0f}",
              f"{len(to_buy)} ETF(s)")
col_t.metric("Capital a realocar", f"R$ {total_move:,.0f}",
             f"{total_move/capital_total*100:.1f}% do portfólio")

st.markdown("---")

# ── Gráfico comparativo: atual vs alvo ────────────────────────────────────────
st.subheader("Composição: Atual vs Alvo")

tickers_list = list(ETF_META.keys())
pct_atual_list = [df_rebal.loc[df_rebal["ETF"]==t, "Atual %"].iloc[0]*100 for t in tickers_list]
pct_alvo_list  = [target[t]*100 for t in tickers_list]
colors = [ETF_META[t]["cor"] for t in tickers_list]

fig = go.Figure()
fig.add_bar(
    name="Alocação Atual", x=tickers_list, y=pct_atual_list,
    marker_color="rgba(100,116,139,0.55)",
    text=[f"{v:.0f}%" for v in pct_atual_list],
    textposition="outside", textfont=dict(size=10))
fig.add_bar(
    name=f"Alvo ({cfg['emoji']} {cfg['titulo']})", x=tickers_list, y=pct_alvo_list,
    marker_color=colors,
    text=[f"{v:.0f}%" for v in pct_alvo_list],
    textposition="outside", textfont=dict(size=10))

fig.update_layout(
    barmode="group", height=360,
    margin=dict(l=8, r=8, t=36, b=8),
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="% do portfólio", ticksuffix="%"),
)
st.plotly_chart(fig, width="stretch")

# ── Comparação por regime: visão completa ────────────────────────────────────
with st.expander("📋 Tabela de alocações alvo por regime — visão completa"):
    rows_all = []
    for t, meta in ETF_META.items():
        rows_all.append({
            "ETF":      t,
            "Ativo":    meta["label"],
            "Bloco":    meta["bloco"],
            "🟢 Normal":      f"{TARGET_BY_REGIME['normal'][t]*100:.0f}%",
            "🟡 Atenção":     f"{TARGET_BY_REGIME['atencao'][t]*100:.0f}%",
            "⚡ Aceleração":  f"{TARGET_BY_REGIME['aceleracao'][t]*100:.0f}%",
            "🔴 Crítico":     f"{TARGET_BY_REGIME['critico'][t]*100:.0f}%",
        })
    st.dataframe(pd.DataFrame(rows_all), hide_index=True, width="stretch")
    st.caption(
        "Lógica de transição: cada regime reduz growth e aumenta proteção progressivamente. "
        "A detecção do regime é automática — score v7 atualizado a cada 6h."
    )

# ── Legenda de risco ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗺️ Legenda — Níveis de Risco Sistêmico")

leg_cols = st.columns(4)
for col, (r_key, r_cfg) in zip(leg_cols, REGIME_CFG.items()):
    alvo = TARGET_BY_REGIME[r_key]
    core_pct      = sum(alvo[t] for t in ["SPY","QQQ","VTV"]) * 100
    def_pct       = sum(alvo[t] for t in ["XLP","XLV","XLU"]) * 100
    fi_pct        = sum(alvo[t] for t in ["IEF","SCHP"]) * 100
    prot_pct      = sum(alvo[t] for t in ["GLD","BIL","PDBC"]) * 100
    is_active     = (r_key == regime)
    bdr = f"3px solid {r_cfg['cor']}" if is_active else f"1px solid {r_cfg['borda']}"
    col.markdown(
        f"""<div style="border:{bdr};border-radius:10px;padding:14px 12px;
            background:{r_cfg['bg']};height:100%">
        <div style="font-size:1.1rem;font-weight:700;color:{r_cfg['cor']};margin-bottom:6px">
            {r_cfg['emoji']} {r_cfg['titulo']}
            {'<br><span style="font-size:0.72rem;color:#6b7280">← REGIME ATUAL</span>' if is_active else ''}
        </div>
        <div style="font-size:0.82rem;color:#374151;margin-bottom:10px">{r_cfg['desc'][:120]}...</div>
        <hr style="border:none;border-top:1px solid {r_cfg['borda']};margin:8px 0">
        <div style="font-size:0.82rem">
            📈 Core Growth: <b>{core_pct:.0f}%</b><br>
            🛡️ Defensivos: <b>{def_pct:.0f}%</b><br>
            💵 Renda Fixa: <b>{fi_pct:.0f}%</b><br>
            🔒 Proteção: <b>{prot_pct:.0f}%</b>
        </div>
        </div>""",
        unsafe_allow_html=True
    )

# ── Nota de atualização ───────────────────────────────────────────────────────
st.markdown("---")
next_score = loaded_at + timedelta(hours=6)
next_price = prices_at + timedelta(hours=1)
st.caption(
    f"🕐 **Próxima atualização do score:** {next_score.strftime('%d/%m %H:%M')}  "
    f"(a cada 6h — alinhado à abertura ~10h30 BRT e fechamento ~17h BRT do mercado americano)  \n"
    f"💱 **Próxima atualização de preços:** {next_price.strftime('%d/%m %H:%M')} (a cada 1h)  \n"
    f"⚠️ *Este painel é informativo. Não constitui recomendação de investimento.*"
)
