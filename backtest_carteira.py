"""
Backtest buy-and-hold da carteira sugerida — últimos 12 meses
Capital inicial: R$ 300.000
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Configuração ──────────────────────────────────────────────────────────────
CAPITAL_BRL   = 300_000
END_DATE      = datetime.today()
START_DATE    = END_DATE - timedelta(days=365)

PORTFOLIO = {
    "SPY":  0.20,
    "QQQ":  0.12,
    "VTV":  0.08,
    "XLP":  0.08,
    "XLV":  0.07,
    "XLU":  0.05,
    "IEF":  0.10,
    "SCHP": 0.08,
    "GLD":  0.12,
    "BIL":  0.07,
    "PDBC": 0.03,
}

LABELS = {
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq-100",
    "VTV":  "Vanguard Value",
    "XLP":  "Consumer Staples",
    "XLV":  "Healthcare",
    "XLU":  "Utilities",
    "IEF":  "Treasuries 7-10a",
    "SCHP": "TIPS",
    "GLD":  "Ouro",
    "BIL":  "T-Bills 1-3m",
    "PDBC": "Commodities",
}

BLOCOS = {
    "SPY": "Crescimento Core",
    "QQQ": "Crescimento Core",
    "VTV": "Crescimento Core",
    "XLP": "Setor Defensivo",
    "XLV": "Setor Defensivo",
    "XLU": "Setor Defensivo",
    "IEF": "Renda Fixa",
    "SCHP":"Renda Fixa",
    "GLD": "Proteção Sistêmica",
    "BIL": "Proteção Sistêmica",
    "PDBC":"Proteção Sistêmica",
}

# ── Download via Yahoo Finance API (requests, sem validação SSL) ──────────────
import requests, urllib3
urllib3.disable_warnings()

def _dl(ticker: str) -> pd.Series:
    """Baixa série de fechamento ajustado via Yahoo Finance API v8."""
    yf_ticker = ticker.replace("^", "%5E")
    p1 = int(START_DATE.timestamp())
    p2 = int(END_DATE.timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
           f"?interval=1d&period1={p1}&period2={p2}&events=div,splits")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, verify=False, timeout=20, headers=headers)
        r.raise_for_status()
        data   = r.json()
        result = data["chart"]["result"][0]
        ts     = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize()
        # adjclose se disponível, senão close
        ind = result["indicators"]
        if "adjclose" in ind and ind["adjclose"]:
            closes = ind["adjclose"][0]["adjclose"]
        else:
            closes = ind["quote"][0]["close"]
        s = pd.Series(closes, index=ts, name=ticker, dtype=float)
        s.index = s.index.tz_localize(None)
        return s.dropna()
    except Exception as e:
        return pd.Series(dtype=float, name=ticker)

print("Baixando dados de ETFs...")
series_dict = {}
for t in list(PORTFOLIO.keys()) + ["USDBRL=X", "BRL=X"]:
    print(f"  {t}...", end=" ", flush=True)
    s = _dl(t)
    print(f"{'OK' if len(s)>0 else 'FALHOU'} ({len(s)} pontos)")
    series_dict[t] = s

# USD/BRL — tenta USDBRL=X, fallback BRL=X
usdbrl_s = series_dict.pop("USDBRL=X")
if usdbrl_s.empty:
    usdbrl_s = series_dict.pop("BRL=X")
else:
    series_dict.pop("BRL=X", None)

etf_prices = pd.DataFrame({t: s for t, s in series_dict.items() if len(s) > 20}).dropna(how="all")

# Taxa de câmbio inicial e final
if usdbrl_s is not None and usdbrl_s.notna().any():
    fx_start = float(usdbrl_s.dropna().iloc[0])
    fx_end   = float(usdbrl_s.dropna().iloc[-1])
else:
    fx_start = fx_end = 5.80   # fallback estimado

fx_ret = fx_end / fx_start - 1

# ── Retornos individuais ──────────────────────────────────────────────────────
results = []
for ticker, weight in PORTFOLIO.items():
    if ticker not in etf_prices.columns:
        print(f"  AVISO: {ticker} não disponível, ignorado.")
        continue
    s = etf_prices[ticker].dropna()
    if len(s) < 20:
        continue

    p0    = float(s.iloc[0])
    pf    = float(s.iloc[-1])
    ret_usd = pf / p0 - 1

    # Retorno em BRL = (1 + ret_usd) * (1 + fx_ret) - 1
    ret_brl = (1 + ret_usd) * (1 + fx_ret) - 1

    capital_etf  = CAPITAL_BRL * weight
    ganho_brl    = capital_etf * ret_brl
    capital_final = capital_etf + ganho_brl

    # Volatilidade anualizada (USD)
    daily_ret = s.pct_change().dropna()
    vol_anual = float(daily_ret.std() * np.sqrt(252) * 100)

    # Max drawdown
    roll_max  = s.cummax()
    drawdown  = (s - roll_max) / roll_max
    max_dd    = float(drawdown.min() * 100)

    results.append({
        "Ticker":       ticker,
        "Bloco":        BLOCOS[ticker],
        "Descrição":    LABELS[ticker],
        "Peso":         f"{weight*100:.0f}%",
        "Capital":      capital_etf,
        "Ret USD":      ret_usd,
        "Ret BRL":      ret_brl,
        "Ganho R$":     ganho_brl,
        "Final R$":     capital_final,
        "Vol anual %":  vol_anual,
        "Max DD %":     max_dd,
    })

df = pd.DataFrame(results)

# ── Portfólio agregado ────────────────────────────────────────────────────────
port_ret_brl  = sum(r["Ret BRL"] * PORTFOLIO[r["Ticker"]] for r in results)
port_ret_usd  = sum(r["Ret USD"] * PORTFOLIO[r["Ticker"]] for r in results)
total_ganho   = sum(r["Ganho R$"] for r in results)
total_final   = CAPITAL_BRL + total_ganho

# Benchmark SPY
spy_row = next((r for r in results if r["Ticker"] == "SPY"), None)
spy_ret_brl = spy_row["Ret BRL"] if spy_row else None

# Portfólio série temporal (normalizada)
norm    = etf_prices[list(PORTFOLIO.keys())].dropna(how="all").ffill()
norm    = norm / norm.iloc[0]
weights_arr = np.array([PORTFOLIO[t] for t in norm.columns])
port_series = (norm * weights_arr).sum(axis=1)
port_daily  = port_series.pct_change().dropna()
port_vol    = float(port_daily.std() * np.sqrt(252) * 100)
port_dd     = float(((port_series - port_series.cummax()) / port_series.cummax()).min() * 100)
sharpe      = (port_ret_usd / (port_vol / 100)) if port_vol > 0 else 0

# ── Exibição ──────────────────────────────────────────────────────────────────
sep = "-" * 110

print(f"\n{sep}")
print(f"  BACKTEST CARTEIRA — {START_DATE.strftime('%d/%m/%Y')} a {END_DATE.strftime('%d/%m/%Y')}")
print(f"  Capital inicial: R$ {CAPITAL_BRL:,.0f}   |   USD/BRL início: {fx_start:.2f}   fim: {fx_end:.2f}   (variação: {fx_ret*100:+.1f}%)")
print(sep)

# Tabela individual
header = f"{'Ticker':<6} {'Bloco':<22} {'Descrição':<20} {'Peso':>5}  {'Capital Ini':>12}  {'Ret USD':>8}  {'Ret BRL':>8}  {'Ganho R$':>12}  {'Final R$':>12}  {'Vol%':>6}  {'MaxDD%':>7}"
print(header)
print("-" * 110)

prev_bloco = ""
for r in results:
    if r["Bloco"] != prev_bloco:
        if prev_bloco:
            print()
        prev_bloco = r["Bloco"]
    ganho_str = f"{'▲' if r['Ganho R$'] >= 0 else '▼'} {abs(r['Ganho R$']):,.0f}"
    print(
        f"{r['Ticker']:<6} {r['Bloco']:<22} {r['Descrição']:<20} {r['Peso']:>5}  "
        f"R${r['Capital']:>10,.0f}  "
        f"{r['Ret USD']*100:>+7.1f}%  "
        f"{r['Ret BRL']*100:>+7.1f}%  "
        f"{ganho_str:>13}  "
        f"R${r['Final R$']:>10,.0f}  "
        f"{r['Vol anual %']:>5.1f}%  "
        f"{r['Max DD %']:>6.1f}%"
    )

print(f"\n{'─'*110}")
print(f"{'PORTFÓLIO TOTAL':<52}  R${CAPITAL_BRL:>10,.0f}  "
      f"{port_ret_usd*100:>+7.1f}%  "
      f"{port_ret_brl*100:>+7.1f}%  "
      f"{'▲' if total_ganho>=0 else '▼'} {abs(total_ganho):>10,.0f}  "
      f"R${total_final:>10,.0f}  "
      f"{port_vol:>5.1f}%  "
      f"{port_dd:>6.1f}%")
if spy_ret_brl is not None:
    print(f"{'BENCHMARK SPY (buy-and-hold)':<52}  {'':>12}  "
          f"{'':>8}  "
          f"{spy_ret_brl*100:>+7.1f}%")

print(sep)

# Resumo executivo
print(f"\n  RESUMO EXECUTIVO")
print(f"  {'Capital inicial':.<35} R$ {CAPITAL_BRL:>12,.0f}")
print(f"  {'Capital final':.<35} R$ {total_final:>12,.0f}")
print(f"  {'Ganho / Perda total':.<35} R$ {total_ganho:>+12,.0f}  ({port_ret_brl*100:+.2f}% em BRL)")
print(f"  {'Retorno em USD':.<35} {port_ret_usd*100:>+11.2f}%")
print(f"  {'Variação USD/BRL no período':.<35} {fx_ret*100:>+11.2f}%")
print(f"  {'Volatilidade anual (USD)':.<35} {port_vol:>11.1f}%")
print(f"  {'Max Drawdown portfólio':.<35} {port_dd:>11.1f}%")
print(f"  {'Sharpe ratio (approx., rf=0)':.<35} {sharpe:>11.2f}")
if spy_ret_brl:
    alpha = port_ret_brl - spy_ret_brl
    print(f"  {'Alpha vs SPY (BRL)':.<35} {alpha*100:>+11.2f}%")

# Breakdown por bloco
print(f"\n  DESEMPENHO POR BLOCO")
for bloco in ["Crescimento Core", "Setor Defensivo", "Renda Fixa", "Proteção Sistêmica"]:
    bloco_rows = [r for r in results if r["Bloco"] == bloco]
    if not bloco_rows:
        continue
    peso_bloco = sum(PORTFOLIO[r["Ticker"]] for r in bloco_rows)
    ret_bloco  = sum(r["Ret BRL"] * PORTFOLIO[r["Ticker"]] for r in bloco_rows) / peso_bloco
    ganho_bloco = sum(r["Ganho R$"] for r in bloco_rows)
    print(f"  {bloco:<25} peso={peso_bloco*100:.0f}%  ret={ret_bloco*100:+.1f}%  ganho=R$ {ganho_bloco:>+10,.0f}")

# Período efetivo
print(f"\n  Período efetivo: {norm.index[0].strftime('%d/%m/%Y')} → {norm.index[-1].strftime('%d/%m/%Y')}  ({len(norm)} pregões)")
print(sep)
