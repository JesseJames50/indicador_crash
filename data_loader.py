import time
import yfinance as yf
import pandas as pd
from fredapi import Fred
from config import FRED_API_KEY, START_DATE

fred = Fred(api_key=FRED_API_KEY)


def _download_close(ticker: str) -> pd.Series:
    """Baixa série de fechamento compatível com yfinance antigo e novo (MultiIndex)."""
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    arr = df.filter(like="Close").to_numpy()
    col = arr[:, 0] if arr.ndim == 2 else arr
    return pd.Series(col, index=df.index, name=ticker, dtype=float)


def get_yahoo_data():
    return pd.DataFrame({
        't10y': _download_close("^TNX"),
        'kre':  _download_close("KRE"),
        'vix':  _download_close("^VIX"),
    })


def _get_fred_series(series_id: str, start: str = START_DATE,
                     retries: int = 2) -> pd.Series:
    """Busca série do FRED com retry; retorna Series vazia após todas as tentativas."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fred.get_series(series_id, observation_start=start)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.5)
    import streamlit as st
    st.warning(f"FRED: não foi possível carregar '{series_id}' ({last_exc}). Série ignorada.")
    return pd.Series(dtype=float, name=series_id)


def get_fred_data():
    # FEDFUNDS (mensal) com fallback para DFF (diário) — mesmos dados, freq diferente
    fed_funds = _get_fred_series("FEDFUNDS")
    if fed_funds.dropna().empty:
        fed_funds = _get_fred_series("DFF")   # Daily Federal Funds Rate

    # SOFR só existe a partir de 04/2018
    sofr = _get_fred_series("SOFR", start="2018-04-02")
    if sofr.dropna().empty:
        sofr = _get_fred_series("SOFR90DAYAVG", start="2018-04-02")

    return pd.DataFrame({
        'hy_spread': _get_fred_series("BAMLH0A0HYM2"),
        'sofr':      sofr,
        'fed_funds': fed_funds,
    })


def merge_data():
    yahoo = get_yahoo_data()
    fred_data = get_fred_data()

    df = yahoo.join(fred_data, how='outer')
    df = df.ffill()

    return df