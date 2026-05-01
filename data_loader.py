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


def _get_fred_series(series_id: str, start: str = START_DATE) -> pd.Series:
    """Busca série do FRED; retorna Series vazia em caso de falha."""
    try:
        return fred.get_series(series_id, observation_start=start)
    except Exception as exc:
        import streamlit as st
        st.warning(f"FRED: não foi possível carregar '{series_id}' ({exc}). Série ignorada.")
        return pd.Series(dtype=float, name=series_id)


def get_fred_data():
    # SOFR só existe a partir de 04/2018; usar essa data evita erro 500 do FRED.
    return pd.DataFrame({
        'hy_spread': _get_fred_series("BAMLH0A0HYM2"),
        'sofr':      _get_fred_series("SOFR", start="2018-04-02"),
        'fed_funds': _get_fred_series("FEDFUNDS"),
    })


def merge_data():
    yahoo = get_yahoo_data()
    fred_data = get_fred_data()

    df = yahoo.join(fred_data, how='outer')
    df = df.ffill()

    return df