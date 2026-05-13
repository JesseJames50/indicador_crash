import os
from dotenv import load_dotenv

load_dotenv()

# Prioridade: variável de ambiente (.env local) → Streamlit secrets (Cloud) → vazio
def _get_fred_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("FRED_API_KEY", "")
    except Exception:
        return ""

FRED_API_KEY = _get_fred_key()

START_DATE = "2015-01-01"

# Pesos do score sistêmico
WEIGHTS = {
    "t10y": 0.2,
    "kre": 0.2,
    "hy_spread": 0.2,
    "funding": 0.2,
    "volatility": 0.2
}
