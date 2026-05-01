import os
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

START_DATE = "2015-01-01"

# Pesos do score sistêmico
WEIGHTS = {
    "t10y": 0.2,
    "kre": 0.2,
    "hy_spread": 0.2,
    "funding": 0.2,
    "volatility": 0.2
}
