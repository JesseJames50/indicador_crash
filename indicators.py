import pandas as pd
from config import WEIGHTS

def zscore(series):
    return (series - series.mean()) / series.std()

def compute_indicators(df):

    result = pd.DataFrame(index=df.index)

    # Normalização
    result['t10y_z'] = zscore(df['t10y'])
    result['kre_z'] = zscore(df['kre'].pct_change())
    result['hy_z'] = zscore(df['hy_spread'])

    # Funding stress
    result['funding'] = df['sofr'] - df['fed_funds']
    result['funding_z'] = zscore(result['funding'])

    # Volatilidade
    result['vix_z'] = zscore(df['vix'])

    # Score agregado
    result['systemic_score'] = (
        WEIGHTS['t10y'] * result['t10y_z'] +
        WEIGHTS['kre'] * (-result['kre_z']) +  # queda é risco
        WEIGHTS['hy_spread'] * result['hy_z'] +
        WEIGHTS['funding'] * result['funding_z'] +
        WEIGHTS['volatility'] * result['vix_z']
    )

    return result
