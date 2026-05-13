# Monitor de Liquidez Sistêmica — v7 + Monitor de Carteira

Dashboard Streamlit para monitoramento contínuo do risco de crise financeira no mercado americano, baseado em 10 indicadores de estresse de liquidez, com rebalanceamento dinâmico de portfólio.

---

## Arquivos principais

| Arquivo | Descrição |
|---|---|
| `app_main.py` | **Painel unificado** — v6 + v7 + Comparação em três abas (recomendado) |
| `portfolio_monitor.py` | **Monitor de carteira** — risco sistêmico + rebalanceamento dinâmico |
| `app2.py` | Dashboard v6 standalone (baseline, 7 indicadores) |
| `app3.py` | Dashboard v7 standalone (Framework MOVE, 10 indicadores) |
| `backtest_carteira.py` | Backtest buy-and-hold da carteira sugerida (script de linha de comando) |

---

## Instalação rápida

```bash
# 1. Clone o repositório
git clone https://github.com/JesseJames50/indicador_crash.git
cd indicador_crash

# 2. Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure a chave da API do FRED
copy .env.example .env
# edite .env e coloque sua chave: FRED_API_KEY=sua_chave_aqui
# Chave gratuita em: https://fred.stlouisfed.org/docs/api/api_key.html

# 5. Execute o painel de sua escolha
streamlit run app_main.py          # painel unificado (recomendado)
streamlit run portfolio_monitor.py # monitor de carteira
```

O dashboard abre automaticamente em `http://localhost:8501`

> Para deploy no Streamlit Community Cloud, consulte [DEPLOY.md](DEPLOY.md).

---

## app_main.py — Painel Unificado (v6 + v7)

Três abas em uma única tela, com um único carregamento de dados:

### Aba v6 — Baseline (7 indicadores)

O modelo original com sete componentes calibrados entre 2018-2024.

| Indicador | Peso | O que capta |
|---|---|---|
| HY Spread de Crédito | 25% | Risco de crédito corporativo — principal preditor histórico |
| T-Bill 3M Stress | 20% | Fuga para títulos de curto prazo — pânico de liquidez |
| KRE — queda do pico | 20% | Deterioração do setor bancário regional |
| Curva 10Y-2Y (invertida) | 15% | Preditor clássico de recessão (6-18 meses) |
| VIX | 10% | Medo do mercado — sinal coincidente |
| T-Note 10 anos | 5% | Nível de juros longos |
| SOFR − Fed Funds | 5% | Stress no mercado overnight |

### Aba v7 — Framework MOVE (10 indicadores)

Incorpora volatilidade do mercado de bonds e spreads investment grade.

| Indicador | Peso | O que capta |
|---|---|---|
| HY Spread de Crédito | 21% | Principal preditor cross-asset de recessão |
| MOVE / Vol T10Y | 15% | Vol implícita de bonds — lidera o VIX em crises de funding |
| T-Bill 3M Stress | 15% | Fuga para T-Bills |
| KRE — queda do pico | 17% | Deterioração bancária gradual |
| Curva 10Y-2Y (invertida) | 12% | Inversão da curva |
| IG OAS | 7% | Spreads investment grade — divergência IG/HY sinaliza contágio |
| Divergência MOVE/VIX | 5% | Bonds estressados antes das equities reagirem |
| VIX | 5% | Confirmação coincidente |
| T-Note 10 anos | 2% | Contexto macro |
| SOFR − Fed Funds | 1% | Stress overnight |

> **DXY removido do score:** em crises de confiança no dólar (Tarifas 2025), o dólar cai e penaliza o sinal. Mantido em **Sinais Avançados** para análise qualitativa.

#### Melhorias do v7 vs v6 (backtest 2018-2026)

| Métrica | v6 | v7 |
|---|---|---|
| Detecção precoce (velocidade) | 5/5 (100%) | 5/5 (100%) |
| Antecipação média | 108 dias | 112 dias |
| COVID — confirmação P92 | ⚠️ durante | ✅ 178 dias antes |
| Bear Market 2022 — velocidade | 15 dias antes | 47 dias antes |
| Falsos positivos P92 | 0,2% | 0,8% |

### Aba Comparação

Tabela lado a lado v6 → v7 por evento de crise, com delta de antecipação e resumo de falsos positivos.

---

## portfolio_monitor.py — Monitor de Carteira

Dashboard de rebalanceamento que integra o score v7 com alocação dinâmica.

### Como funciona

1. **Detecta o regime** atual com base no score v7
2. **Compara** suas posições atuais (informadas na sidebar) com a alocação alvo do regime
3. **Indica** o que comprar, vender ou manter — em R$ e percentual

### Níveis de risco e alocações alvo

| Regime | Score | Core Growth | Defensivos | Renda Fixa | Proteção |
|---|---|---|---|---|---|
| 🟢 Normal | < P70 | 50% | 15% | 5% | 30% |
| 🟡 Atenção | ≥ P70 (3d) | 40% | 20% | 18% | 22% |
| ⚡ Aceleração | velocidade P90 | 30% | 24% | 20% | 26% |
| 🔴 Crítico | ≥ P92 (3d) | 20% | 26% | 24% | 30% |

### Carteira monitorada (11 ETFs)

| Bloco | ETF | Descrição |
|---|---|---|
| Crescimento Core | SPY, QQQ, VTV | S&P 500, Nasdaq-100, Vanguard Value |
| Setor Defensivo | XLP, XLV, XLU | Consumer Staples, Healthcare, Utilities |
| Renda Fixa | IEF, SCHP | Treasuries 7-10a, TIPS |
| Proteção Sistêmica | GLD, BIL, PDBC | Ouro, T-Bills 1-3m, Commodities |

### Atualização automática

- **Score v7:** a cada 6 horas — alinhado à abertura (~10h30 BRT) e fechamento (~17h BRT) do mercado americano
- **Preços dos ETFs:** a cada 1 hora
- **Refresh manual:** botões na sidebar

---

## Como interpretar o dashboard (app_main.py)

### Painel de métricas (4 cartões no topo)

| Cartão | O que significa |
|---|---|
| **Score EMA-21** | Nível atual de estresse sistêmico, suavizado por média móvel de 21 dias |
| **Threshold Atenção** | P70 do score histórico 2018-2024 — 70% dos dias estiveram abaixo |
| **Threshold Crítico** | P92 do mesmo período — apenas 8% dos dias históricos superaram esse nível |
| **Velocidade (5d)** | Variação do score em 5 dias. Alerta ⚡ quando acelera acima do P90 histórico |

### Status do sistema

| Status | Significado prático |
|---|---|
| 🟢 **Normal** | Score abaixo do threshold de atenção. Mercado em regime tranquilo. |
| 🟡 **Atenção** | Score acima do P70 por 3+ dias consecutivos. Monitoramento diário recomendado. |
| ⚡ **Aceleração** | Score subindo rapidamente — deterioração em curso antes de atingir o threshold crítico. |
| 🔴 **Crítico** | Score acima do P92 por 3+ dias. Historicamente associado a eventos de crise. |

> O filtro de **3 dias consecutivos** evita falsos alarmes causados por spikes isolados.

### Gráfico principal — Score histórico

- **Linha azul**: Score suavizado (EMA-21) — indicador principal
- **Linha pontilhada cinza**: Score bruto — volatilidade real dos componentes
- **Triângulos laranja (⚡)**: Dias com alerta de aceleração
- **Faixas vermelhas**: Janelas de crise mapeadas
- **Faixas cinzas**: Períodos de QE ativo (spreads artificialmente comprimidos)
- **Linhas tracejadas**: Thresholds fixos de Atenção (amarelo) e Crítico (vermelho)

### Sinais Avançados (aba v7)

**Quadrante MOVE/VIX** — o mais valioso é o quadrante Q2:

| Quadrante | Condição | Interpretação |
|---|---|---|
| Q2 ⚠️ | MOVE ↑ VIX ↓ | Bonds estressados, equities calmas — antecipação de stress |
| Q1 🔴 | MOVE ↑ VIX ↑ | Ambos estressados — crise sistêmica confirmada |
| Q3 🟡 | MOVE ↓ VIX ↑ | Stress isolado em equities |
| Q4 🟢 | MOVE ↓ VIX ↓ | Regime calmo — melhor ambiente para risco |

### Backtest (aba v7)

| Coluna | Significado |
|---|---|
| **⚡ Vel. pré-crise** | Alerta de velocidade disparou antes do evento (sinal precoce) |
| **P90 confirmação** | Threshold P92 cruzado antes (✅), durante (⚠️) ou não detectado (❌) |
| **Score máx. pré** | Score máximo nos 180 dias antes do evento |
| **Score máx. crise** | Score máximo durante a janela de crise |

---

## backtest_carteira.py — Backtest da Carteira

Script de linha de comando que baixa 1 ano de dados e calcula o desempenho da carteira:

```bash
python backtest_carteira.py
```

Saída: retorno em USD e BRL por ETF, volatilidade anual, max drawdown, Sharpe ratio e alpha vs SPY.

> **Nota de execução:** use o Python do Anaconda base (`python`) ou configure o ambiente virtual com SSL válido. O script usa a API do Yahoo Finance diretamente (bypass SSL corporativo).

---

## Atualização dos dados

Os dados são recarregados automaticamente:

| Dashboard | Frequência | Horários (BRT) |
|---|---|---|
| app_main.py | A cada 12h | ~10h30 e ~22h30 |
| portfolio_monitor.py (score) | A cada 6h | ~10h30, ~16h30, ~22h30, ~04h30 |
| portfolio_monitor.py (preços) | A cada 1h | Continuamente |

Para forçar atualização imediata: botão **🔄 Forçar atualização** na sidebar.

---

## Fontes de dados

| Dado | Fonte | Frequência |
|---|---|---|
| HY Spread (`BAMLH0A0HYM2`) | FRED | Diária |
| IG OAS (`BAMLC0A0CM`) | FRED | Diária |
| T-Bill 3M (`DTB3`) | FRED | Diária |
| Curva 10Y-2Y (`T10Y2Y`) | FRED | Diária |
| SOFR, Fed Funds (`SOFR`, `FEDFUNDS`) | FRED | Diária / Mensal |
| Balanço do Fed (`WALCL`) | FRED | Semanal |
| MOVE Index (`^MOVE`) | Yahoo Finance | Diária |
| KRE, VIX, QQQ, T-Note (`^TNX`) | Yahoo Finance | Diária |
| ETFs da carteira (SPY, QQQ, …) | Yahoo Finance | Diária |
| USD/BRL (`USDBRL=X`) | Yahoo Finance | Diária |

---

## Deploy no Streamlit Community Cloud

Consulte [DEPLOY.md](DEPLOY.md) para o guia completo passo-a-passo.

Resumo:
1. Fork do repositório no GitHub
2. Deploy em [share.streamlit.io](https://share.streamlit.io) apontando para `app_main.py`
3. Configurar `FRED_API_KEY` em **Settings → Secrets**

---

*Este dashboard é uma ferramenta de monitoramento informativo e não constitui recomendação de investimento.*
