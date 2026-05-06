# 📊 Monitor de Liquidez Sistêmica v6

Dashboard Streamlit para monitoramento contínuo do risco de crise financeira no mercado americano, baseado em 7 indicadores de estresse de liquidez.

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

# 5. Execute
streamlit run app2.py
```

O dashboard abre automaticamente em `http://localhost:8501`

---

## Como interpretar o dashboard

### Painel de métricas (4 cartões no topo)

| Cartão | O que significa |
|---|---|
| **Score EMA-21** | Nível atual de estresse sistêmico. Parte de 0 em mercado calmo e sobe durante crises. Suavizado por média móvel de 21 dias para eliminar ruído. |
| **Threshold Atenção** | Valor de referência para alerta amarelo. Foi o nível P70 (percentil 70) do score histórico entre 2018 e 2024 — ou seja, em 70% dos dias desse período o mercado estava mais tranquilo do que isso. |
| **Threshold Crítico** | Valor de referência para alerta vermelho. Foi o P90 do mesmo período — apenas 10% dos dias históricos tiveram estresse acima desse nível, correspondendo às janelas de crise mapeadas. |
| **Velocidade (5d)** | Variação do score nos últimos 5 dias. Positivo = estresse acelerando. O alerta ⚡ só dispara quando a aceleração é ao mesmo tempo rápida (P90 histórico) e expressiva (mínimo 0.04 em 5 dias). |

---

### Status do sistema

| Status | Significado prático |
|---|---|
| 🟢 **Normal** | Score abaixo do threshold de atenção. Mercado em regime tranquilo. |
| 🟡 **Atenção** | Score acima do P70 por 3+ dias consecutivos. Monitoramento diário recomendado. |
| ⚡ **Aceleração** | Score subindo rapidamente mesmo sem cruzar o threshold crítico. Sinal de deterioração em curso — atenção redobrada. |
| 🔴 **Crítico** | Score acima do P90 por 3+ dias consecutivos. Historicamente associado a eventos de crise. |

> O filtro de **3 dias consecutivos** evita falsos alarmes causados por spikes isolados de um único dia.

---

### Gráfico principal — Score histórico

- **Linha azul**: Score suavizado (EMA-21) — o indicador principal.
- **Linha pontilhada cinza**: Score bruto antes da suavização — mostra a volatilidade real dos componentes.
- **Triângulos laranja (⚡)**: Dias com alerta de aceleração.
- **Faixas vermelhas**: Janelas de crise mapeadas (COVID, SVB, Tarifas, etc.) para referência visual.
- **Faixas cinzas**: Períodos de QE ativo (expansão do balanço do Fed) — durante esses períodos os spreads ficam artificialmente comprimidos, o que reduz a sensibilidade do score.
- **Linhas tracejadas**: Thresholds fixos de Atenção (amarelo) e Crítico (vermelho).

---

### Gráfico de Velocidade

Mostra a variação do score em janelas de 5 dias:
- **Barras azuis (acima de zero)**: score acelerando — estresse aumentando.
- **Barras cinzas (abaixo de zero)**: score desacelerando — estresse diminuindo.
- **Linha laranja pontilhada**: threshold P90 da velocidade. Quando ultrapassado junto com valor absoluto > 0.04 e score em zona de atenção, dispara o alerta ⚡.

---

### Score vs QQQ

Gráfico de eixo duplo comparando o score (azul, eixo esquerdo) com o preço do QQQ — ETF do Nasdaq-100 (verde, eixo direito). Permite visualizar se picos de estresse coincidiram ou anteciparam quedas no mercado.

- **Pontos vermelhos sobre o QQQ**: dias em que o alerta crítico estava ativo.
- O objetivo é que os picos do score apareçam *antes* das quedas do QQQ — sinalizando antecipação.

---

### Painel de componentes (7 gráficos)

Cada gráfico mostra a contribuição individual de um indicador ao score:
- **Linha cinza clara**: z-score bruto do componente (pode ser negativo).
- **Área colorida preenchida**: contribuição efetiva — só conta quando o z-score ultrapassa 0.25 (deadzone). Abaixo disso, o componente não contribui para o score.

| Indicador | Peso | O que capta |
|---|---|---|
| HY Spread de Crédito | 25% | Risco de crédito corporativo — principal preditor histórico de recessão |
| T-Bill 3M Stress | 20% | Fuga para títulos do Tesouro de curto prazo — sinal de pânico de liquidez |
| KRE — queda do pico | 20% | Deterioração do setor bancário regional — capturou SVB antes do colapso |
| Curva 10Y-2Y (invertida) | 15% | Inversão da curva de juros — preditor clássico de recessão com 6–18 meses de antecedência |
| VIX | 10% | Medo do mercado — sinal coincidente (não antecipa, confirma) |
| T-Note 10 anos | 5% | Nível de juros longos — contexto macroeconômico |
| SOFR − Fed Funds | 5% | Stress no mercado overnight — histórico mais curto (desde 2018) |

---

### Tabela de backtest

Resume o desempenho do score nas 5 crises mapeadas:

| Coluna | Significado |
|---|---|
| **Detectado?** | ✅ Antecipado = alerta antes do evento; ⚠️ Durante = detectou mas só durante; ❌ Não detectado |
| **Antecipação (dias)** | Quantos dias antes do evento o alerta crítico disparou. Negativo = detectou depois |
| **Score máx. pré** | Score máximo nos 45 dias antes do evento |
| **Score máx. crise** | Score máximo durante a janela de crise |
| **⚡ Velocidade antes?** | Se o alerta de aceleração disparou antes do evento |

Abaixo da tabela: percentual de **falsos positivos** — dias em que o alerta estava ativo fora de qualquer janela de crise mapeada.

---

## Atualização dos dados

Os dados são **recarregados automaticamente a cada 12 horas**, sincronizando com:
- **Abertura do mercado americano** (~9h30 ET / ~10h30 horário de Brasília)
- **Fechamento do mercado americano** (~16h ET / ~17h horário de Brasília)

Para forçar uma atualização imediata, use o botão **🔄 Forçar atualização** na barra lateral.

---

## Fontes de dados

| Dado | Fonte | Frequência |
|---|---|---|
| HY Spread (`BAMLH0A0HYM2`) | FRED — Federal Reserve | Diária |
| T-Bill 3M (`DTB3`) | FRED | Diária |
| Curva 10Y-2Y (`T10Y2Y`) | FRED | Diária |
| SOFR, Fed Funds (`SOFR`, `FEDFUNDS`) | FRED | Diária / Mensal |
| Balanço do Fed (`WALCL`) | FRED | Semanal |
| KRE, VIX, QQQ, T-Note (`^TNX`) | Yahoo Finance | Diária |

---

*Este dashboard é uma ferramenta de monitoramento informativo e não constitui recomendação de investimento.*
