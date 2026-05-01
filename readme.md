# 📊 Monitor de Risco Sistêmico

Dashboard Streamlit para monitorar os 3 indicadores de alerta precoce de crise financeira americana.

## Indicadores monitorados

| # | Indicador | Ticker | Alerta |
|---|-----------|--------|--------|
| 1 | T-Note 10 anos | `^TNX` (Yahoo Finance) | > 5% por 3 dias consecutivos |
| 2 | KRE — Bancos Regionais | `KRE` (Yahoo Finance) | Queda > 30% do pico recente |
| 3 | Spread HY de Crédito | `BAMLH0A0HYM2` (FRED) | > 6% |

## Instalação

### 1. Clone ou baixe o projeto

```bash
mkdir financial_monitor && cd financial_monitor
# copie os arquivos app.py e requirements.txt aqui
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o app

```bash
streamlit run app.py
```

O app abrirá automaticamente no navegador em `http://localhost:8501`

## Funcionalidades

- **Atualização automática** a cada 5 minutos (com cache)
- **3 níveis de alerta**: Normal ✓ / Atenção ⚡ / Crítico ⚠
- **Banner de alerta sistêmico** quando 2+ indicadores disparam
- **Gráficos históricos** com linhas de limiar (1 mês a 2 anos)
- **Tabela de referência** rápida para interpretação
- **Contador de alertas ativos** em tempo real

## Sem necessidade de API Key

- Yahoo Finance: acesso gratuito via `yfinance`
- FRED (Federal Reserve): CSV público, sem cadastro

## Interpretação dos alertas

```
Normal:     Todos os indicadores dentro do esperado
1 Atenção:  Monitoramento diário recomendado
2 Atenções: Revise sua liquidez e reservas
1 Crítico:  Verifique os outros indicadores imediatamente
2+ Críticos: Sinal histórico de crise sistêmica — consulte assessor
```

---
*Não constitui recomendação de investimento.*
