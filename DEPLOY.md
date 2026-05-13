# Deploy no Streamlit Community Cloud

## Pré-requisitos

- Conta no [Streamlit Community Cloud](https://streamlit.io/cloud) (gratuita)
- Repositório público no GitHub com o código já commitado
- Chave de API do FRED ([obtenha aqui](https://fred.stlouisfed.org/docs/api/api_key.html), gratuita)

---

## Passo 1 — Verificar os arquivos no repositório

Certifique-se de que estes arquivos estão commitados:

```
app_main.py          ← arquivo principal (painel unificado v6 + v7)
app2.py              ← v6 standalone (opcional)
app3.py              ← v7 standalone (opcional)
data_loader.py
config.py
requirements.txt
.streamlit/secrets.toml.example   ← template (NÃO commitar secrets.toml)
```

> **Importante:** `.env` e `.streamlit/secrets.toml` **nunca** devem ir para o git — já estão no `.gitignore`.

---

## Passo 2 — Fazer o deploy

1. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com sua conta GitHub
2. Clique em **"New app"**
3. Preencha:
   - **Repository:** `seu-usuario/indicador_crash`
   - **Branch:** `master`
   - **Main file path:** `app_main.py`
4. Clique em **"Deploy!"**

---

## Passo 3 — Configurar a chave do FRED (Secrets)

Depois do deploy (ou antes, via "Advanced settings"):

1. No painel do app, clique em **⋮ → Settings → Secrets**
2. Cole o seguinte conteúdo, substituindo pelo valor real:

```toml
FRED_API_KEY = "sua_chave_aqui"
```

3. Clique em **Save** — o app reinicia automaticamente

> O `config.py` já está configurado para ler tanto de variável de ambiente local (`.env`) quanto dos Streamlit Secrets (Cloud).

---

## Passo 4 — URL pública

Após o deploy, o app fica disponível em:
```
https://seu-usuario-indicador-crash-app-main-xxxxx.streamlit.app
```

Você pode compartilhar essa URL diretamente.

---

## Atualizações automáticas

Cada `git push` para `master` dispara um redeploy automático no Streamlit Cloud.

---

## Dica — Testar localmente antes do deploy

```bash
# Ative o ambiente virtual
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Execute o painel unificado
streamlit run app_main.py
```

O dashboard abre em `http://localhost:8501`.
