"""
Análise de qualquer ativo do Yahoo Finance (ações da B3 e do exterior, ETFs,
cripto, índices, câmbio):
  1. Coleta de preços ajustados via yfinance
  2. Indicadores técnicos (médias móveis, RSI, Bollinger)
  3. Métricas de risco/retorno (volatilidade, Sharpe, drawdown)

Instalação:
    pip install yfinance pandas numpy requests

Uso:
    python analise_b3.py
"""

import re

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["PETR4", "VALE3", "ITUB4", "BBAS3", "WEGE3"]  # edite à vontade
INICIO = "2020-01-01"
TAXA_LIVRE_RISCO = 0.10  # a.a. (use algo próximo da Selic/CDI atual)
DIAS_UTEIS = 252


# ------------------------------------------------------ símbolos e busca
_PADRAO_B3 = re.compile(r"^[A-Z]{4}\d{1,2}$")  # PETR4, VALE3, BOVA11...


def normalizar_simbolo(simbolo: str) -> str:
    """Código no formato do Yahoo Finance.

    Códigos de ações da B3 digitados sem sufixo (PETR4) ganham o ".SA".
    Todo o resto é mantido como está: AAPL, BTC-USD, ^BVSP, EURUSD=X, PETR4.SA...
    """
    simbolo = simbolo.strip().upper()
    return simbolo + ".SA" if _PADRAO_B3.match(simbolo) else simbolo


TIPOS = {
    "EQUITY": "Ação",
    "ETF": "ETF",
    "MUTUALFUND": "Fundo",
    "INDEX": "Índice",
    "CURRENCY": "Câmbio",
    "CRYPTOCURRENCY": "Cripto",
    "FUTURE": "Futuro",
}


def buscar_ativos(consulta: str, limite: int = 10) -> list[dict]:
    """Pesquisa ativos no Yahoo Finance por nome ou código."""
    try:
        quotes = yf.Search(consulta, max_results=limite, news_count=0).quotes
    except Exception:  # yfinance antigo ou falha da classe: usa o endpoint direto
        import requests

        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": consulta, "quotesCount": limite, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        r.raise_for_status()
        quotes = r.json().get("quotes", [])

    achados = []
    for q in quotes:
        if not q.get("symbol") or q.get("quoteType") == "OPTION":
            continue
        achados.append(
            {
                "simbolo": q["symbol"],
                "nome": q.get("longname") or q.get("shortname") or q["symbol"],
                "tipo": TIPOS.get(q.get("quoteType"), q.get("typeDisp") or "Ativo"),
                "bolsa": q.get("exchDisp") or q.get("exchange") or "",
            }
        )
    return achados


SIMBOLOS_MOEDA = {"BRL": "R$", "USD": "US$", "EUR": "€", "GBP": "£", "JPY": "¥"}


def moeda_do_ativo(simbolo: str) -> str:
    """Código da moeda em que o ativo é cotado (BRL, USD...)."""
    try:
        return str(yf.Ticker(simbolo).fast_info["currency"]).upper()
    except Exception:
        return "BRL" if simbolo.endswith(".SA") else ""


def simbolo_moeda(codigo: str) -> str:
    return SIMBOLOS_MOEDA.get(codigo, codigo)


# ---------------------------------------------------------------- dados
# Intervalos do gráfico. O Yahoo Finance limita o histórico intradiário:
#   1 min ≈ 7 dias | 5, 15 e 30 min ≈ 60 dias | 1 h ≈ 2 anos.
# Não existe barra de 4 h pronta: ela é montada a partir de barras de 1 h,
# agrupadas a partir da abertura do pregão do próprio ativo (ex.: B3 10h-14h e
# 14h-18h; cripto, que negocia 24h, 00h-04h, 04h-08h...).
INTERVALOS = {
    "Diário":     {"yf": "1d",  "resample": None, "max_dias": None},
    "4 horas":    {"yf": "60m", "resample": "4h", "max_dias": 729},
    "30 minutos": {"yf": "30m", "resample": None, "max_dias": 59},
    "15 minutos": {"yf": "15m", "resample": None, "max_dias": 59},
    "5 minutos":  {"yf": "5m",  "resample": None, "max_dias": 59},
    "1 minuto":   {"yf": "1m",  "resample": None, "max_dias": 6},
}


def _inicio_efetivo(inicio, cfg) -> pd.Timestamp:
    """Aplica o limite de histórico do Yahoo nos intervalos intradiários."""
    start = pd.Timestamp(inicio)
    if cfg["max_dias"]:
        limite = pd.Timestamp.now().normalize() - pd.Timedelta(days=cfg["max_dias"])
        start = max(start, limite)
    return start


def _ajustar_indice(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Intradiário: mantém o horário local da bolsa do ativo, sem o fuso no índice."""
    if cfg["max_dias"] and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _reamostrar(df: pd.DataFrame, cfg, agregacao=None) -> pd.DataFrame:
    """Agrupa barras (ex.: 1 h -> 4 h) a partir da abertura do pregão do ativo."""
    if df.empty:
        return df
    grupos = df.resample(cfg["resample"], offset=f"{int(df.index.hour.min())}h")
    return grupos.last() if agregacao is None else grupos.agg(agregacao)


def baixar_precos(tickers, inicio=INICIO, intervalo: str = "Diário") -> pd.DataFrame:
    """Baixa preços de fechamento ajustados no intervalo escolhido."""
    cfg = INTERVALOS[intervalo]
    simbolos = [normalizar_simbolo(t) for t in tickers]
    df = yf.download(
        simbolos,
        start=_inicio_efetivo(inicio, cfg),
        interval=cfg["yf"],
        auto_adjust=True,
        progress=False,
    )["Close"]
    df = _ajustar_indice(df, cfg)
    if cfg["resample"]:
        df = _reamostrar(df, cfg)
    return df.dropna(how="all")


AGREGACAO_OHLC = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


def baixar_ohlc(ticker: str, inicio=INICIO, intervalo: str = "Diário") -> pd.DataFrame:
    """Candles (Open, High, Low, Close, Volume) de um ativo no intervalo escolhido."""
    cfg = INTERVALOS[intervalo]
    df = yf.download(
        normalizar_simbolo(ticker),
        start=_inicio_efetivo(inicio, cfg),
        interval=cfg["yf"],
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):  # versões novas do yfinance
        df.columns = df.columns.get_level_values(0)
    df = _ajustar_indice(df[list(AGREGACAO_OHLC)], cfg)
    if cfg["resample"]:
        df = _reamostrar(df, cfg, AGREGACAO_OHLC)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


# ----------------------------------------------------------- indicadores
def rsi(preco: pd.Series, janela: int = 14) -> pd.Series:
    delta = preco.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / janela, adjust=False).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / janela, adjust=False).mean()
    return 100 - 100 / (1 + ganho / perda)


def indicadores(preco: pd.Series) -> pd.DataFrame:
    """Indicadores técnicos de um único ativo."""
    out = pd.DataFrame({"preco": preco})
    out["mm20"] = preco.rolling(20).mean()
    out["mm50"] = preco.rolling(50).mean()
    out["mm200"] = preco.rolling(200).mean()
    out["rsi14"] = rsi(preco)

    desvio = preco.rolling(20).std()
    out["boll_sup"] = out["mm20"] + 2 * desvio
    out["boll_inf"] = out["mm20"] - 2 * desvio
    return out


# --------------------------------------------------------------- risco
def max_drawdown(preco: pd.Series) -> float:
    pico = preco.cummax()
    return float((preco / pico - 1).min())


def metricas_risco(
    precos: pd.DataFrame, taxa_livre_risco: float = TAXA_LIVRE_RISCO
) -> pd.DataFrame:
    """Retorno, volatilidade, Sharpe e drawdown por ativo (cada um no seu período)."""
    linhas = {}
    for t in precos.columns:
        p = precos[t].dropna()
        anos = max((p.index[-1] - p.index[0]).days / 365.25, 1e-9)
        ret_anual = (p.iloc[-1] / p.iloc[0]) ** (1 / anos) - 1
        vol_anual = p.pct_change().std() * np.sqrt(DIAS_UTEIS)
        linhas[t] = {
            "retorno_a.a.": ret_anual,
            "volatilidade_a.a.": vol_anual,
            "sharpe": (ret_anual - taxa_livre_risco) / vol_anual,
            "max_drawdown": max_drawdown(p),
        }
    return pd.DataFrame(linhas).T.round(3)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    precos = baixar_precos(TICKERS)
    print("\n=== Métricas de risco/retorno ===")
    print(metricas_risco(precos))

    print("\n=== Correlação entre retornos diários ===")
    print(precos.pct_change(fill_method=None).corr().round(2))

    print("\n=== Último sinal técnico por ativo ===")
    for t in precos.columns:
        ind = indicadores(precos[t].dropna()).iloc[-1]
        tendencia = "alta" if ind["preco"] > ind["mm200"] else "baixa"
        print(
            f"{t}: preço {ind['preco']:.2f} | RSI {ind['rsi14']:.0f} | "
            f"tendência longa: {tendencia}"
        )
