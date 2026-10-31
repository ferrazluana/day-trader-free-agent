"""
Dashboard Streamlit para análise de qualquer ativo do Yahoo Finance.

Instalação:
    pip install streamlit plotly yfinance pandas numpy anthropic requests
    export ANTHROPIC_API_KEY="sua-chave"   # OPCIONAL: só para resumo/sentimento por IA

Uso (na mesma pasta do analise_b3.py):
    streamlit run dashboard_b3.py
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analise_b3 import (
    INTERVALOS,
    TAXA_LIVRE_RISCO,
    baixar_ohlc,
    baixar_precos,
    buscar_ativos,
    indicadores,
    metricas_risco,
    moeda_do_ativo,
    normalizar_simbolo,
    simbolo_moeda,
)
from noticias_b3 import analisar_sentimento, coletar

st.set_page_config(page_title="Análise de ativos", layout="wide")
st.title("📈 Análise de ativos")


# ------------------------------------------------------------- dados
@st.cache_data(ttl=300, show_spinner="Baixando preços...")
def carregar(tickers: tuple, inicio: str, intervalo: str = "Diário") -> pd.DataFrame:
    return baixar_precos(list(tickers), inicio, intervalo)


@st.cache_data(ttl=300, show_spinner="Baixando candles...")
def carregar_ohlc(ticker: str, inicio: str, intervalo: str) -> pd.DataFrame:
    return baixar_ohlc(ticker, inicio, intervalo)


@st.cache_data(ttl=600, show_spinner=False)
def buscar(consulta: str) -> list:
    return buscar_ativos(consulta)


@st.cache_data(ttl=86400, show_spinner=False)
def carregar_moeda(simbolo: str) -> str:
    return moeda_do_ativo(simbolo)


# ----------------------------------------------------------- sidebar
PADRAO = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "WEGE3.SA"]
if "ativos" not in st.session_state:
    st.session_state["ativos"] = list(PADRAO)       # lista de opções
    st.session_state["ativos_sel"] = list(PADRAO)   # ativos marcados na análise
    st.session_state["nomes"] = {}                  # símbolo -> nome (vem da busca)


def adicionar_ativo(simbolo: str, nome: str | None = None):
    if simbolo not in st.session_state["ativos"]:
        st.session_state["ativos"].append(simbolo)
    if simbolo not in st.session_state["ativos_sel"]:
        st.session_state["ativos_sel"] = st.session_state["ativos_sel"] + [simbolo]
    if nome:
        st.session_state["nomes"][simbolo] = nome


st.sidebar.header("Ativos")
consulta = st.sidebar.text_input(
    "🔎 Buscar qualquer ativo do Yahoo Finance",
    placeholder="Apple, bitcoin, IBOV, EURUSD, PETR4...",
)
if consulta.strip():
    try:
        achados = buscar(consulta.strip())
    except Exception as e:
        achados = []
        st.sidebar.warning(f"A busca do Yahoo falhou ({e}). Você ainda pode adicionar o código direto.")

    if achados:
        rotulos = {
            f"{a['simbolo']} — {a['nome']} ({a['tipo']}, {a['bolsa']})": a for a in achados
        }
        escolha = rotulos[st.sidebar.selectbox("Resultados", list(rotulos))]
        st.sidebar.button(
            "➕ Adicionar à lista",
            on_click=adicionar_ativo,
            args=(escolha["simbolo"], escolha["nome"]),
        )
    else:
        st.sidebar.caption("Nenhum resultado na busca do Yahoo.")

    direto = normalizar_simbolo(consulta)
    st.sidebar.button(
        f"Adicionar o código {direto} direto",
        on_click=adicionar_ativo,
        args=(direto,),
        key="adicionar_direto",
    )

tickers = tuple(
    st.sidebar.multiselect(
        "Ativos na análise", options=st.session_state["ativos"], key="ativos_sel"
    )
)

st.sidebar.header("Configurações")
inicio = st.sidebar.date_input("Data inicial", pd.Timestamp("2022-01-01"))
taxa_rf = st.sidebar.number_input(
    "Taxa livre de risco (% a.a.)",
    min_value=0.0,
    value=TAXA_LIVRE_RISCO * 100,
    step=0.5,
    help="Usada no Sharpe. Para ativos em reais, algo próximo do CDI; em dólar, da taxa dos EUA.",
) / 100

if not tickers:
    st.info("Busque e adicione ao menos um ativo na barra lateral.")
    st.stop()

try:
    precos = carregar(tickers, str(inicio))
except Exception as e:  # falha de rede, ticker inválido etc.
    st.error(f"Não consegui baixar os dados: {e}")
    st.stop()

# Tickers que o Yahoo não devolveu (ou devolveu vazios) são removidos e avisados,
# em vez de quebrar os cálculos das abas Comparação e Risco.
sem_dados = [t for t in tickers if t not in precos.columns or precos[t].dropna().empty]
precos = precos.drop(columns=[t for t in sem_dados if t in precos.columns])
if sem_dados:
    st.warning(
        "Sem dados do Yahoo Finance para: " + ", ".join(sem_dados) + ". "
        "Confira o código no formato do Yahoo (ex.: PETR4.SA, AAPL, BTC-USD, ^BVSP). Se estiver certo, pode ser "
        "uma falha temporária do Yahoo: aguarde alguns minutos e recarregue."
    )

if precos.empty:
    st.error("Nenhum dado retornado para os ativos informados. Confira os códigos e a data inicial.")
    st.stop()

ativo = st.sidebar.selectbox("Ativo para análise detalhada", list(precos.columns))
intervalo = st.sidebar.selectbox(
    "Intervalo do gráfico", list(INTERVALOS), index=0,
    help="Vale só para a aba Ativo. Comparação e Risco usam dados diários.",
)
mostrar_mm = st.sidebar.multiselect(
    "Médias móveis", ["mm20", "mm50", "mm200"], default=["mm20", "mm50"]
)
mostrar_boll = st.sidebar.checkbox("Bandas de Bollinger", value=False)
tipo_grafico = st.sidebar.radio("Tipo de gráfico", ["Candles", "Linha"], horizontal=True)

aba1, aba2, aba3, aba4 = st.tabs(["Ativo", "Comparação", "Risco", "Notícias e IA"])

# --------------------------------------------------- aba 1: ativo
with aba1:
    try:
        ohlc = carregar_ohlc(ativo, str(inicio), intervalo)
        preco_graf = ohlc["Close"]
    except Exception as e:
        preco_graf = None
        st.error(f"Não consegui baixar o intervalo '{intervalo}': {e}")

    if preco_graf is not None and preco_graf.empty:
        st.warning(
            "Sem dados para este intervalo (mercado fechado, ativo sem negócios "
            "ou fora do limite de histórico do Yahoo)."
        )
    elif preco_graf is not None:
        ind = indicadores(preco_graf)
        sim = simbolo_moeda(carregar_moeda(ativo))

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.56, 0.2, 0.24],
            vertical_spacing=0.03,
            subplot_titles=(
                f"Preço ({intervalo})",
                f"Volume financeiro ({sim})",
                "RSI (14)",
            ),
        )
        if tipo_grafico == "Candles":
            fig.add_trace(
                go.Candlestick(
                    x=ohlc.index,
                    open=ohlc["Open"],
                    high=ohlc["High"],
                    low=ohlc["Low"],
                    close=ohlc["Close"],
                    name="Preço",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(go.Scatter(x=ind.index, y=ind["preco"], name="Preço"), row=1, col=1)
        for mm in mostrar_mm:
            fig.add_trace(
                go.Scatter(x=ind.index, y=ind[mm], name=mm.upper(), line=dict(width=1)),
                row=1,
                col=1,
            )
        if mostrar_boll:
            for col, nome in [("boll_sup", "Boll. sup."), ("boll_inf", "Boll. inf.")]:
                fig.add_trace(
                    go.Scatter(
                        x=ind.index,
                        y=ind[col],
                        name=nome,
                        line=dict(width=1, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

        # Volume financeiro = quantidade negociada x preço típico da barra (na moeda do ativo).
        # É uma aproximação: o Yahoo não informa o valor financeiro exato.
        tipico = (ohlc["High"] + ohlc["Low"] + ohlc["Close"]) / 3
        vol_fin = ohlc["Volume"] * tipico
        cores = [
            "#26a69a" if fech >= abre else "#ef5350"
            for abre, fech in zip(ohlc["Open"], ohlc["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=ohlc.index,
                y=vol_fin,
                marker_color=cores,
                name="Volume financeiro",
                hovertemplate=f"{sim} " + "%{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(tickformat="~s", row=2, col=1)

        fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi14"], name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", row=3, col=1)

        # Esconde os "buracos" sem negociação, conforme o ativo: fim de semana
        # (não vale para cripto) e horário fora do pregão (não vale para ativos 24h).
        quebras = []
        if ind.index.dayofweek.max() < 5:
            quebras.append(dict(bounds=["sat", "mon"]))
        if intervalo != "Diário":
            horas = ind.index.hour.unique()
            if len(horas) < 20:
                quebras.append(
                    dict(bounds=[int(horas.max()) + 1, int(horas.min())], pattern="hour")
                )
        fig.update_xaxes(rangebreaks=quebras, rangeslider_visible=False)
        fig.update_layout(height=750, hovermode="x unified", margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if vol_fin.sum() == 0:
            st.caption("Este ativo não informa volume (comum em índices e câmbio).")

        ultimo = ind.iloc[-1]
        if pd.isna(ultimo["mm200"]):
            tendencia = "n/d"
        else:
            tendencia = "Alta" if ultimo["preco"] > ultimo["mm200"] else "Baixa"
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Último preço ({sim})", f"{ultimo['preco']:.2f}")
        c2.metric("RSI (14)", "n/d" if pd.isna(ultimo["rsi14"]) else f"{ultimo['rsi14']:.0f}")
        c3.metric(
            f"Tendência longa ({intervalo})",
            tendencia,
            help="Compara o preço com a média móvel de 200 barras do intervalo escolhido.",
        )

        if intervalo != "Diário":
            st.caption(
                "Limites do Yahoo Finance: 1 min ≈ 6 dias, 5/15/30 min ≈ 59 dias, "
                "4 h ≈ 2 anos (montada a partir de barras de 1 h, agrupadas desde a "
                "abertura do pregão do ativo). Os horários são os da bolsa do ativo. "
                "A data inicial da barra lateral é limitada a essas janelas."
            )

# ---------------------------------------------- aba 2: comparação
with aba2:
    base100 = precos / precos.apply(lambda s: s.dropna().iloc[0]) * 100
    fig2 = go.Figure()
    for t in base100.columns:
        fig2.add_trace(go.Scatter(x=base100.index, y=base100[t], name=t))
    fig2.update_layout(
        title="Desempenho relativo (base 100)",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------- aba 3: risco
with aba3:
    st.subheader("Métricas de risco e retorno")
    st.dataframe(metricas_risco(precos, taxa_rf), use_container_width=True)
    st.caption(
        "Retornos na moeda de cada ativo, sem converter câmbio. A volatilidade usa "
        "252 pregões por ano, o que subestima um pouco ativos que negociam todo dia (cripto)."
    )

    st.subheader("Correlação entre retornos diários")
    corr = precos.pct_change(fill_method=None).corr().round(2)
    fig3 = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=corr.values,
            texttemplate="%{text}",
        )
    )
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------- aba 4: notícias + IA
with aba4:
    st.subheader(f"Notícias e comunicados — {ativo}")
    chave = f"analise_{ativo}"
    EMOJI = {"positivo": "🟢", "neutro": "⚪", "negativo": "🔴"}
    tem_chave = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not tem_chave:
        st.info(
            "Sem ANTHROPIC_API_KEY: esta aba mostra só as notícias e os comunicados, "
            "sem resumo nem sentimento por IA. Para ativar a IA depois, basta definir "
            "a variável de ambiente e reiniciar o app."
        )

    if st.button("Buscar e analisar" if tem_chave else "Buscar notícias"):
        with st.spinner("Coletando notícias e comunicados..."):
            itens = coletar(ticker=ativo, nome=st.session_state["nomes"].get(ativo) or None)
        if itens.empty:
            st.session_state.pop(chave, None)
            st.warning("Nenhuma notícia ou comunicado encontrado agora.")
        else:
            resultado = None
            if tem_chave:
                try:
                    with st.spinner("Analisando com Claude..."):
                        resultado = analisar_sentimento(ativo, itens)
                except Exception as e:
                    st.error(f"Falha na análise com IA (mostrando só as notícias): {e}")
            st.session_state[chave] = (itens, resultado)

    if chave in st.session_state:
        itens, resultado = st.session_state[chave]
        tabela = itens.copy()
        colunas = ["data", "fonte", "titulo", "link"]

        if resultado:
            geral = resultado.get("sentimento_geral", "neutro")
            c1, c2 = st.columns(2)
            c1.metric("Sentimento geral", f"{EMOJI.get(geral, '⚪')} {geral.capitalize()}")
            c2.metric("Score (-1 a +1)", f"{float(resultado.get('score', 0)):+.2f}")
            st.write(resultado.get("resumo", ""))

            por_n = {i["n"]: i for i in resultado.get("itens", [])}
            numeros = range(1, len(tabela) + 1)
            tabela["sentimento"] = [
                EMOJI.get(por_n.get(n, {}).get("sentimento", "neutro"), "⚪") for n in numeros
            ]
            tabela["motivo"] = [por_n.get(n, {}).get("motivo", "") for n in numeros]
            colunas = ["data", "sentimento", "fonte", "titulo", "motivo", "link"]

        st.dataframe(
            tabela[colunas],
            column_config={"link": st.column_config.LinkColumn("link", display_text="abrir")},
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    "Dados do Yahoo Finance (podem ter atraso ou falhas). "
    "Ferramenta de apoio à análise, não é recomendação de investimento."
)
