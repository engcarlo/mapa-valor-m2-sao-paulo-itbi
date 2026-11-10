"""comum.py: utilitários genéricos compartilhados (terreno e apartamento).

Constantes de coluna vivem em ``config.py`` e são reexportadas aqui por
compatibilidade com os módulos de terreno.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import *  # noqa: F401,F403  (constantes de coluna)
from .config import (COL_AREA_TERRENO, COL_BAIRRO, COL_CEP, COL_DATA_TRANSACAO,  # noqa: F401
                     COL_SQL, COL_VALOR_TRANSACAO)

MESES_ABAS_PADRAO = [
    "JAN-2022", "FEV-2022", "MAR-2022", "ABR-2022", "MAI-2022", "JUN-2022",
    "JUL-2022", "AGO-2022", "SET-2022", "OUT-2022", "NOV-2022", "DEZ-2022",
]


def carregar_itbi(caminho_xlsx: str, abas: list[str] | None = None) -> pd.DataFrame:
    """Lê e concatena as abas mensais de um arquivo de guias de ITBI da
    Prefeitura de SP. Adiciona a coluna 'mes_planilha' identificando a aba
    de origem de cada linha.
    """
    abas = abas or MESES_ABAS_PADRAO
    xl = pd.ExcelFile(caminho_xlsx)
    dfs = []
    for aba in abas:
        d = xl.parse(aba)
        d["mes_planilha"] = aba
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def filtrar_sql_unico(df: pd.DataFrame, limite: int = 1) -> pd.DataFrame:
    """Mantém apenas registros cujo N° de Cadastro (SQL) aparece no máximo
    `limite` vezes no dataframe. Uso típico: `limite=1` para terreno (onde a
    repetição quase sempre indica unidade de empreendimento não
    individualizado). Para categorias com pouca contaminação (ex.
    apartamento), pode ser usado apenas como diagnóstico, sem aplicar como
    filtro obrigatório — ver `relatorio_qualidade`.
    """
    contagem = df[COL_SQL].value_counts()
    sqls_ok = contagem[contagem <= limite].index
    return df[df[COL_SQL].isin(sqls_ok)].copy()


def separar_por_ano_transacao(df: pd.DataFrame, ano: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa registros cuja Data de Transação cai dentro do `ano` desejado
    dos que caem fora ("regularização tardia" — negócio antigo registrado/
    pago no ano corrente). Retorna (dentro, fora).
    """
    ano_transacao = df[COL_DATA_TRANSACAO].dt.year
    dentro = df[ano_transacao == ano].copy()
    fora = df[ano_transacao != ano].copy()
    return dentro, fora


def calcular_preco_m2(df: pd.DataFrame, col_area: str) -> pd.Series:
    """Preço por m² = Valor de Transação / área (informar a coluna de área
    apropriada: COL_AREA_TERRENO para terreno, COL_AREA_CONSTRUIDA para
    apartamento/casa)."""
    return df[COL_VALOR_TRANSACAO] / df[col_area]


def relatorio_qualidade(df: pd.DataFrame, col_area: str, ano: int = 2022) -> dict:
    """Relatório rápido de qualidade de uma base já filtrada por uso +
    natureza de transação, ANTES de qualquer depuração adicional. Útil para
    decidir, para uma nova categoria de imóvel, o quanto de depuração é
    necessário (ver comparação terreno x casa x apartamento na análise
    original).
    """
    contagem_sql = df[COL_SQL].value_counts()
    pct_sql_unico = (contagem_sql == 1).sum() / len(df) * 100  # SQLs que aparecem 1 vez / linhas
    ano_transacao = df[COL_DATA_TRANSACAO].dt.year
    return {
        "n_registros": len(df),
        "pct_sql_unico": round(pct_sql_unico, 1),
        "pct_bairro_ausente": round(df[COL_BAIRRO].isna().mean() * 100, 1),
        "pct_cep_ausente": round(df[COL_CEP].isna().mean() * 100, 1),
        "pct_fora_do_ano": round((ano_transacao != ano).mean() * 100, 2),
        "preco_m2_mediano": round(float((df[COL_VALOR_TRANSACAO] / df[col_area]).median()), 2),
    }


# --- Treino/avaliação de modelo (genérico, independe do tipo de imóvel) ----

MODELOS_PADRAO = {
    "Huber (robusta)": HuberRegressor(epsilon=1.35, max_iter=2000),
}


def treinar_avaliar(df: pd.DataFrame, features_num: list[str], features_cat: list[str],
                     alvo: str = "log_valor", modelos: dict | None = None,
                     test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Treina cada modelo em `modelos` usando `features_num` + `features_cat`
    (one-hot) para prever `alvo`, com split treino/teste. Retorna
    DataFrame-resumo com R² (escala log) e erro percentual mediano absoluto.
    """
    modelos = modelos or MODELOS_PADRAO
    cols = features_num + features_cat
    X = df[cols]
    y = df[alvo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)],
        remainder="passthrough",
    ) if features_cat else "passthrough"

    resultados = []
    for nome, estimador in modelos.items():
        pipe = Pipeline([("pre", pre), ("reg", clone(estimador))])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        r2 = r2_score(y_test, pred)
        erro_pct = np.median(np.abs((np.exp(pred) - np.exp(y_test)) / np.exp(y_test))) * 100
        resultados.append({"modelo": nome, "r2_log": round(r2, 3), "erro_pct_mediano": round(erro_pct, 1)})

    return pd.DataFrame(resultados)


def comparar_configuracoes(df: pd.DataFrame, configuracoes: dict[str, tuple[list[str], list[str]]],
                            alvo: str = "log_valor", modelo=None, test_size: float = 0.2,
                            random_state: int = 42) -> pd.DataFrame:
    """Roda `treinar_avaliar` para várias combinações de features de uma vez.
    `configuracoes` é {nome_da_config: (features_num, features_cat)}.
    """
    modelo = modelo or HuberRegressor(epsilon=1.35, max_iter=2000)
    linhas = []
    for nome_config, (f_num, f_cat) in configuracoes.items():
        resumo = treinar_avaliar(
            df, f_num, f_cat, alvo=alvo,
            modelos={"modelo": modelo}, test_size=test_size, random_state=random_state,
        )
        linhas.append({
            "configuracao": nome_config,
            "r2_log": resumo.loc[0, "r2_log"],
            "erro_pct_mediano": resumo.loc[0, "erro_pct_mediano"],
        })
    return pd.DataFrame(linhas)


def agrupar_categoria_rara(serie: pd.Series, min_obs: int = 15, rotulo_ausente: str = "DESCONHECIDO",
                            rotulo_outros: str = "OUTROS") -> pd.Series:
    """Agrupa uma coluna categórica (ex. Bairro, CEP, Padrão) em categorias
    com pelo menos `min_obs` observações, jogando o resto em `rotulo_outros`
    e valores ausentes em `rotulo_ausente`. Evita explosão de dummies em
    categorias raras ao montar features para regressão.
    """
    ajustada = serie.fillna(rotulo_ausente)
    contagem = ajustada.value_counts()
    ok = set(contagem[contagem >= min_obs].index)
    return ajustada.where(ajustada.isin(ok), rotulo_outros)
