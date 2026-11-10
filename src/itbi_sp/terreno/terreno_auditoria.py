"""
terreno_auditoria.py

Funções para identificar candidatos a transação não-mercado (subdeclaração,
erro de digitação, valor simbólico) dentro de uma base já depurada de
terrenos (ver terreno_depuracao.py).

Achado central: um valor declarado anormalmente baixo não implica, por si
só, perda de arrecadação — o município pode já cobrar o ITBI sobre o Valor
Venal de Referência quando ele excede o valor declarado ("piso fiscal").
Sempre rodar `piso_fiscal_protegeu` antes de classificar um caso como
candidato a fraude/sonegação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor

from ..comum import COL_VALOR_TRANSACAO, COL_AREA_TERRENO, COL_BASE_CALCULO, COL_VALOR_VENAL_REF, COL_BAIRRO


def _features_log(df: pd.DataFrame, col_area: str = COL_AREA_TERRENO, col_preco_m2: str = "preco_m2",
                   piso_preco: float = 1e-3) -> np.ndarray:
    log_area = np.log(df[col_area])
    log_preco = np.log(df[col_preco_m2].clip(lower=piso_preco))
    X = np.column_stack([log_area, log_preco])
    return StandardScaler().fit_transform(X)


def clusterizar_kmeans(df: pd.DataFrame, k: int = 3, col_area: str = COL_AREA_TERRENO,
                        col_preco_m2: str = "preco_m2", random_state: int = 42) -> pd.DataFrame:
    """K-means sobre log(área) e log(preço/m²) padronizados. Em terreno de
    SP/2022, k=3 isolou: lotes urbanos padrão, lotes maiores/periféricos, e
    um cluster de valor simbólico (candidatos a não-mercado)."""
    Xs = _features_log(df, col_area, col_preco_m2)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(Xs)
    out = df.copy()
    out["cluster_kmeans"] = km.labels_
    return out


def detectar_ruido_dbscan(df: pd.DataFrame, eps: float = 0.3, min_samples: int = 5,
                           col_area: str = COL_AREA_TERRENO, col_preco_m2: str = "preco_m2") -> pd.DataFrame:
    """DBSCAN sobre log(área) e log(preço/m²) padronizados — mais adequado
    que K-means para achar anomalias (pontos "densos demais para se
    agrupar" viram ruído, rótulo -1)."""
    Xs = _features_log(df, col_area, col_preco_m2)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
    out = df.copy()
    out["ruido_dbscan"] = db.labels_ == -1
    return out


def calcular_score_suspeicao(df: pd.DataFrame, col_bairro: str = COL_BAIRRO,
                              col_area: str = COL_AREA_TERRENO, col_valor: str = COL_VALOR_TRANSACAO,
                              min_obs_bairro: int = 8) -> pd.DataFrame:
    """Regressão robusta (HuberRegressor) de log(valor) em função de
    log(área) e bairro. Retorna cópia de `df` com log_valor_esperado,
    residuo, z_score e razao_atual_esperado. z_score muito negativo ->
    candidato a subdeclaração; muito positivo -> candidato a erro de
    digitação/outlier real.
    """
    d = df.copy()
    bairro_ajustado = d[col_bairro].fillna("DESCONHECIDO")
    contagem = bairro_ajustado.value_counts()
    categorias_ok = set(contagem[contagem >= min_obs_bairro].index)
    d["_bairro_grp"] = bairro_ajustado.where(bairro_ajustado.isin(categorias_ok), "OUTROS")

    d["_log_area"] = np.log(d[col_area])
    d["_log_valor"] = np.log(d[col_valor].clip(lower=0.01))

    X = d[["_log_area", "_bairro_grp"]]
    y = d["_log_valor"]
    pre = ColumnTransformer(
        [("bairro", OneHotEncoder(handle_unknown="ignore"), ["_bairro_grp"])],
        remainder="passthrough",
    )
    pipe = Pipeline([("pre", pre), ("reg", HuberRegressor(epsilon=1.35, max_iter=500))])
    pipe.fit(X, y)

    d["log_valor_esperado"] = pipe.predict(X)
    d["residuo"] = d["_log_valor"] - d["log_valor_esperado"]
    d["z_score"] = d["residuo"] / d["residuo"].std()
    d["razao_atual_esperado"] = np.exp(d["residuo"])

    return d.drop(columns=["_bairro_grp", "_log_area", "_log_valor"])


def piso_fiscal_protegeu(df: pd.DataFrame, col_base_calculo: str = COL_BASE_CALCULO,
                          col_valor_venal_ref: str = COL_VALOR_VENAL_REF,
                          tolerancia_relativa: float = 0.001) -> pd.Series:
    """Verifica se a Base de Cálculo adotada já corresponde ao Valor Venal
    de Referência — ou seja, se o piso fiscal já neutralizou um valor
    declarado artificialmente baixo. Rodar SEMPRE antes de reportar um
    registro como "possível sonegação" por valor baixo.
    """
    diff_relativa = (df[col_base_calculo] - df[col_valor_venal_ref]).abs() / df[col_valor_venal_ref]
    return diff_relativa < tolerancia_relativa


def montar_lista_candidatos_auditoria(df_com_score: pd.DataFrame, z_threshold: float = -2.0) -> pd.DataFrame:
    """Filtra candidatos com z_score <= z_threshold e adiciona
    'protegido_por_piso_fiscal'. Ordena do mais suspeito para o menos.
    """
    candidatos = df_com_score[df_com_score["z_score"] <= z_threshold].copy()
    candidatos["protegido_por_piso_fiscal"] = piso_fiscal_protegeu(candidatos)
    return candidatos.sort_values("z_score")
