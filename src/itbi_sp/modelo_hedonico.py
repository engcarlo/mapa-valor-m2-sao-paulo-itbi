"""
modelo_hedonico.py

Modelo hedônico de valor de apartamento usando distrito (geográfico
preciso, via geo_distrito.py) + padrão construtivo (IPTU) + área
construída, sobre valores já corrigidos temporalmente (temporal_ivgr.py).

Achado central desta etapa: distrito preciso (ponto-em-polígono) + amostra
multi-ano (377 mil registros, 2019-2026) supera claramente qualquer
configuração testada com CEP/bairro como texto e uma amostra de um ano só:

    área + bairro/CEP texto, 1 ano (2022)     R² ~ 0,58-0,64
    área + distrito preciso + padrão, 8 anos  R² = 0,624, erro 20,3%

A comparação não é direta (bases diferentes), mas confirma que precisão
geográfica e volume amostral pesam mais que a escolha do algoritmo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from .config import (PADROES_RESIDENCIAL_VERTICAL, RANDOM_STATE, ROTULOS_PADRAO,  # noqa: F401
                     VALOR_MINIMO_TRANSACAO)
from .carregamento import normalizar_padrao



def preparar_base_modelo(df: pd.DataFrame, col_area: str, col_valor_corrigido: str,
                          col_distrito: str, col_padrao: str,
                          padroes_validos: list[str] = PADROES_RESIDENCIAL_VERTICAL,
                          valor_minimo: float = VALOR_MINIMO_TRANSACAO) -> pd.DataFrame:
    """Filtra para os padrões residencial-vertical de interesse, remove
    valor simbólico e distrito ausente, e deriva log_area/log_valor.
    """
    d = df.dropna(subset=[col_distrito]).copy()
    # normaliza Padrão pra string limpa independente de vir como int ou float
    # (achado real: em 2010/2012 o Excel guarda o campo como float, gerando
    # "22.0" em vez de "22" — sem isso, esses anos somem silenciosamente do
    # filtro de padroes_validos)
    d[col_padrao] = normalizar_padrao(d[col_padrao])
    d = d[d[col_padrao].isin(padroes_validos)]
    d = d[d[col_valor_corrigido] >= valor_minimo]
    d['log_area'] = np.log(d[col_area])
    d['log_valor'] = np.log(d[col_valor_corrigido])
    d['padrao_grp'] = d[col_padrao]
    d['distrito'] = d[col_distrito]
    return d


def treinar_modelo_hedonico(d: pd.DataFrame, test_size: float = 0.2,
                             random_state: int = RANDOM_STATE) -> tuple[Pipeline, dict]:
    """Treina o HuberRegressor (log_valor ~ log_area + distrito + padrao_grp)
    com split treino/teste, e retorna (pipeline treinado, métricas).
    O pipeline treinado é reaproveitado por `gerar_grade_precos` para
    prever valores em qualquer combinação distrito × padrão.
    """
    X = d[['log_area', 'distrito', 'padrao_grp']]
    y = d['log_valor']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['distrito', 'padrao_grp'])],
                             remainder='passthrough')
    pipe = Pipeline([('pre', pre), ('reg', HuberRegressor(epsilon=1.35, max_iter=3000))])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    erro_pct = float(np.median(np.abs((np.exp(pred) - np.exp(y_test)) / np.exp(y_test))) * 100)

    metricas = {'r2_log': round(r2, 3), 'erro_pct_mediano': round(erro_pct, 1), 'n_treino': len(X_train), 'n_teste': len(X_test)}
    return pipe, metricas


def gerar_grade_precos(pipe: Pipeline, d: pd.DataFrame, col_area: str,
                        padroes: list[str] = PADROES_RESIDENCIAL_VERTICAL) -> pd.DataFrame:
    """Gera a grade distrito × padrão de preço/m² estimado, usando a área
    mediana da base como referência (isola o efeito de distrito+padrão do
    efeito de área). Inclui o tamanho da amostra real por célula — células
    com n=0 são extrapolação pura do modelo, não valor observado, e devem
    ser sinalizadas como tal em qualquer apresentação (ver gerar_mapa_html.py).
    """
    area_ref = d[col_area].median()
    distritos = sorted(d['distrito'].unique())

    linhas = []
    for dist in distritos:
        for pad in padroes:
            n_amostra = len(d[(d['distrito'] == dist) & (d['padrao_grp'] == pad)])
            Xp = pd.DataFrame({'log_area': [np.log(area_ref)], 'distrito': [dist], 'padrao_grp': [pad]})
            pred_log = pipe.predict(Xp)[0]
            preco_m2 = np.exp(pred_log) / area_ref
            linhas.append({'distrito': dist, 'padrao': pad, 'preco_m2_estimado': preco_m2, 'n_amostra': n_amostra})

    return pd.DataFrame(linhas)
