"""
terreno_modelo.py

Preparação de features para o baseline de previsão de valor de terreno.
Treino/avaliação de modelo (genérico) vem de comum.py.

Achado central: com as features disponíveis no ITBI puro (área do terreno +
localização categórica grosseira via bairro/CEP), o teto de desempenho
observado foi R² ~ 0,20-0,26 — o gargalo é a falta de geolocalização real e
de variáveis de zoneamento, não a escolha do algoritmo.
"""

import numpy as np
import pandas as pd

from ..comum import COL_VALOR_TRANSACAO, COL_AREA_TERRENO, COL_TESTADA, COL_BAIRRO, COL_CEP, agrupar_categoria_rara


def preparar_features(df: pd.DataFrame, col_area: str = COL_AREA_TERRENO, col_valor: str = COL_VALOR_TRANSACAO,
                       col_testada: str = COL_TESTADA, col_bairro: str = COL_BAIRRO, col_cep: str = COL_CEP,
                       min_obs_categoria: int = 8) -> pd.DataFrame:
    """Deriva log_area, log_valor, log_testada + tem_testada, bairro_grp e
    cep5_grp (5 primeiros dígitos do CEP). Retorna cópia de `df`.
    """
    d = df.copy()
    d["log_area"] = np.log(d[col_area])
    d["log_valor"] = np.log(d[col_valor])

    d["tem_testada"] = (d[col_testada] > 0).astype(int)
    d["log_testada"] = np.log(d[col_testada].clip(lower=1))

    d["bairro_grp"] = agrupar_categoria_rara(d[col_bairro], min_obs=min_obs_categoria)
    cep5 = d[col_cep].astype(str).str[:5]
    d["cep5_grp"] = agrupar_categoria_rara(cep5, min_obs=min_obs_categoria)

    return d
