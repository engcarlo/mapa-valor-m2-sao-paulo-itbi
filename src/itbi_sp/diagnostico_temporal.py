"""
diagnostico_temporal.py

Diagnóstico que motivou a decisão de manter o modelo de produção (mapa)
treinado só em 2019-2026, mesmo com 21 anos de dado disponível (2006-2026).

ACHADO: agrupar (pool) muitos anos numa única regressão distrito+padrão+área
PIOROU o modelo (R²=0,566, 21 anos) frente a usar só o período recente
(R²=0,629, 2019-2026) — mesmo depois de já ter corrigido os valores pela
inflação de mercado via IVG-R.

Adicionar 'ano' como controle explícito recupera parte da perda (R²=0,584),
mas não tudo. E cada ERA testada isoladamente prediz bem por conta própria:

    2006-2010 (boom inicial)      R²=0,623
    2011-2015 (pico + crise)      R²=0,484  <- pior mesmo sozinha: período de virada estrutural
    2016-2018 (recessão)          R²=0,535
    2019-2026 (atual)             R²=0,629

CONCLUSÃO: o problema não é qualidade do dado antigo (cada era prediz tão
bem quanto as outras isoladamente) — é que a estrutura de preço relativo
entre distritos MUDA ao longo de duas décadas (gentrificação, novas linhas
de metrô, valorização desigual), e um efeito de distrito fixo (aditivo, não
variante no tempo) não consegue representar isso quando os períodos são
agrupados. Um modelo raso (área+distrito+padrão) treinado sobre 20 anos
mistura relações que já não são as mesmas.

IMPLICAÇÃO PRÁTICA: mais dado não é automaticamente melhor para este tipo
de modelo. O próximo passo defensável não é só "somar mais anos" — é
modelar o efeito de distrito como variante no tempo (interação distrito×
período, ou uma feature temporal contínua além da correção IVG-R) antes de
voltar a usar o histórico completo na produção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from .config import RANDOM_STATE


def _r2_erro(d: pd.DataFrame, features_cat: list[str], test_size=0.2, random_state=RANDOM_STATE) -> dict:
    X = d[['log_area'] + features_cat]
    y = d['log_valor']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)], remainder='passthrough')
    pipe = Pipeline([('pre', pre), ('reg', HuberRegressor(epsilon=1.35, max_iter=2000))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    erro = float(np.median(np.abs((np.exp(pred) - np.exp(y_test)) / np.exp(y_test))) * 100)
    return {'n': len(d), 'r2_log': round(r2, 3), 'erro_pct_mediano': round(erro, 1)}


def diagnostico_por_era(base_modelo: pd.DataFrame, col_ano: str = 'ano_arquivo',
                         eras: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Treina o mesmo modelo (área+distrito+padrão) separadamente em cada
    era e compara R²/erro. Usa isso para checar se um resultado ruim ao
    agrupar vários anos vem de dado antigo ruim (cada era teria R² baixo
    isoladamente) ou de mudança estrutural entre eras (cada era prediz bem
    sozinha, mas o pool geral piora).
    """
    eras = eras or {
        'inicial': (2006, 2010), 'pico_crise': (2011, 2015),
        'recessao': (2016, 2018), 'atual': (2019, 2026),
    }
    linhas = []
    for nome, (ini, fim) in eras.items():
        sub = base_modelo[(base_modelo[col_ano] >= ini) & (base_modelo[col_ano] <= fim)]
        if len(sub) < 100:
            continue
        metricas = _r2_erro(sub, ['distrito', 'padrao_grp'])
        linhas.append({'era': nome, 'periodo': f'{ini}-{fim}', **metricas})
    return pd.DataFrame(linhas)


def comparar_pool_vs_era_unica(base_modelo: pd.DataFrame, col_ano: str = 'ano_arquivo',
                                ano_corte: int = 2019) -> pd.DataFrame:
    """Compara: (a) modelo treinado em tudo, (b) modelo treinado em tudo +
    'ano' como controle explícito, (c) modelo treinado só no período mais
    recente (>= ano_corte). Reproduz a comparação que motivou manter o
    modelo de produção restrito a 2019+.
    """
    base_modelo = base_modelo.copy()
    base_modelo['ano_grp'] = base_modelo[col_ano].astype(str)

    linhas = []
    linhas.append({'configuracao': 'pool completo (sem controle de ano)',
                    **_r2_erro(base_modelo, ['distrito', 'padrao_grp'])})
    linhas.append({'configuracao': 'pool completo + ano como controle',
                    **_r2_erro(base_modelo, ['distrito', 'padrao_grp', 'ano_grp'])})
    recente = base_modelo[base_modelo[col_ano] >= ano_corte]
    linhas.append({'configuracao': f'só período recente (>= {ano_corte})',
                    **_r2_erro(recente, ['distrito', 'padrao_grp'])})
    return pd.DataFrame(linhas)
