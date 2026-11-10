"""Camada de Machine Learning: estimar o valor por classe de padrão construtivo.

Pergunta: dá para melhorar a estimativa do baseline hedônico (Huber aditivo:
``log_valor ~ log_area + distrito + padrão``) com um modelo que aprenda
*interações* (distrito x padrão x tempo) e use atributos extras (idade do
prédio, terreno, localização contínua)?

Decisões de projeto (documentadas para a seção de metodologia):

1. **Validação temporal, não aleatória.** O uso real é estimar o valor *hoje*
   a partir do passado. Um split aleatório mistura transações do mesmo prédio
   e mês entre treino e teste e infla a métrica. Treino em ``< data_corte``,
   teste em ``>= data_corte``.
2. **Alvo em log** (``log_valor``) com ``log_area`` como feature: o erro
   percentual do valor coincide com o do preço/m² (a área é conhecida).
3. **Métrica principal: erro percentual absoluto mediano (MdAPE)**, mais
   robusta que o RMSE a outliers, e reportada *por classe de padrão*, pois o
   erro médio global esconde classes onde o modelo é pior (20 e 25 são raras).
4. **Sem vazamento:** não entram ``Valor Financiado`` nem ``Base de Cálculo``
   (derivados do preço). ``Valor Venal de Referência`` é tratado como
   ablação opcional e rotulado como tal: é um valor fiscal conhecido antes da
   venda, mas torna o modelo uma calibração do valor fiscal.
5. **Incerteza:** intervalo de predição por *conformal split* separado por
   classe de padrão, com cobertura empírica medida no teste.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (COL_ACC, COL_AREA_CONSTRUIDA, COL_AREA_TERRENO, COL_FRACAO_IDEAL,
                     COL_PADRAO_IPTU, COL_VALOR_VENAL_REF, PADROES_RESIDENCIAL_VERTICAL,
                     RANDOM_STATE)

MES_ZERO = pd.Timestamp("2006-01-01")


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def montar_features(d: pd.DataFrame, col_area: str = COL_AREA_CONSTRUIDA,
                    col_valor: str = "valor_corrigido") -> pd.DataFrame:
    """Deriva as features a partir da base preparada (ver ``depuracao``).

    * ``log_area``: log da área construída.
    * ``padrao_num``: 20..25 como número (a escala é ordinal: tratar como
      numérico deixa a árvore usar a ordem baixo -> luxo).
    * ``idade``: ano da transação menos ano de construção (ACC); negativo =
      venda na planta. Limitado a [-5, 120].
    * ``meses``: meses desde jan/2006 (deriva de mercado não capturada pelo
      IVG-R nacional).
    * ``log_terreno`` e ``log_terreno_ideal``: área do terreno e a parcela do
      terreno atribuída à unidade (terreno x fração ideal), proxy de densidade
      e de exclusividade do empreendimento.
    * ``cep_num``: CEP como número. CEPs próximos são geograficamente próximos,
      então a árvore consegue usar faixas de CEP como localização contínua
      (proxy até haver coordenadas).
    """
    f = pd.DataFrame(index=d.index)
    f["log_area"] = np.log(d[col_area])
    f["padrao_num"] = pd.to_numeric(d[COL_PADRAO_IPTU]).astype(float)
    f["idade"] = (d["ano_transacao"] - d["acc_valido"]).clip(-5, 120)
    f["meses"] = ((d["mes_transacao"].dt.year - MES_ZERO.year) * 12
                  + d["mes_transacao"].dt.month - MES_ZERO.month).astype(float)
    terreno = d[COL_AREA_TERRENO].where(d[COL_AREA_TERRENO] > 0)
    f["log_terreno"] = np.log(terreno)
    ideal = d[COL_FRACAO_IDEAL].where((d[COL_FRACAO_IDEAL] > 0) & (d[COL_FRACAO_IDEAL] <= 1))
    f["log_terreno_ideal"] = np.log(terreno * ideal)
    f["cep_num"] = pd.to_numeric(d["CEP"], errors="coerce").astype(float)
    if "LAT" in d.columns:
        f["lat"], f["lon"] = d["LAT"].astype(float), d["LON"].astype(float)
    if COL_VALOR_VENAL_REF in d.columns:
        vv = d[COL_VALOR_VENAL_REF].where(d[COL_VALOR_VENAL_REF] > 0)
        f["log_valor_venal"] = np.log(vv)
    f["log_valor"] = np.log(d[col_valor])
    return f


FEATURES_BASICAS = ["log_area", "padrao_num"]
FEATURES_LOCALIZACAO_CEP = ["cep_num"]
FEATURES_LOCALIZACAO_GEO = ["lat", "lon"]
FEATURES_ATRIBUTOS = ["idade", "log_terreno", "log_terreno_ideal"]
FEATURES_TEMPO = ["meses"]
FEATURES_ABLACAO_FISCAL = ["log_valor_venal"]


# --------------------------------------------------------------------------
# Split temporal
# --------------------------------------------------------------------------
def split_temporal(d: pd.DataFrame, data_corte: str, col_data: str = "mes_transacao",
                   inicio_treino: str | None = None):
    """Treino ``[inicio_treino, data_corte)`` e teste ``>= data_corte``."""
    corte = pd.Timestamp(data_corte)
    treino = d[d[col_data] < corte]
    if inicio_treino:
        treino = treino[treino[col_data] >= pd.Timestamp(inicio_treino)]
    teste = d[d[col_data] >= corte]
    return treino, teste


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def baseline_huber(localizacao_cat: str = "loc_grp") -> Pipeline:
    """Baseline hedônico aditivo: log_area + localização (one-hot) + padrão (one-hot)."""
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), [localizacao_cat, "padrao_grp"])],
        remainder="passthrough")
    return Pipeline([("pre", pre), ("reg", HuberRegressor(epsilon=1.35, max_iter=3000))])


def novo_hgb(max_iter: int = 500, learning_rate: float = 0.08, max_leaf_nodes: int = 63,
             min_samples_leaf: int = 40, l2: float = 1.0) -> HistGradientBoostingRegressor:
    """Gradient boosting em histogramas. Early stopping interno (10 % de validação)
    evita ajustar a quantidade de árvores à mão."""
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=max_iter, learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def metricas(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    """R² (log), MdAPE, MAPE, viés mediano (sinal: + = superestima)."""
    ape = np.abs(np.exp(pred_log - y_log) - 1.0)
    return {
        "n": int(len(y_log)),
        "r2_log": float(r2_score(y_log, pred_log)) if len(y_log) > 2 else np.nan,
        "mdape_%": float(np.median(ape) * 100),
        "mape_%": float(np.mean(ape) * 100),
        "vies_mediano_%": float(np.median(np.exp(pred_log - y_log) - 1.0) * 100),
        "dentro_de_20%": float(np.mean(ape <= 0.20) * 100),
    }


def metricas_por_padrao(y_log, pred_log, padrao) -> pd.DataFrame:
    """Linha 'GLOBAL' + uma linha por classe de padrão (20..25)."""
    y_log, pred_log, padrao = map(np.asarray, (y_log, pred_log, padrao))
    linhas = [{"padrao": "GLOBAL", **metricas(y_log, pred_log)}]
    for p in PADROES_RESIDENCIAL_VERTICAL:
        m = padrao == p
        if m.sum():
            linhas.append({"padrao": p, **metricas(y_log[m], pred_log[m])})
    return pd.DataFrame(linhas)


def corrigir_vies(y_cal, p_cal, padrao_cal, p_alvo, padrao_alvo, minimo: int = 100):
    """Recentra as previsões pelo viés mediano (em log) de cada classe, estimado numa
    janela de calibração posterior ao treino e anterior ao teste. Mitiga a extrapolação
    do tempo: árvores não projetam tendência além do treino, e o IVG-R é nacional.
    Classes com menos de ``minimo`` observações usam o viés global.
    Retorna ``(p_cal_corrigido, p_alvo_corrigido)``."""
    y_cal, p_cal, padrao_cal, padrao_alvo = map(np.asarray, (y_cal, p_cal, padrao_cal, padrao_alvo))
    ok = np.isfinite(p_cal)
    glob = np.median(y_cal[ok] - p_cal[ok])
    desloc = {}
    for p in PADROES_RESIDENCIAL_VERTICAL:
        m = ok & (padrao_cal == p)
        desloc[p] = np.median(y_cal[m] - p_cal[m]) if m.sum() >= minimo else glob
    f = lambda pad, pr: pr + np.array([desloc.get(x, glob) for x in pad])  # noqa: E731
    return f(padrao_cal, p_cal), f(padrao_alvo, np.asarray(p_alvo))


# --------------------------------------------------------------------------
# Intervalo de predição (conformal split por classe)
# --------------------------------------------------------------------------
@dataclass
class IntervaloConformal:
    """Meia-largura (em log) por classe de padrão, calibrada em dados NÃO usados
    no treino. O intervalo para o valor é ``exp(pred ± q)``."""
    alpha: float = 0.20
    q_por_padrao: dict = field(default_factory=dict)
    q_global: float = np.nan

    def ajustar(self, y_cal, pred_cal, padrao_cal):
        y_cal, pred_cal, padrao_cal = map(np.asarray, (y_cal, pred_cal, padrao_cal))
        res = np.abs(y_cal - pred_cal)
        nivel = lambda n: min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)  # noqa: E731
        self.q_global = float(np.quantile(res, nivel(len(res))))
        for p in PADROES_RESIDENCIAL_VERTICAL:
            r = res[padrao_cal == p]
            self.q_por_padrao[p] = float(np.quantile(r, nivel(len(r)))) if len(r) >= 50 else self.q_global
        return self

    def limites(self, pred_log, padrao):
        q = np.array([self.q_por_padrao.get(p, self.q_global) for p in padrao])
        return np.exp(pred_log - q), np.exp(pred_log + q)

    def cobertura(self, y_log, pred_log, padrao) -> pd.DataFrame:
        y_log, pred_log, padrao = map(np.asarray, (y_log, pred_log, padrao))
        q = np.array([self.q_por_padrao.get(p, self.q_global) for p in padrao])
        dentro = np.abs(y_log - pred_log) <= q
        linhas = [{"padrao": "GLOBAL", "cobertura_%": dentro.mean() * 100,
                   "meia_largura_mediana_%": (np.exp(np.median(q)) - 1) * 100}]
        for p in PADROES_RESIDENCIAL_VERTICAL:
            m = padrao == p
            if m.sum():
                linhas.append({"padrao": p, "cobertura_%": dentro[m].mean() * 100,
                               "meia_largura_mediana_%": (np.exp(self.q_por_padrao[p]) - 1) * 100})
        return pd.DataFrame(linhas)
