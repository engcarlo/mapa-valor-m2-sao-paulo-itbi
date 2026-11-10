"""Experimento de ML com validação temporal e ablações (ver ``ml.py``).

Linha do tempo (fixa para comparabilidade entre configurações):

    treino        [inicio_treino, data_calibracao)
    calibração    [data_calibracao, data_teste)    -> intervalo conformal
    teste         [data_teste, fim)                -> todas as métricas

Cada configuração responde a uma pergunta específica (coluna ``pergunta``).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import ml
from .config import PADROES_RESIDENCIAL_VERTICAL

log = logging.getLogger(__name__)


def _agrupar_raros(s: pd.Series, minimo: int) -> pd.Series:
    s = s.fillna("DESCONHECIDO")
    vc = s.value_counts()
    return s.where(s.isin(vc[vc >= minimo].index), "OUTROS")


def preparar(base: pd.DataFrame) -> pd.DataFrame:
    """Une features e metadados; define a coluna de localização categórica do baseline
    (distrito se houver, senão CEP de 5 dígitos)."""
    f = ml.montar_features(base)
    d = pd.concat([f, base[["mes_transacao", "ano_transacao", "CEP"]]], axis=1)
    d["padrao_grp"] = base["Padrão (IPTU)"].astype(str)
    if "distrito" in base.columns:
        d["loc_grp"] = base["distrito"].fillna("SEM_DISTRITO")
    else:
        d["loc_grp"] = base["CEP"].str[:5]
    d["loc_grp"] = _agrupar_raros(d["loc_grp"], 30)
    return d


def configuracoes(tem_geo: bool) -> dict:
    loc = ml.FEATURES_LOCALIZACAO_GEO if tem_geo else ml.FEATURES_LOCALIZACAO_CEP
    B = ml.FEATURES_BASICAS
    return {
        "A baseline Huber aditivo (2019+)": dict(tipo="huber", inicio="2019-01-01", pergunta="ponto de partida"),
        "B HGB: área+padrão+localização": dict(tipo="hgb", feats=B + loc, inicio="2019-01-01",
                                                pergunta="só trocar o algoritmo ajuda?"),
        "C + idade, terreno, fração ideal": dict(tipo="hgb", feats=B + loc + ml.FEATURES_ATRIBUTOS, inicio="2019-01-01",
                                                  pergunta="atributos do edifício agregam?"),
        "D + tempo (meses)": dict(tipo="hgb", feats=B + loc + ml.FEATURES_ATRIBUTOS + ml.FEATURES_TEMPO,
                                   inicio="2019-01-01", pergunta="deriva de mercado local?"),
        "E = D com histórico completo (2006+)": dict(tipo="hgb",
                                                       feats=B + loc + ml.FEATURES_ATRIBUTOS + ml.FEATURES_TEMPO,
                                                       inicio="2006-01-01", pergunta="o ML resolve o drift de 21 anos?"),
        "F = D, um modelo por padrão": dict(tipo="hgb_por_padrao",
                                             feats=B + loc + ml.FEATURES_ATRIBUTOS + ml.FEATURES_TEMPO,
                                             inicio="2019-01-01", pergunta="modelo por classe vs global"),
        "H = D + correção de viés (calibração)": dict(
            tipo="hgb", feats=B + loc + ml.FEATURES_ATRIBUTOS + ml.FEATURES_TEMPO, inicio="2019-01-01",
            corrige_vies=True, pergunta="o viés do teste é removível sem olhar o teste?"),
        "I = F + correção de viés (calibração)": dict(
            tipo="hgb_por_padrao", feats=B + loc + ml.FEATURES_ATRIBUTOS + ml.FEATURES_TEMPO, inicio="2019-01-01",
            corrige_vies=True, pergunta="melhor combinação: por classe + viés corrigido"),
        "G = D + valor venal (ablação fiscal)": dict(tipo="hgb",
                                                      feats=B + loc + ml.FEATURES_ATRIBUTOS + ml.FEATURES_TEMPO
                                                      + ml.FEATURES_ABLACAO_FISCAL, inicio="2019-01-01",
                                                      pergunta="quanto do ganho vem do valor fiscal?"),
    }


def _ajustar_prever(cfg: dict, tr: pd.DataFrame, previsao_em: list[pd.DataFrame]) -> list[np.ndarray]:
    if cfg["tipo"] == "huber":
        m = ml.baseline_huber()
        m.fit(tr[["log_area", "loc_grp", "padrao_grp"]], tr["log_valor"])
        return [m.predict(x[["log_area", "loc_grp", "padrao_grp"]]) for x in previsao_em]
    feats = cfg["feats"]
    if cfg["tipo"] == "hgb":
        m = ml.novo_hgb().fit(tr[feats], tr["log_valor"])
        return [m.predict(x[feats]) for x in previsao_em]
    # um modelo por padrão
    saidas = [np.full(len(x), np.nan) for x in previsao_em]
    for p in PADROES_RESIDENCIAL_VERTICAL:
        trp = tr[tr["padrao_grp"] == p]
        if len(trp) < 200:
            continue
        m = ml.novo_hgb().fit(trp[feats], trp["log_valor"])
        for x, s in zip(previsao_em, saidas):
            mk = (x["padrao_grp"] == p).to_numpy()
            if mk.any():
                s[mk] = m.predict(x.loc[mk, feats])
    return saidas


def rodar_experimento(base: pd.DataFrame, data_calibracao="2024-07-01", data_teste="2025-01-01",
                      alpha=0.20, tem_geo: bool | None = None):
    """Retorna ``(resumo, por_padrao, cobertura, detalhe)``."""
    tem_geo = ("LAT" in base.columns) if tem_geo is None else tem_geo
    d = preparar(base)
    resumo, por_padrao, cobertura = [], [], []
    detalhe = {}
    for nome, cfg in configuracoes(tem_geo).items():
        t0 = time.time()
        tr = d[(d.mes_transacao >= pd.Timestamp(cfg["inicio"])) & (d.mes_transacao < pd.Timestamp(data_calibracao))]
        ca = d[(d.mes_transacao >= pd.Timestamp(data_calibracao)) & (d.mes_transacao < pd.Timestamp(data_teste))]
        te = d[d.mes_transacao >= pd.Timestamp(data_teste)]
        p_ca, p_te = _ajustar_prever(cfg, tr, [ca, te])
        ok_te = np.isfinite(p_te)
        ok_ca = np.isfinite(p_ca)
        if cfg.get("corrige_vies"):
            p_ca, p_te = ml.corrigir_vies(ca["log_valor"].to_numpy(), p_ca, ca["padrao_grp"].to_numpy(),
                                          p_te, te["padrao_grp"].to_numpy())
        te_ok, ca_ok = te[ok_te], ca[ok_ca]
        mt = ml.metricas_por_padrao(te_ok["log_valor"], p_te[ok_te], te_ok["padrao_grp"])
        mt.insert(0, "configuracao", nome)
        por_padrao.append(mt)
        g = mt[mt.padrao == "GLOBAL"].iloc[0]
        resumo.append({"configuracao": nome, "pergunta": cfg["pergunta"], "n_treino": len(tr), "n_teste": int(g.n),
                       "r2_log": g.r2_log, "mdape_%": g["mdape_%"], "dentro_de_20%": g["dentro_de_20%"],
                       "segundos": round(time.time() - t0)})
        ic = ml.IntervaloConformal(alpha=alpha).ajustar(ca_ok["log_valor"], p_ca[ok_ca], ca_ok["padrao_grp"])
        cv = ic.cobertura(te_ok["log_valor"], p_te[ok_te], te_ok["padrao_grp"])
        cv.insert(0, "configuracao", nome)
        cobertura.append(cv)
        detalhe[nome] = (te_ok.index, p_te[ok_te])
        log.info("%s: MdAPE %.1f%%  R2 %.3f  (%ds)", nome, g["mdape_%"], g.r2_log, time.time() - t0)
    return (pd.DataFrame(resumo), pd.concat(por_padrao, ignore_index=True),
            pd.concat(cobertura, ignore_index=True), detalhe)
