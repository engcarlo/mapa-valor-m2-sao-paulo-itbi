"""Correção temporal de valores via IVG-R (BCB, série SGS 21340).

Por que um índice específico do setor e não IPCA/IGP-M: o valor de imóveis
residenciais não anda junto com a inflação geral e pode se mover no sentido
oposto (2015-2017: IVG-R caiu, IPCA subiu). Exemplo numérico completo em
``docs/metodologia_correcao_ivgr.md``.

Limitações (declarar em qualquer laudo/artigo): índice NACIONAL (sem abertura
por cidade ou padrão), apenas imóveis residenciais FINANCIADOS, inaplicável a
terreno e comercial.

    fator = IVGR(data_base) / IVGR(mês da transação)
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

URL_SGS_IVGR = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.21340/dados?formato=json"


def buscar_ivgr_api(data_inicial: str | None = None, data_final: str | None = None) -> pd.Series:
    """Série IVG-R direto da API do BCB (datas ``dd/mm/aaaa``)."""
    import requests

    url = URL_SGS_IVGR
    if data_inicial and data_final:
        url += f"&dataInicial={data_inicial}&dataFinal={data_final}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.Series(
        {pd.to_datetime(i["data"], format="%d/%m/%Y"): float(i["valor"]) for i in resp.json()}
    ).sort_index()


def carregar_ivgr_csv(caminho_csv: str) -> pd.Series:
    """CSV exportado do site do BCB (sep ';', decimal ','). Reprodutível offline."""
    df = pd.read_csv(caminho_csv, sep=";", decimal=",", encoding="utf-8")
    df.columns = ["data", "valor"]
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    return df.set_index("data")["valor"].sort_index()


def aplicar_correcao_temporal(df: pd.DataFrame, serie_ivgr: pd.Series, col_data: str,
                              col_valor: str, data_base: pd.Timestamp | None = None) -> pd.DataFrame:
    """Adiciona ``mes_transacao``, ``ivgr_na_data``, ``fator_correcao`` e
    ``valor_corrigido``. ``data_base`` padrão = último ponto da série.

    Transações fora do intervalo da série (< mar/2001, ou datas futuras) são
    descartadas com aviso, nunca corrigidas com valor incorreto.
    """
    d = df.copy()
    d["mes_transacao"] = pd.to_datetime(d[col_data]).dt.to_period("M").dt.to_timestamp()
    data_base = data_base if data_base is not None else serie_ivgr.index.max()
    ivgr_base = serie_ivgr.loc[data_base]

    d["ivgr_na_data"] = d["mes_transacao"].map(serie_ivgr)
    n_fora = int(d["ivgr_na_data"].isna().sum())
    if n_fora:
        log.warning("%d registros fora do range do IVG-R (%s a %s) descartados.",
                    n_fora, serie_ivgr.index.min().date(), serie_ivgr.index.max().date())
    d = d.dropna(subset=["ivgr_na_data"])
    d["fator_correcao"] = ivgr_base / d["ivgr_na_data"]
    d["valor_corrigido"] = d[col_valor] * d["fator_correcao"]
    return d
