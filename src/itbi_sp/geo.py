"""Geocodificação por CEP e associação a distrito de São Paulo.

1. Join por CEP com uma base de coordenadas (lat/lon por CEP).
2. Ponto-em-polígono contra os 96 distritos oficiais.

Por que CEP e não Bairro: o campo Bairro do ITBI tem ~14 mil valores distintos,
com lixo de digitação ("TORRE 1", "BLOCO A"). CEP não falta em nenhum registro e,
com coordenadas por CEP, a cobertura sobe para ~85 %.

**Otimização**: o ponto-em-polígono é feito uma vez por *CEP distinto* (dezenas
de milhares) e o resultado é propagado às linhas (centenas de milhares), de forma
vetorizada com ``STRtree.query(..., predicate="within")`` do shapely 2.

Fontes (de terceiros, não redistribuídas, salvo o GeoJSON simplificado):
* Coordenadas: Kaggle ``millengustavo/ibge-censo-cep-coordenadas`` (POSTCODE, LON, LAT).
* Distritos: GeoSampa via github.com/codigourbano/distritos-sp. Cópia simplificada
  (~120 KB) em ``itbi_sp/data/``.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree

from .config import CAMINHO_GEOJSON_PADRAO

log = logging.getLogger(__name__)


def carregar_coordenadas_cep(caminho_csv: str) -> pd.DataFrame:
    """DataFrame (LAT, LON) indexado por CEP no formato ``XXXXX-XXX``.
    CEP duplicado na fonte: mantém a primeira ocorrência."""
    coords = pd.read_csv(caminho_csv, usecols=["POSTCODE", "LAT", "LON"])
    return coords.drop_duplicates("POSTCODE").set_index("POSTCODE")[["LAT", "LON"]]


def carregar_poligonos_distrito(caminho_geojson=CAMINHO_GEOJSON_PADRAO):
    """Retorna ``(poligonos, nomes, STRtree)``."""
    with open(caminho_geojson, encoding="utf-8") as fh:
        geo = json.load(fh)
    chave = "nome" if "nome" in geo["features"][0]["properties"] else "ds_nome"
    poligonos = [shape(f["geometry"]) for f in geo["features"]]
    nomes = [f["properties"][chave] for f in geo["features"]]
    return poligonos, nomes, STRtree(poligonos)


def atribuir_distrito_pontos(lat, lon, poligonos, nomes, arvore) -> np.ndarray:
    """Distrito (ou None) de cada ponto, vetorizado. Pontos sem coordenada ou
    fora de todos os polígonos retornam None."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    saida = np.full(len(lat), None, dtype=object)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return saida
    pontos = shapely.points(lon[ok], lat[ok])
    idx_pt, idx_poly = arvore.query(pontos, predicate="within")
    # ponto exatamente na divisa pode cair em 2 polígonos: fica o primeiro
    primeiro = pd.Series(idx_poly, index=idx_pt).groupby(level=0).first()
    destino = np.flatnonzero(ok)
    nomes_arr = np.asarray(nomes, dtype=object)
    saida[destino[primeiro.index.to_numpy()]] = nomes_arr[primeiro.to_numpy()]
    return saida


def pipeline_geocodificacao(df: pd.DataFrame, col_cep: str, caminho_coords_csv: str,
                            caminho_geojson=CAMINHO_GEOJSON_PADRAO) -> pd.DataFrame:
    """Adiciona ``LAT``, ``LON`` e ``distrito``. ``col_cep`` deve ter 8 dígitos
    (ver ``carregamento.formatar_cep``)."""
    tabela = carregar_coordenadas_cep(caminho_coords_csv)
    poligonos, nomes, arvore = carregar_poligonos_distrito(caminho_geojson)

    d = df.copy()
    cep = d[col_cep]
    d["_cep_fmt"] = cep.str[:5] + "-" + cep.str[5:]
    d = d.join(tabela, on="_cep_fmt")

    # distrito calculado por CEP distinto, depois propagado
    unicos = d.drop_duplicates("_cep_fmt")[["_cep_fmt", "LAT", "LON"]].reset_index(drop=True)
    unicos["distrito"] = atribuir_distrito_pontos(unicos["LAT"], unicos["LON"], poligonos, nomes, arvore)
    d = d.merge(unicos[["_cep_fmt", "distrito"]], on="_cep_fmt", how="left").drop(columns="_cep_fmt")

    log.info("coordenada encontrada: %.1f %%  |  distrito atribuído: %.1f %%",
             d["LAT"].notna().mean() * 100, d["distrito"].notna().mean() * 100)
    return d
