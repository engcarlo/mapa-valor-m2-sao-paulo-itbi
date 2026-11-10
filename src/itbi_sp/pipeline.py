"""Orquestração ponta a ponta. Cada etapa é uma função separada e cacheável.

    ITBI (xlsx por ano) -> filtro apto+C&V (durante a leitura) -> depuração
    -> correção IVG-R -> geocodificação por CEP -> distrito -> {mapa, ML, diagnóstico}
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .carregamento import carregar_arquivo_anual
from .config import (COL_AREA_CONSTRUIDA, COL_CEP, COL_DATA_TRANSACAO, COL_PADRAO_IPTU,
                     COL_VALOR_TRANSACAO, PADROES_RESIDENCIAL_VERTICAL, PRODUCAO_ANO_MINIMO,
                     ROTULOS_PADRAO)
from .depuracao import filtrar_apartamento_compra_e_venda, preparar_apartamentos
from .temporal import aplicar_correcao_temporal, carregar_ivgr_csv

log = logging.getLogger(__name__)


def carregar_configuracao(caminho_toml: str | Path) -> dict:
    import tomllib
    with open(caminho_toml, "rb") as fh:
        cfg = tomllib.load(fh)
    raiz = Path(caminho_toml).resolve().parent
    for chave in ("itbi_dir", "ivgr_csv", "cep_coords_csv", "cache_dir", "saida_dir"):
        if chave in cfg["caminhos"]:
            cfg["caminhos"][chave] = str((raiz / cfg["caminhos"][chave]).resolve())
    return cfg


def construir_base(cfg: dict, usar_cache: bool = True) -> pd.DataFrame:
    """Base de apartamentos depurada, corrigida no tempo e (se houver CSV de
    coordenadas) geocodificada. Cache por ano em parquet: a leitura dos xlsx
    leva ~20 s por ano e só precisa acontecer uma vez."""
    cam = cfg["caminhos"]
    cache = Path(cam["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    partes = []
    for ano, nome in cfg["arquivos"].items():
        arq_cache = cache / f"apto_{ano}.parquet"
        if usar_cache and arq_cache.exists():
            partes.append(pd.read_parquet(arq_cache))
            continue
        d = carregar_arquivo_anual(str(Path(cam["itbi_dir"]) / nome), int(ano),
                                   filtro=filtrar_apartamento_compra_e_venda)
        for c in d.columns:  # colunas de tipo misto não vão para parquet
            if d[c].dtype == object:
                d[c] = d[c].astype(str).where(d[c].notna())
        d.to_parquet(arq_cache)
        partes.append(d)
    bruto = pd.concat(partes, ignore_index=True)

    d = preparar_apartamentos(bruto)
    ivgr = carregar_ivgr_csv(cam["ivgr_csv"])
    d = aplicar_correcao_temporal(d, ivgr, COL_DATA_TRANSACAO, COL_VALOR_TRANSACAO)
    d = d[d[COL_PADRAO_IPTU].isin(PADROES_RESIDENCIAL_VERTICAL)].copy()

    coords = cam.get("cep_coords_csv")
    if coords and Path(coords).exists():
        from .geo import pipeline_geocodificacao
        d = pipeline_geocodificacao(d, COL_CEP, coords)
    else:
        log.warning("CSV de coordenadas por CEP ausente: sem LAT/LON/distrito (mapa e ML geográfico indisponíveis).")
    d["preco_m2"] = d["valor_corrigido"] / d[COL_AREA_CONSTRUIDA]
    return d


def gerar_mapa(base: pd.DataFrame, caminho_saida: str) -> str:
    from .mapa import calcular_stats_por_ano, gerar_mapa_html
    stats = calcular_stats_por_ano(base, "preco_m2", "distrito", COL_PADRAO_IPTU, "ano_arquivo")
    return gerar_mapa_html(stats, ROTULOS_PADRAO, caminho_saida)


def diagnostico_temporal(base: pd.DataFrame, ano_corte: int = PRODUCAO_ANO_MINIMO):
    from .diagnostico_temporal import comparar_pool_vs_era_unica, diagnostico_por_era
    from .modelo_hedonico import preparar_base_modelo
    b = preparar_base_modelo(base, COL_AREA_CONSTRUIDA, "valor_corrigido", "distrito", COL_PADRAO_IPTU)
    return diagnostico_por_era(b), comparar_pool_vs_era_unica(b, ano_corte=ano_corte)
