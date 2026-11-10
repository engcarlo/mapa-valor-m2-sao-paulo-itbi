"""
mapa.py

Monta o mapa coroplético HTML autônomo (sem dependências externas em tempo
de execução — os dados vão embutidos no arquivo) com slider de intervalo de
anos e caixa de estatísticas descritivas por distrito (mín, máx, média,
mediana, desvio-padrão, amplitude).

O mapa mostra ESTATÍSTICA DESCRITIVA observada, não uma regressão — o
motivo é técnico, não de preferência: um HTML estático não pode reajustar
um HuberRegressor em tempo real a cada posição do slider (isso exigiria um
backend Python). Agregar estatísticas (n, mín, máx, média, desvio-padrão)
por ano e recombinar no navegador é exato matematicamente (fórmula de
variância combinada, não uma aproximação) e funciona sem servidor. A única
aproximação é a mediana do período (ponderada pelas medianas anuais) — não
existe forma exata de recombinar medianas sem guardar os valores brutos, o
que inflaria demais o arquivo.

O template (data/mapa_template.html) embute o D3.js v7.9.0 (ISC) — funciona em qualquer navegador ao abrir o arquivo
localmente ou hospedado, sem servidor por trás.

O GeoJSON de distritos já vem simplificado (data/distritos_sp_simplificado.geojson,
tolerância ~0,0008°) para caber embutido no HTML sem pesar o arquivo — a
função `simplificar_geojson` está aqui caso precise regenerar a partir de
uma fonte de maior resolução.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .carregamento import normalizar_padrao
from .config import CAMINHO_D3, CAMINHO_GEOJSON_PADRAO, CAMINHO_TEMPLATE_PADRAO, PADROES_RESIDENCIAL_VERTICAL


def simplificar_geojson(caminho_geojson_origem: str, caminho_saida: str, tolerancia: float = 0.0008) -> None:
    """Simplifica a geometria de um GeoJSON de distritos (reduz nº de
    vértices) para caber embutido num HTML sem pesar o arquivo. Mantém só
    a propriedade 'nome' (renomeada a partir de 'ds_nome', se presente).
    Requer shapely.
    """
    from shapely.geometry import shape, mapping

    geo = json.load(open(caminho_geojson_origem, encoding='utf-8'))
    for f in geo['features']:
        geom_simpl = shape(f['geometry']).simplify(tolerancia, preserve_topology=True)
        f['geometry'] = mapping(geom_simpl)
        nome = f['properties'].get('ds_nome') or f['properties'].get('nome')
        f['properties'] = {'nome': nome}

    with open(caminho_saida, 'w', encoding='utf-8') as fh:
        json.dump(geo, fh, ensure_ascii=False)


def calcular_stats_por_ano(df: pd.DataFrame, col_preco_m2: str, col_distrito: str,
                            col_padrao: str, col_ano: str,
                            padroes_validos: list[str] | None = None) -> pd.DataFrame:
    """Calcula, para cada (padrão, distrito, ano), as estatísticas
    descritivas do preço/m² observado: n, mínimo, máximo, média, mediana,
    desvio-padrão. Espera valores já corrigidos temporalmente (ver
    temporal_ivgr.py) — sem isso, agregar anos diferentes não faz sentido.

    IMPORTANTE: normaliza a coluna de padrão via pd.to_numeric antes de
    comparar contra `padroes_validos` — em alguns anos (2010, 2012 nesta
    análise) o Excel guarda o campo como float ("22.0"), e uma comparação
    de string direta descartaria esses anos silenciosamente.
    """
    padroes_validos = padroes_validos or PADROES_RESIDENCIAL_VERTICAL
    d = df.dropna(subset=[col_distrito]).copy()
    d['_padrao'] = normalizar_padrao(d[col_padrao])
    d = d[d['_padrao'].isin(padroes_validos)]
    d = d[(d[col_preco_m2] > 0) & np.isfinite(d[col_preco_m2])]

    grupo = d.groupby(['_padrao', col_distrito, col_ano])[col_preco_m2]
    stats = grupo.agg(n='count', minimo='min', maximo='max', media='mean',
                       mediana='median', desvio='std').reset_index()
    stats['desvio'] = stats['desvio'].fillna(0)
    return stats.rename(columns={'_padrao': 'padrao', col_distrito: 'distrito', col_ano: 'ano'})


def montar_stats_data(stats: pd.DataFrame) -> dict:
    """Converte o DataFrame de calcular_stats_por_ano para a estrutura
    compacta {padrao: {distrito: {ano: [n,min,max,media,mediana,desvio]}}}
    esperada pelo template (arrays em vez de objetos, pra reduzir tamanho).
    """
    estrutura: dict = {}
    for _, row in stats.iterrows():
        pad, dist, ano = row['padrao'], row['distrito'], str(int(row['ano']))
        estrutura.setdefault(pad, {}).setdefault(dist, {})[ano] = [
            int(row['n']), round(float(row['minimo']), 1), round(float(row['maximo']), 1),
            round(float(row['media']), 1), round(float(row['mediana']), 1), round(float(row['desvio']), 1),
        ]
    return estrutura


def gerar_mapa_html(stats: pd.DataFrame, rotulos_padrao: dict, caminho_saida: str,
                     caminho_template: str = CAMINHO_TEMPLATE_PADRAO,
                     caminho_geojson: str = CAMINHO_GEOJSON_PADRAO) -> str:
    """Monta o arquivo HTML final embutindo o GeoJSON de distritos e as
    estatísticas por (padrão, distrito, ano) no template. `stats` é a
    saída de `calcular_stats_por_ano`. Retorna o caminho do arquivo gerado.
    """
    with open(caminho_template, encoding='utf-8') as f:
        html = f.read()
    with open(caminho_geojson, encoding='utf-8') as f:
        geodata_str = f.read()

    stats_data_str = json.dumps(montar_stats_data(stats), ensure_ascii=False)
    rotulos_str = json.dumps(rotulos_padrao, ensure_ascii=False)

    with open(CAMINHO_D3, encoding='utf-8') as f:
        d3_js = f.read()

    html = html.replace('__D3JS__', d3_js)
    html = html.replace('__GEODATA__', geodata_str)
    html = html.replace('__STATSDATA__', stats_data_str)
    html = html.replace('__ROTULOS__', rotulos_str)

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        f.write(html)

    return caminho_saida
