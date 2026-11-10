"""
terreno_depuracao.py

Funil de depuração específico para terreno (Uso IPTU = 0). Achado central:
a categoria TERRENO está fortemente contaminada por unidades de
empreendimentos ainda não individualizados no cadastro fiscal (o apartamento
é registrado sob o SQL do terreno-mãe durante a incorporação). O critério
mais confiável para detectar essa contaminação é a repetição de SQL no ano
— validado empiricamente: 100% dos SQLs repetidos têm Área do Terreno
idêntica em todas as ocorrências, o que só é fisicamente possível se forem
unidades do mesmo empreendimento, não terrenos distintos.

Heurísticas de texto no Complemento (AP/TORRE/BLOCO) foram testadas e
descartadas como critério principal por terem lacunas conhecidas.
"""

import re
import pandas as pd

from ..comum import (
    COL_USO_IPTU, COL_NATUREZA, COL_FRACAO_IDEAL, COL_COMPLEMENTO,
    COL_AREA_TERRENO, COL_SQL, NATUREZA_COMPRA_VENDA, carregar_itbi,
    filtrar_sql_unico, separar_por_ano_transacao, calcular_preco_m2,
)

USO_TERRENO = 0


def filtrar_terreno_compra_e_venda(df: pd.DataFrame) -> pd.DataFrame:
    """Uso (IPTU) = TERRENO, Natureza = Compra e venda, Fração Ideal = 1.0
    (exclui frações de condomínio)."""
    return df[
        (df[COL_USO_IPTU] == USO_TERRENO)
        & (df[COL_NATUREZA] == NATUREZA_COMPRA_VENDA)
        & (df[COL_FRACAO_IDEAL] == 1.0)
    ].copy()


def flag_contaminacao_texto(df: pd.DataFrame) -> pd.Series:
    """Heurística textual (INSUFICIENTE — mantida só para comparação/
    diagnóstico). Tem lacunas conhecidas: não pega 'Ap12' colado,
    'APARTAMENTO' por extenso, nem numeração pura de unidade. Prefira
    `filtrar_sql_unico` (do módulo comum) como critério principal.
    """
    padrao = re.compile(r"\b(AP|APTO|APT|TORRE|BLOCO|BLC|UNID)\b", flags=re.IGNORECASE)
    comp = df[COL_COMPLEMENTO].fillna("").astype(str)
    return comp.str.contains(padrao, na=False, regex=True)


def validar_area_identica_em_repetidos(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Diagnóstico de apoio: para cada SQL que se repete em `df_bruto`,
    conta quantas Áreas do Terreno distintas aparecem. Esperado: quase
    sempre 1 área distinta por SQL repetido, confirmando contaminação por
    unidades do mesmo empreendimento.
    """
    contagem = df_bruto[COL_SQL].value_counts()
    repetidos = contagem[contagem > 1].index
    sub = df_bruto[df_bruto[COL_SQL].isin(repetidos)]
    return sub.groupby(COL_SQL).agg(
        n_ocorrencias=(COL_SQL, "size"),
        areas_distintas=(COL_AREA_TERRENO, "nunique"),
    )


def pipeline_depuracao_terreno(caminho_xlsx: str, ano: int = 2022) -> dict:
    """Executa o funil completo de terreno e devolve um dicionário com os
    dataframes intermediários e finais.
    """
    bruto = carregar_itbi(caminho_xlsx)
    terreno_cv = filtrar_terreno_compra_e_venda(bruto)
    sql_unico = filtrar_sql_unico(terreno_cv, limite=1)
    dentro, fora = separar_por_ano_transacao(sql_unico, ano=ano)
    dentro = dentro.copy()
    dentro["preco_m2"] = calcular_preco_m2(dentro, COL_AREA_TERRENO)

    return {
        "bruto": bruto,
        "terreno_compra_venda": terreno_cv,
        "sql_unico": sql_unico,
        "dentro_do_ano": dentro,
        "fora_do_ano_regularizacao_tardia": fora,
    }
