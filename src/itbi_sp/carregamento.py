"""Carregamento robusto dos arquivos anuais de ITBI (Prefeitura de SP).

Inconsistências reais encontradas nos arquivos 2006-2026 e tratadas aqui:

1. **Abas sem linha de cabeçalho** (JAN-2024, OUT-2024): sem tratamento, a
   primeira linha de dado seria lida como nome de coluna. Detectado pelo
   conteúdo da primeira célula; os nomes são então atribuídos por posição.
2. **Cabeçalho mal rotulado em 2019-2022 e 2024**: as duas últimas colunas
   chegam como ``ACC (IPTU)`` e ``ACC (IPTU).1``. Na verdade a penúltima é a
   *descrição do padrão* (texto) e a última é o *ano de construção* (ACC).
   Uma versão anterior deste código renomeava ``.1`` e depois mantinha a
   primeira coluna duplicada, ou seja, **descartava o ano de construção e
   colocava texto no seu lugar**. Corrigido: nomes atribuídos por posição.
3. **Typo no cabeçalho de 2026** ("Descrição do pardão"): resolvido pela
   mesma atribuição posicional.
4. **CEP numérico** perde o zero à esquerda (04516000 -> 4516000):
   ``formatar_cep`` restaura com zfill(8).
5. **Padrão (IPTU) como float** em alguns anos ("22.0"): ``normalizar_padrao``.
6. **Aba com 27 colunas** (OUT-2024, última coluna vazia ausente): completada
   com NA.

O filtro de interesse pode ser aplicado *durante* a leitura (parâmetro
``filtro``), de modo que o DataFrame de 3 milhões de linhas nunca precise
existir inteiro em memória.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from .config import COL_ACC, COL_CEP, COL_PADRAO_IPTU, COLUNAS_PADRAO

log = logging.getLogger(__name__)

MESES = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
_PRIMEIRA_COLUNA = COLUNAS_PADRAO[0]


def _abas_mensais(xl: pd.ExcelFile) -> list[str]:
    return [s for s in xl.sheet_names if "-" in s and s.split("-")[0] in MESES]


def _carregar_aba_robusta(xl: pd.ExcelFile, aba: str) -> pd.DataFrame:
    """Lê uma aba mensal e devolve as 28 colunas padrão, por posição."""
    primeira_celula = xl.parse(aba, header=None, nrows=1).iloc[0, 0]
    d = xl.parse(aba, header=None if primeira_celula != _PRIMEIRA_COLUNA else 0)

    n_esperado = len(COLUNAS_PADRAO)
    if d.shape[1] > n_esperado:
        # achado real (MAR-2008, FEV-2013, OUT-2023): coluna 29 com 1 única
        # célula preenchida em >7 mil linhas = lixo de edição manual.
        extras = d.iloc[:, n_esperado:]
        if extras.notna().sum().sum() > 5:
            raise ValueError(f"aba {aba}: {d.shape[1]} colunas com dado real além da 28ª")
        log.warning("aba %s: descartando %d coluna(s) excedente(s) quase vazia(s)",
                    aba, d.shape[1] - n_esperado)
        d = d.iloc[:, :n_esperado]
    if d.shape[1] < len(COLUNAS_PADRAO):
        log.warning("aba %s: %d colunas; completando as restantes com NA", aba, d.shape[1])

    d.columns = COLUNAS_PADRAO[: d.shape[1]]
    for c in COLUNAS_PADRAO[d.shape[1]:]:
        d[c] = pd.NA
    return d[COLUNAS_PADRAO]


def carregar_arquivo_anual(
    caminho_xlsx: str,
    ano_rotulo,
    filtro: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Carrega todas as abas mensais de um arquivo anual.

    ``ano_rotulo`` é atribuído por você (não inferido do nome do arquivo, que
    às vezes traz a data de extração: ``..._28012026_`` contém 2025 inteiro).
    ``filtro`` é aplicado aba a aba para limitar o uso de memória.
    """
    xl = pd.ExcelFile(caminho_xlsx)
    partes = []
    for aba in _abas_mensais(xl):
        d = _carregar_aba_robusta(xl, aba)
        if filtro is not None:
            d = filtro(d)
        d["ano_arquivo"] = ano_rotulo
        d["mes_planilha"] = aba
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def carregar_multiplos_anos(
    arquivos: dict, filtro: Callable[[pd.DataFrame], pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Carrega e consolida ``{rotulo_do_ano: caminho_do_xlsx}``."""
    partes = []
    for ano, caminho in arquivos.items():
        log.info("carregando %s (%s)", ano, caminho)
        partes.append(carregar_arquivo_anual(caminho, ano, filtro=filtro))
    return pd.concat(partes, ignore_index=True)


def formatar_cep(serie_cep: pd.Series) -> pd.Series:
    """CEP com 8 dígitos e zero à esquerda preservado (ver achado 4)."""
    numerico = pd.to_numeric(serie_cep, errors="coerce").astype("Int64")
    return numerico.astype(str).str.zfill(8).where(numerico.notna())


def normalizar_padrao(serie: pd.Series) -> pd.Series:
    """Padrão (IPTU) como string de inteiro: 22, 22.0 e '22' viram '22'."""
    return pd.to_numeric(serie, errors="coerce").astype("Int64").astype(str)


def normalizar_acc(serie: pd.Series, ano_transacao: pd.Series) -> pd.Series:
    """Ano de construção válido ou NA.

    Descarta zero/ausente e valores fora de [1900, ano_transacao + 5] (a
    margem permite venda "na planta", com ano de construção futuro).
    """
    acc = pd.to_numeric(serie, errors="coerce")
    valido = (acc >= 1900) & (acc <= ano_transacao + 5)
    return acc.where(valido)
