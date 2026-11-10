"""Funil de depuração de apartamento em condomínio (Uso IPTU = 20).

Ao contrário de terreno, apartamento vem relativamente limpo: em 2022, 96,7 %
dos registros têm SQL exclusivo no ano (contra 6,2 % em terreno), pois a unidade
só entra no ITBI depois de individualizada no cadastro fiscal. Por isso o filtro
de SQL único é *diagnóstico* aqui (ver ``comum.relatorio_qualidade``), não filtro
obrigatório; aplicá-lo descartaria revendas legítimas dentro do ano.

A depuração relevante é: (i) compra e venda, (ii) valor simbólico (< R$ 5 mil),
(iii) área e datas válidas.
"""

from __future__ import annotations

import pandas as pd

from .carregamento import formatar_cep, normalizar_acc, normalizar_padrao
from .config import (COL_ACC, COL_AREA_CONSTRUIDA, COL_AREA_TERRENO, COL_CEP,
                     COL_DATA_TRANSACAO, COL_FRACAO_IDEAL, COL_NATUREZA,
                     COL_BASE_CALCULO, COL_PADRAO_IPTU, COL_USO_IPTU, COL_VALOR_TRANSACAO,
                     COL_VALOR_VENAL_REF,
                     NATUREZA_COMPRA_VENDA, USO_APARTAMENTO,
                     VALOR_MINIMO_TRANSACAO)


def filtrar_apartamento_compra_e_venda(df: pd.DataFrame) -> pd.DataFrame:
    """Uso (IPTU) = apartamento em condomínio e Natureza = compra e venda."""
    uso = pd.to_numeric(df[COL_USO_IPTU], errors="coerce")
    return df[(uso == USO_APARTAMENTO) & (df[COL_NATUREZA] == NATUREZA_COMPRA_VENDA)].copy()


def remover_valor_simbolico(df: pd.DataFrame, valor_minimo: float = VALOR_MINIMO_TRANSACAO,
                            col_valor: str = COL_VALOR_TRANSACAO) -> pd.DataFrame:
    """Remove Valor de Transação abaixo de ``valor_minimo`` (~0,7 % do bruto)."""
    return df[df[col_valor] >= valor_minimo].copy()


def preparar_apartamentos(df: pd.DataFrame, valor_minimo: float = VALOR_MINIMO_TRANSACAO) -> pd.DataFrame:
    """Tipos, validade e normalização de um DataFrame já filtrado para
    apartamento + compra e venda. Adiciona ``ano_transacao`` e ``acc_valido``.

    Cada linha descartada aqui é uma decisão metodológica: valor ou área
    ausentes/não positivos, data inválida, valor simbólico.
    """
    d = df.copy()
    for c in (COL_VALOR_TRANSACAO, COL_AREA_CONSTRUIDA, COL_AREA_TERRENO, COL_FRACAO_IDEAL,
              COL_VALOR_VENAL_REF, COL_BASE_CALCULO):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d[COL_DATA_TRANSACAO] = pd.to_datetime(d[COL_DATA_TRANSACAO], errors="coerce")
    d = d.dropna(subset=[COL_VALOR_TRANSACAO, COL_AREA_CONSTRUIDA, COL_DATA_TRANSACAO])
    d = d[d[COL_AREA_CONSTRUIDA] > 0]
    d = remover_valor_simbolico(d, valor_minimo)
    d[COL_CEP] = formatar_cep(d[COL_CEP])
    d[COL_PADRAO_IPTU] = normalizar_padrao(d[COL_PADRAO_IPTU])
    d["ano_transacao"] = d[COL_DATA_TRANSACAO].dt.year
    d["acc_valido"] = normalizar_acc(d[COL_ACC], d["ano_transacao"])
    return d
