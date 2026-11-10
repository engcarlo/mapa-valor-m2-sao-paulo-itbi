"""Constantes compartilhadas: nomes de coluna da planilha da Prefeitura de SP,
padrões construtivos, e caminhos dos dados empacotados.

Decisões de projeto documentadas aqui (para reuso na seção de metodologia):

* Escopo do modelo: apartamento em condomínio (Uso IPTU = 20), natureza
  "1.Compra e venda", padrões residenciais verticais 20-25.
* Valor mínimo de transação R$ 5.000: remove valor simbólico (integralização
  de capital, doação mal classificada etc.). Ver docs/metodologia_terrenos_2022.md.
* Período de produção >= 2019: o efeito de distrito não é estacionário
  (ver itbi_sp.diagnostico_temporal).
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CAMINHO_GEOJSON_PADRAO = DATA_DIR / "distritos_sp_simplificado.geojson"
CAMINHO_TEMPLATE_PADRAO = DATA_DIR / "mapa_template.html"
CAMINHO_D3 = DATA_DIR / "d3.v7.9.0.min.js"  # D3 (ISC) embutido: o mapa funciona offline

# --- Colunas da planilha (28, na ordem original) ---------------------------
COL_SQL = "N° do Cadastro (SQL)"
COL_LOGRADOURO = "Nome do Logradouro"
COL_NUMERO = "Número"
COL_COMPLEMENTO = "Complemento"
COL_BAIRRO = "Bairro"
COL_REFERENCIA = "Referência"
COL_CEP = "CEP"
COL_NATUREZA = "Natureza de Transação"
COL_VALOR_TRANSACAO = "Valor de Transação (declarado pelo contribuinte)"
COL_DATA_TRANSACAO = "Data de Transação"
COL_VALOR_VENAL_REF_TOTAL = "Valor Venal de Referência"
COL_PROPORCAO = "Proporção Transmitida (%)"
COL_VALOR_VENAL_REF = "Valor Venal de Referência (proporcional)"
COL_BASE_CALCULO = "Base de Cálculo adotada"
COL_TIPO_FINANCIAMENTO = "Tipo de Financiamento"
COL_VALOR_FINANCIADO = "Valor Financiado"
COL_CARTORIO = "Cartório de Registro"
COL_MATRICULA = "Matrícula do Imóvel"
COL_SITUACAO_SQL = "Situação do SQL"
COL_AREA_TERRENO = "Área do Terreno (m2)"
COL_TESTADA = "Testada (m)"
COL_FRACAO_IDEAL = "Fração Ideal"
COL_AREA_CONSTRUIDA = "Área Construída (m2)"
COL_USO_IPTU = "Uso (IPTU)"
COL_DESC_USO = "Descrição do uso (IPTU)"
COL_PADRAO_IPTU = "Padrão (IPTU)"
COL_DESC_PADRAO = "Descrição do padrão (IPTU)"
COL_ACC = "ACC (IPTU)"  # ano de construção corrigido

COLUNAS_PADRAO = [
    COL_SQL, COL_LOGRADOURO, COL_NUMERO, COL_COMPLEMENTO, COL_BAIRRO,
    COL_REFERENCIA, COL_CEP, COL_NATUREZA, COL_VALOR_TRANSACAO,
    COL_DATA_TRANSACAO, COL_VALOR_VENAL_REF_TOTAL, COL_PROPORCAO,
    COL_VALOR_VENAL_REF, COL_BASE_CALCULO, COL_TIPO_FINANCIAMENTO,
    COL_VALOR_FINANCIADO, COL_CARTORIO, COL_MATRICULA, COL_SITUACAO_SQL,
    COL_AREA_TERRENO, COL_TESTADA, COL_FRACAO_IDEAL, COL_AREA_CONSTRUIDA,
    COL_USO_IPTU, COL_DESC_USO, COL_PADRAO_IPTU, COL_DESC_PADRAO, COL_ACC,
]

NATUREZA_COMPRA_VENDA = "1.Compra e venda"
USO_TERRENO = 0
USO_APARTAMENTO = 20

# --- Padrão construtivo (residencial vertical) ------------------------------
PADROES_RESIDENCIAL_VERTICAL = ["20", "21", "22", "23", "24", "25"]
ROTULOS_PADRAO = {
    "20": "Padrão baixo",
    "21": "Baixo-médio",
    "22": "Médio",
    "23": "Médio-alto",
    "24": "Alto",
    "25": "Luxo",
}

# --- Parâmetros de modelagem ------------------------------------------------
VALOR_MINIMO_TRANSACAO = 5000.0
PRODUCAO_ANO_MINIMO = 2019
RANDOM_STATE = 42
