# Dados

Nada aqui é gerado pelo código, exceto `cache/`. O que está versionado no Git e o que não está:

| Caminho | No Git? | Fonte |
|---|---|---|
| `raw/bcdata_sgs_21340.csv` | sim (pequeno) | Banco Central, série SGS 21340 (IVG-R). API: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.21340/dados?formato=json` |
| `raw/itbi/*.xlsx` (21 arquivos, ~480 MB) | **não** | Prefeitura de São Paulo, "Guias de ITBI pagas" (portal de dados abertos da SF/SP), um arquivo por ano, 2006 a 2026 (parcial). |
| `raw/census_code_cep_coordinates.csv` (21 MB) | **sim** | Kaggle, `millengustavo/ibge-censo-cep-coordenadas` (POSTCODE, LON, LAT, CD_GEOCODI). Dataset de terceiros versionado aqui para reprodutibilidade; **confira a licença na página do Kaggle** antes de redistribuir o repositório em outros contextos. |
| `cache/` | não | gerado por `itbi-sp base` (parquet por ano, ~20 s por ano na primeira leitura) |

`raw/MANIFEST.sha256` lista tamanho e SHA-256 de cada arquivo bruto, para conferir que você tem exatamente os mesmos
dados usados nos resultados publicados.

## Correspondência arquivo → ano (conferir pelo conteúdo!)

* `GUIAS_DE_ITBI_PAGAS__28012026__XLS.xlsx` contém **2025 inteiro** (o nome traz a data de extração).
* `GUIAS_DE_ITBI_PAGAS__27082026__XLS.xlsx` é **2026 parcial** (jan a jul).
* O IVG-R termina em jun/2026: transações posteriores (≈5,9 mil linhas, somando datas antes de mar/2001) são descartadas com aviso.
