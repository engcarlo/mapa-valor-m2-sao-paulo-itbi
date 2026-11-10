# Changelog e revisão de código

## 0.1.0 (reestruturação do checkpoint `itbi_analises`)

### Bugs corrigidos
* **Perda do ano de construção (ACC) em 2019-2022 e 2024.** O cabeçalho desses arquivos chega com duas colunas `ACC (IPTU)`; a penúltima é a descrição do padrão (texto)
  e a última é o ano. O loader antigo renomeava `.1` e mantinha a *primeira* duplicata, descartando o ano e gravando texto no campo. Agora os nomes são
  atribuídos por posição (`carregamento.py`), com teste de regressão. Cobertura de ACC válido: 100% em todos os anos.
* **Abas com 29 colunas** (MAR-2008, FEV-2013, OUT-2023 e outras): uma única célula preenchida em >7 mil linhas; agora descartada com aviso, e o código falha alto se houver dado real.
* **Typo "Descrição do pardão" (2026)** e aba OUT-2024 com 27 colunas: tratados pela atribuição posicional.
* `apartamento_modelo.preparar_features` convertia `Padrão` com `astype(str)` e produzia "22.0" (float) em 2010 e 2012; usa `normalizar_padrao`.
* `MODELOS_PADRAO` compartilhava a mesma instância de estimador entre chamadas; agora `clone()`.
* `relatorio_qualidade.pct_sql_unico` tinha uma expressão circular (`index.isin(df[...])`); simplificada.
* Anotação de retorno errada em `carregar_coordenadas_cep` (declarava `Series`, devolve `DataFrame`).
* Colunas de tipo misto (valor venal, base de cálculo) vindas do cache em texto: convertidas em `preparar_apartamentos`.

### Desempenho
* Filtro de apartamento aplicado **durante a leitura** de cada aba (antes carregava os ~3 milhões de linhas para depois filtrar; estourava o tempo do ambiente).
* Cache parquet por ano. Geocodificação: ponto-em-polígono **por CEP distinto** e vetorizado (`STRtree.query` do shapely 2): de minutos para ~1 s em 1,1 milhão de linhas.

### Estrutura
* Pacote instalável `src/itbi_sp/` (antes: módulos soltos com imports relativos ao diretório), CLI `itbi-sp`, caminhos em `config.toml` (nada hardcoded), constantes e decisões em `config.py`.
* Dados de apoio empacotados (`data/`), manifesto SHA-256 dos dados brutos, `.gitignore`, licença MIT, CI de testes, 17 testes.
* Módulos de terreno movidos para `itbi_sp.terreno` (exploratório, fase anterior).

### Novo
* Camada de ML (`ml.py`, `experimento_ml.py`): validação temporal, métricas por classe, ablações, recalibração de viés e intervalo conformal por classe. Ver `docs/ML.md`.

### Pendências conhecidas
* A mediana do mapa por período é aproximada (média das medianas anuais ponderada por n); não existe recombinação exata de medianas sem guardar os valores.
* Cobertura de geocodificação menor em 2020+ (ver `docs/ML.md`).
