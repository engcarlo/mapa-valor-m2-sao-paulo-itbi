# Correção Temporal de Dados de Mercado via IVG-R (BCB) — Metodologia e Exemplo Aplicado

**Aplicável a:** laudos de avaliação de imóveis residenciais urbanos (apto/casa) com finalidade de garantia bancária, quando a amostra de pesquisa de mercado reúne dados coletados em datas diferentes.
**Não aplicável a:** terrenos/glebas (o índice reflete imóveis residenciais financiados, não terreno bruto), nem a imóveis comerciais.
**Fonte do índice:** Banco Central do Brasil, série SGS nº 21340 — Índice de Valores de Garantia de Imóveis Residenciais Financiados (IVG-R). API pública, sem chave: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.21340/dados?formato=json`.

---

## 1. Por que corrigir por um índice específico, e não pela inflação geral

Quando a amostra de mercado reúne dados (comparáveis) coletados em momentos diferentes, a norma exige trazer todos os valores a uma data-base comum antes de qualquer tratamento estatístico ou comparação — é a chamada correção/homogeneização de data. A prática mais comum usa um índice de inflação geral (IPCA, IGP-M). Isso é uma aproximação frágil: **o valor de mercado de imóveis residenciais não anda junto com a inflação geral**, e pode inclusive se mover na direção oposta (ver Exemplo 2, abaixo). O IVG-R, por medir especificamente o valor de imóveis residenciais dados em garantia de financiamento, é uma base mais adequada para essa correção — desde que o imóvel avaliando e a amostra sejam residenciais.

## 2. Fórmula

```
Fator de correção de data = IVG-R(data-base do laudo) / IVG-R(data do dado de pesquisa)

Valor corrigido = Valor observado × Fator de correção de data
```

Onde "data do dado de pesquisa" é a data da transação (ou da coleta, se dado de oferta) do comparável, e "data-base do laudo" é a data de referência da avaliação. Ambas as datas devem ser ajustadas para o primeiro dia do mês correspondente, já que o IVG-R é uma série mensal.

## 3. Exemplo 1 — aplicação simples

Comparável coletado em **15/01/2024**, laudo com data-base em **01/06/2026** (último ponto disponível na série na data desta análise), valor observado de **R$ 450.000,00**.

| | IVG-R |
|---|---|
| jan/2024 | 687,56 |
| jun/2026 | 783,90 |

```
Fator = 783,90 / 687,56 = 1,1401  (+14,01%)
Valor corrigido = R$ 450.000,00 × 1,1401 = R$ 513.053,41
```

## 4. Exemplo 2 — por que isso importa (caso de alerta)

Comparável de **dez/2014** (R$ 300.000,00), corrigido para uma data-base de **dez/2017**:

| | IVG-R |
|---|---|
| dez/2014 | 550,72 |
| dez/2017 | 520,90 |

```
Fator (IVG-R) = 520,90 / 550,72 = 0,9459  (−5,41%)
Valor corrigido (IVG-R) = R$ 300.000,00 × 0,9459 = R$ 283.755,81
```

Nesse período específico (2015–2017), o valor de garantia de imóveis residenciais **caiu** em termos nominais — reflexo da recessão. Se a correção tivesse sido feita pelo IPCA (acumulado de aproximadamente 21,1% no período, segundo o IBGE, considerando os fechamentos anuais de 2015, 2016 e 2017), o resultado seria:

```
Valor corrigido (IPCA, abordagem inadequada para este caso) = R$ 300.000,00 × 1,211 = R$ 363.303,79
```

**Diferença entre as duas abordagens: 28,0%.** Um laudo que corrigisse esse comparável pela inflação geral, em vez de por um índice específico do setor, teria superestimado a referência de mercado em mais de um quarto do valor — nesse cenário específico, um erro grande o suficiente para comprometer o grau de precisão do trabalho.

## 5. Texto sugerido para a seção de metodologia do laudo

> *"Para trazer os dados de pesquisa a uma data-base comum, foi aplicado um fator de correção temporal calculado a partir do Índice de Valores de Garantia de Imóveis Residenciais Financiados (IVG-R), publicado pelo Banco Central do Brasil (série SGS nº 21340). Optou-se por esse indicador, específico do mercado de imóveis residenciais, em detrimento de índices de inflação geral, por refletir mais adequadamente a variação de valor do bem avaliando no período considerado. O fator foi calculado pela razão entre o valor do índice na data-base do laudo e o valor do índice na data de cada dado de pesquisa, aplicado multiplicativamente sobre o valor observado de cada comparável."*

Ajuste o texto para citar as datas e valores efetivamente usados no laudo em questão.

## 6. Limitações a declarar no laudo (não omitir)

- O IVG-R é um índice **nacional**, sem abertura por cidade, bairro ou padrão construtivo — ele capta a tendência agregada do mercado financiado no Brasil, não especificamente da praça do imóvel avaliando. Isso deve ser mencionado como uma limitação da correção aplicada, não apresentado como se fosse um índice local.
- Aplicável apenas quando o comparável (e o imóvel avaliando) é residencial. Não usar para terreno, comercial ou industrial.
- Cobre apenas imóveis **financiados** (SFH/SFI) — é uma amostra específica desse segmento, não o mercado residencial como um todo.
- Sempre registrar no laudo a data de extração do dado do IVG-R e o código da série (21340), para rastreabilidade.

## 7. Função Python reutilizável

```python
import requests

def buscar_ivgr(data_inicial: str, data_final: str) -> dict:
    """Busca a série IVG-R (SGS 21340) do Banco Central entre duas datas
    (formato 'dd/mm/aaaa'). Retorna {data: valor}.
    """
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.21340/dados"
        f"?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return {item["data"]: float(item["valor"]) for item in resp.json()}


def fator_correcao_data(ivgr_data_base: float, ivgr_data_dado: float) -> float:
    """Fator multiplicativo para corrigir um valor observado na
    'data_dado' para a 'data_base'. Ambos os argumentos são o valor do
    IVG-R (não a data) nos respectivos meses.
    """
    return ivgr_data_base / ivgr_data_dado


def valor_corrigido(valor_observado: float, ivgr_data_base: float, ivgr_data_dado: float) -> float:
    """Aplica a correção temporal a um valor observado."""
    return valor_observado * fator_correcao_data(ivgr_data_base, ivgr_data_dado)
```

Uso típico: buscar os dois pontos da série via `buscar_ivgr`, e aplicar `valor_corrigido` a cada dado da amostra antes do tratamento estatístico (Frente 1 da avaliação para bancos).
