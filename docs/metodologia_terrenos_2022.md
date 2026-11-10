# Exploração de dados ITBI (São Paulo, 2022) — Terrenos: modelo de previsão e estudo de auditoria

**Data da análise:** 13/09/2026
**Fonte dos dados:** `GUIAS_DE_ITBI_PAGAS_12-2022.xlsx` — guias de ITBI pagas, Prefeitura de São Paulo, ano-calendário 2022 (12 abas mensais + aba de explicações de campos)
**Nota de escopo:** esta base é de São Paulo, não de Belém/PA. Serviu como piloto metodológico enquanto a viabilidade de dado equivalente para Belém (LAI/SEFIN) não é resolvida.

---

## 1. Objetivo

Avaliar se os dados de ITBI têm utilidade para (a) um modelo de previsão de valor de terrenos e (b) um estudo de auditoria/identificação de transações potencialmente fraudulentas, como insumo de conhecimento para perícia.

---

## 2. Funil de depuração

A categoria "Uso (IPTU) = TERRENO" do ITBI bruto está fortemente contaminada por vendas de unidades em empreendimentos ainda não individualizados no cadastro fiscal (o apartamento é registrado sob o SQL do terreno-mãe durante a incorporação). Isso foi o achado metodológico central da exploração.

| Etapa | Registros | Observação |
|---|---|---|
| Total ITBI pago (ano) | 163.903 | todas as naturezas e usos |
| Uso (IPTU) = TERRENO | 26.608 | inclui doação, permuta etc. |
| + Natureza = Compra e venda | 25.350 | filtra para transação de mercado |
| + Fração Ideal = 1,0 (lote inteiro) | 24.922 | exclui frações de condomínio |
| Filtro de texto no Complemento (AP/APTO/TORRE/BLOCO) | 6.843 (25,7%) | **insuficiente** — perde padrões como "Ap12" sem espaço, "APARTAMENTO" por extenso, numeração pura de unidade |
| **Filtro por SQL único no ano** | **1.540 (6,2%)** | critério validado: 100% dos SQLs repetidos têm Área do Terreno idêntica em todas as linhas — prova de que são unidades do mesmo empreendimento, não terrenos distintos |
| − Data de Transação fora de 2022 ("regularização tardia") | 1.522 | 18 registros com datas de 1995–2021, registrados/pagos em 2022 |
| − Confirmados não-mercado (ver §4) | **1.485** | base final de modelagem |

**Conclusão do funil:** o filtro por unicidade de SQL (validado pela identidade de área) é o critério correto e substancialmente mais rigoroso que qualquer heurística textual. Qualquer uso futuro de "Uso=TERRENO" sem esse filtro está contaminado.

---

## 3. Qualidade dos dados (base final, n=1.485)

- **Bairro ausente em ~28%** dos registros; **CEP 100% completo** (1.145 CEPs distintos) — geocodificar por CEP é o caminho, não confiar em Bairro.
- **Valor Venal de Referência não é âncora confiável**: no universo bruto de terreno (lote inteiro, pré-filtro), a razão mediana Valor de Transação/Valor Venal de Referência foi de ~19,5x — a referência fiscal está muito defasada frente ao mercado para terrenos.
- **Testada**: ausente/zero para ~13% dos casos, mas isso é esperado para lotes de esquina/interior segundo o dicionário de campos da própria planilha — não é dado faltante no sentido usual.
- **Preço/m² (base final)**: mediana R$899/m², P25=R$334, P75=R$1.630, min R$2,79, max R$51.875 — distribuição ainda bastante assimétrica, plausível para heterogeneidade real de submercados dentro de uma megacidade.

---

## 4. Estudo de auditoria (candidatos a transação não-mercado/fraude)

### 4.1 Métodos testados
- **K-means (k=3)** sobre log(área) e log(preço/m²) padronizados: isolou sozinho um cluster de 43 registros com valor declarado essencialmente nulo (mediana R$0,01/m²), sem que isso fosse pedido explicitamente.
- **DBSCAN** (eps=0,3, min_samples=5): mais adequado para achar anomalia por densidade; 50 pontos de ruído dentro do universo restrito a 2022 (3,3%), capturando tanto o lado barato quanto o caro.
- **Score de suspeição por regressão robusta** (Huber, log(valor) ~ log(área) + bairro agrupado, mín. 8 obs/categoria): resíduo padronizado (z-score) contra valor esperado. 37 registros com z < -2; 0 registros com z > 2 (assimetria do próprio resíduo, dominada pelos casos de valor simbólico).
- Sobreposição DBSCAN × score de regressão: 11 de 50 — os dois métodos capturam sinais parcialmente distintos, tratados como evidências complementares, não redundantes.

### 4.2 Achado central — e sua reversão pela checagem fiscal
37 registros com valor declarado simbólico (R$0,01 a R$1.000, para terrenos de R$70 mil a R$1 milhão pelo valor venal). Padrões notáveis: seis transações de R$1,00 exato (três em sequência de SQL no Itaim Paulista); duas transações idênticas (mesma rua, mesmo valor R$500, mesma data, SQLs consecutivos) na R Marquês de Sabará.

**Checagem decisiva:** comparando `Base de Cálculo adotada` com `Valor Venal de Referência (proporcional)` — **100% dos 37 casos foram tributados sobre o valor venal de referência**, não sobre o valor simbólico declarado. **Não há evidência de perda de arrecadação** nesses casos; o piso fiscal municipal já opera como proteção.

**Reenquadramento:** o problema não é sonegação bem-sucedida, é **classificação incorreta de natureza de transação** — operações societárias (integralização de capital, reestruturação, transferência entre partes relacionadas) sendo registradas como `"1.Compra e venda"` quando não são transações de mercado. Isso confirma que a exclusão desses 37 registros da base de modelagem (§3, §5) está correta, e sinaliza um problema de qualidade de dado cadastral (não necessariamente do escopo de uma auditoria fiscal).

### 4.3 Em aberto (não executado)
- Verificar se o mesmo mecanismo de piso protegeu os 18 registros de "regularização tardia" (datas fora de 2022).
- Investigação manual dos 10 casos de valor **alto** suspeito (R$21 mil–52 mil/m²) — sem mecanismo de piso equivalente do lado superior; candidatos a erro de dígito ou terreno genuinamente premium (ex. R Barata Ribeiro, Av Ibirapuera, Av Brig Luís Antônio).

---

## 5. Modelo de previsão de valor de terreno — baseline

**Base:** 1.485 registros (após exclusão dos 37 casos não-mercado confirmados). Split treino/teste 80/20 (random_state=42).

| Configuração de features | Modelo | R² (log) | Erro % mediano absoluto |
|---|---|---|---|
| log(área) + bairro agrupado | Huber (robusta) | 0,230 | 65,8% |
| log(área) + bairro agrupado | OLS | 0,225 | 66,9% |
| log(área) + bairro agrupado | Random Forest | 0,202 | 64,5% |
| log(área) + CEP (5 dígitos) | Huber | 0,217 | 60,3% |
| log(área) + testada + bairro | Huber | 0,225 | 65,0% |
| log(área) + testada + CEP (5 dígitos) | Huber | 0,215 | 61,1% |

**Diagnóstico:** todas as variações convergem para o mesmo teto (R² 0,20–0,23; erro ~60–66%). Testada não agrega poder preditivo relevante. CEP de 5 dígitos melhora ligeiramente o erro percentual mas não o R². O teto é estrutural, não um problema de escolha de modelo:

1. Localização categórica (bairro/CEP) é grosseira demais — falta coordenada geográfica real.
2. Ausência de variáveis de zoneamento/potencial construtivo (coeficiente de aproveitamento), que tipicamente dominam o valor de terreno urbano.
3. Amostra (1.485) espalhada por uma megacidade heterogênea, em um único ano.

---

## 6. Próximos passos identificados (não executados)

1. **Geocodificar** CEP/endereço para latitude/longitude real (prioridade mais alta para destravar o teto de R²).
2. Cruzar com dado de **zoneamento** (PDE/LOA de São Paulo, dado aberto municipal).
3. Avaliar **puxar múltiplos anos** de ITBI para aumentar densidade amostral por submercado.
4. Fechar a frente de auditoria: checar piso fiscal no grupo de regularização tardia; investigar manualmente os 10 outliers de valor alto.
5. Replicar o funil de depuração (SQL único + área idêntica) em qualquer nova leva de dados de ITBI antes de qualquer análise — é pré-requisito, não etapa opcional.
