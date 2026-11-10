# Camada de ML: melhorando a estimativa por classe de padrão

Código: `src/itbi_sp/ml.py` e `src/itbi_sp/experimento_ml.py`. Reproduzir: `itbi-sp ml --config config.toml`.
Tabelas completas em `docs/resultados/`.

## Pergunta

O baseline hedônico (regressão robusta Huber, `log_valor ~ log_área + distrito + padrão`, efeitos **aditivos**)
estima o valor de um apartamento a partir de onde ele fica e de que padrão construtivo tem. Um modelo que aprenda
**interações** (um bairro valoriza mais o padrão alto do que o baixo) e use atributos extras estima melhor, **classe por classe**?

## Protocolo de validação (decisão metodológica)

| Janela | Período | Linhas |
|---|---|---|
| Treino | jan/2019 a jun/2024 | 306.170 |
| Calibração (viés e intervalo) | jul a dez/2024 | — |
| **Teste** | **jan/2025 a jun/2026** | **99.649** |

* **Validação temporal, não aleatória.** O uso real é estimar o valor *hoje* com o passado. Split aleatório
  mistura vendas do mesmo prédio e mês entre treino e teste e infla a métrica.
* **Métrica principal: MdAPE** (erro percentual absoluto mediano), reportada **por classe de padrão**; o erro global
  esconde as classes raras (20 e 25 somam 1,8% do teste).
* Todos os valores já corrigidos pelo IVG-R para jun/2026. O alvo é `log(valor)` com `log(área)` como feature, de modo que
  o erro percentual do valor é o do preço/m².
* **Sem vazamento:** `Valor Financiado` e `Base de Cálculo` ficam de fora (derivam do preço).

## Resultados (MdAPE %, teste jan/2025 a jun/2026)

| Configuração | Global | 20 baixo | 21 | 22 médio | 23 | 24 alto | 25 luxo |
|---|---|---|---|---|---|---|---|
| **A** baseline Huber (distrito + padrão) | 22,4 | 23,3 | 21,0 | 21,4 | 25,5 | 37,8 | 44,6 |
| A′ baseline Huber (CEP de 5 dígitos), *execução sem coordenadas* | 18,7 | 22,4 | 17,2 | 17,7 | 22,9 | 33,0 | 39,5 |
| B HGB: área + padrão + lat/lon | 19,2 | 25,2 | 16,3 | 18,2 | 23,0 | 31,6 | 37,3 |
| C + idade, terreno, fração ideal | 17,8 | 24,7 | 16,3 | 16,7 | 21,7 | 27,6 | 33,7 |
| D + tempo (meses) | 16,9 | 24,6 | 15,7 | 16,1 | 19,7 | 24,1 | 29,1 |
| E = D com histórico completo (2006+) | 17,2 | 27,3 | 16,6 | 16,4 | 20,1 | 24,6 | 30,4 |
| F = D, um modelo por padrão | 16,8 | 26,2 | 15,7 | 15,9 | 20,2 | 24,5 | 32,2 |
| **H = D + recalibração de viés** (recomendado) | **16,3** | **21,2** | **15,0** | **15,7** | **18,8** | **23,5** | **28,2** |
| I = F + recalibração de viés | 16,3 | 23,4 | 15,5 | 15,4 | 19,0 | 24,5 | 30,8 |
| G = D + valor venal (ablação fiscal, não recomendado) | 15,9 | 22,4 | 15,4 | 15,3 | 17,8 | 22,2 | 26,3 |

HGB = `HistGradientBoostingRegressor` (scikit-learn), early stopping interno.

### O que cada passo ensina

1. **Trocar só o algoritmo não é o ganho.** O A′ (Huber com CEP de 5 dígitos como categoria) fica em 18,7%, melhor que o HGB B
   com lat/lon (19,2%). A comparação com o baseline *por distrito* (22,4%) mistura dois efeitos: algoritmo e finura da localização.
   O ganho honesto do ML sobre um linear forte é de 18,7% para 16,3% (**−13%**); sobre o baseline por distrito, −27%.
2. **Atributos do edifício agregam** (B→C: −1,4 p.p.). A idade (`ano da transação − ACC`) só ficou disponível após corrigir o bug do loader.
3. **Tempo local agrega** (C→D: −0,9 p.p., e mais nas classes altas: 24 vai de 27,6 para 24,1). O IVG-R é nacional; a árvore capta deriva local residual.
4. **O ML não resolve o drift de 21 anos** (D→E piora). Dado antigo não ajuda mesmo com `meses` e árvores: o diagnóstico do
   `diagnostico_temporal.py` continua valendo e o treino de produção fica em 2019+.
5. **Um modelo por classe não ajuda** (D→F ≈ igual; piora na 25 e na 20, que têm poucas observações). Um modelo global com `padrão` como
   número ordinal aproveita as classes vizinhas.
6. **Viés de extrapolação:** os modelos com tempo subestimam o teste (−3,8% mediano; −10% na classe 20), pois a árvore não projeta tendência
   além do treino. Recalibrar com o viés mediano da janela de calibração (sem tocar no teste) leva o viés a ≈ +1% e ganha 0,6 p.p.
7. **Valor venal de referência (G):** o ganho em MdAPE é de só 0,4 p.p. sobre H, embora o R² suba (0,70 → 0,80, concentrado nas caudas).
   Fica fora do modelo por transformá-lo numa calibração do valor fiscal.

## Incerteza: intervalo conformal por classe

Intervalo de predição de 80% via *split conformal* separado por classe (calibrado em jul a dez/2024).
Cobertura empírica no teste: global 79,3%; 20: 82,9; 21: 81,3; 22: 78,9; 23: 80,2; 24: 79,1; 25: 82,2 (alvo: 80%).
O intervalo é **largo**: meia-largura mediana de ±40% (22: ±40%; 24: ±67%; 25: ±107%).

## Limitações (declare-as ao publicar)

* O alvo é o **valor declarado** ao ITBI, que tende a ficar abaixo do preço real e tem ruído (o piso fiscal protege a arrecadação, não o dado).
* ITBI não traz **estado de conservação, andar, vagas, banheiros ou face solar**. Isso limita o piso de erro de qualquer modelo (MdAPE ≈ 16%).
* **Geocodificação:** 91% das linhas ganham coordenada, mas a cobertura cai de ~96% (2006 a 2013) para ~85% (2020+), porque CEPs novos não estão na base de
  coordenadas. Edifícios novos ficam sub-representados na localização contínua.
* Janela de teste de 18 meses, uma única divisão temporal. Um *rolling origin* daria uma estimativa de variabilidade.
* Hiperparâmetros do HGB não foram otimizados (valores padrão razoáveis); há folga a explorar.
* **Isto não substitui um laudo NBR 14653.** É uma referência estatística de mercado a partir de valores declarados.

## Próximos passos

Rolling origin; ajuste de hiperparâmetros com validação temporal; `QuantileRegression` do HGB como alternativa ao conformal; efeito de distrito
variante no tempo no modelo linear (interação distrito × período); aplicar o pipeline a casas e a salas comerciais.
