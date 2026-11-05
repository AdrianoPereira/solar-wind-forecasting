# Previsão orientada ao TIMSE

Experimento de 05/11/2026 (`src/benchlab_forecasting/timse_blend.py`).

O ranking do torneio usa o TIMSE, mas a submissão atual foi otimizada para o RMSE. A nota
dos organizadores sobre o TIMSE (fórmula em [timse.md](timse.md)) não diz sobre qual
conjunto ele é calculado: o período ao vivo inteiro, janelas ou cada emissão. Também não
diz como escolher os limiares, se valores observados distintos ou quantis. Por isso este
experimento escolhe uma previsão **robusta a essas leituras**.

## Método

1. **Modelo TIMSE.** O LightGBM do híbrido, com os hiperparâmetros ajustados, foi
   retreinado com os pesos do TIMSE (`timse_weights`). O número de árvores foi escolhido
   pelo TIMSE de validação e ficou em 50. Cortes: treino 2011-02 a 2021-12, validação
   2022–2023.
2. **Candidatos na validação:** `(1 − β) · submissão + β · modelo TIMSE + δ`, com β de 0 a
   1 em passos de 0,1 e δ de 0 a 100 km/s em passos de 10.
3. **Leituras do TIMSE** usadas para avaliar cada candidato:
   - somado no período, com limiares em valores únicos ou em 20 quantis;
   - média em janelas de 90 dias, que é a leitura mais parecida com o período ao vivo, de
     ~3 meses (valores únicos e 20 quantis);
   - média por mês;
   - média por emissão;
   - mediana por emissão.
4. **Escolha robusta:** o candidato com o menor pior caso da razão até o melhor candidato
   de cada leitura (*minimax regret*). Foi calculada com todas as leituras e sem a mediana
   por emissão.
5. **Teste** (2024-01 a 2026-10) avaliado uma vez.

```bash
mkdir -p data/slurm_logs && sbatch scripts/slurm/timse_blend.sbatch   # ~1,5 min em 32 CPUs
```

## Escolhas na validação

| Critério | β | δ (km/s) |
| --- | ---: | ---: |
| Robusto, sem a mediana por emissão | 0,6 | 40 |
| Robusto, com todas as leituras | 0,4 | 20 |
| Melhor para cada leitura somada, em janelas ou por mês | 0,6–0,7 | 30–70 |
| Melhor para a média por emissão | 0,7 | 10 |
| Melhor para a mediana por emissão | 0,0 | 10 |

Só a mediana por emissão prefere a previsão atual. Todas as outras leituras preferem
misturar 60–70% do modelo TIMSE e somar uma constante.

## Teste

Menor é melhor. As colunas somadas, em janelas e por mês estão em milhões de (km/s)³; as
colunas por emissão, em milhares.

| Previsão | RMSE | Somado, valores únicos | Somado, 20 quantis | Janelas de 90 dias | Por mês | Por emissão (média) | Por emissão (mediana) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Submissão atual (orientada ao RMSE) | **74,4** | 70,6 | 23,9 | 28,8 | 14,4 | 3.178 | **326** |
| Modelo TIMSE sozinho | 118,1 | 51,1 | 13,1 | 18,9 | 9,1 | 2.233 | 882 |
| **Robusta, sem mediana** (β 0,6, δ 40) | 118,0 | 46,7 | **12,2** | **17,6** | **8,6** | **2.168** | 951 |
| Robusta, com todas (β 0,4, δ 20) | 91,3 | 55,1 | 15,5 | 21,1 | 10,1 | 2.216 | 518 |
| Constante 700 km/s (referência) | 277,8 | **42,1** | 17,7 | 19,4 | 14,3 | 7.496 | 7.227 |

As colunas de janelas de 90 dias usam limiares em valores únicos. Com 20 quantis, a
submissão atual tem 14,4, a robusta sem mediana 8,0 e a robusta com todas 9,6.

**Leitura:**

- **Robusta, sem mediana.** Em comparação com a submissão atual, o TIMSE cai de **32% a
  49%** em todas as leituras somadas, em janelas, por mês e na média por emissão. Na mediana
  por emissão, piora quase 3 vezes. O RMSE vai de 74 para 118 km/s.
- **Robusta, com todas.** É um meio-termo: ganha de 22% a 35% nas leituras somadas e na
  média por emissão, e piora 59% na mediana por emissão. O RMSE fica em 91 km/s.
- **As escolhas da validação se repetiram no teste**, mesmo com o teste no máximo solar: a
  ordem dos candidatos é a mesma nos dois períodos.
- **A constante de 700 km/s só ganha** no TIMSE somado com limiares em valores únicos. Isso
  mostra como essa leitura premia o viés para cima.

## Decisão pendente

Qual versão submeter depende de como o TIMSE oficial é agregado. Pergunta aos
organizadores:

- O TIMSE do ranking é calculado uma vez sobre todas as previsões do período ao vivo, ou
  por emissão (ou por dia) e depois agregado? Se agregado, por média ou mediana?
- Como os limiares são escolhidos: todos os valores observados distintos ou quantis? Se
  quantis, quantos?

Se for somado, em janelas ou com média por emissão, a versão robusta sem mediana é muito
melhor que a atual. Se for mediana por emissão, a atual é a melhor.
