# Modelo híbrido: buracos coronais + recorrência + L1

`hybrid.py` treina um único `HistGradientBoostingRegressor` para as 72 horas de
antecedência. Cada linha é um par (emissão, hora prevista), com a hora como variável.
Todas as entradas respeitam o corte em `t0 − 1 h`:

- **L1 (ACE):** última velocidade válida e sua idade, médias de 6, 24 e 72 h, máximo
  de 24 h, tendência de 24 h, densidade e temperatura médias de 24 h;
- **recorrência:** velocidade (média centrada de 7 h) 27 e 54 dias antes de cada hora
  prevista e o erro da recorrência para "agora";
- **buracos coronais (AIA 193 Å):** fração de área em seis faixas de longitude de
  15° (−60° a +30°), |lat| < 60°, limiares 0,25 e 0,35 da mediana do disco, na imagem
  da emissão e nas de 24 h e 48 h antes, mais a idade da imagem.

Velocidade, densidade e temperatura usam o ACE Level 2 onde ele é válido e o produto
operacional no resto (`load_plasma`). Em operação só existe o operacional; a diferença
mediana entre os dois é 2 km/s, com 95% abaixo de 8 km/s.

## Dados e variáveis de buraco coronal

```bash
bash scripts/download_training_data.sh            # ACE Level 2/operacional e AIA, 2011–2026
.venv/bin/python scripts/build_coronal_hole_features.py --workers 48
```

`coronal_holes.py` converte cada pixel em latitude/longitude heliográficas
(projeção ortográfica com B0), descarta μ < 0,2 e pondera por 1/μ, de modo que
áreas reais iguais dão frações iguais no centro e perto do limbo. O lote grava
`data/features/coronal_holes_0193_6h.parquet`: 22.844 de 23.112 emissões de
2011 a out/2026, em cerca de 3 minutos. Filamentos escuros também entram na máscara.

A correlação de Spearman entre a fração de cada faixa e a velocidade X dias depois
tem pico que anda cerca de um dia por faixa (2,5 d em +15°/+30° até 7,5 d em
−60°/−45°), como esperado da rotação de 13°/dia mais o trânsito até L1. O pico é
0,33–0,36, comparável à recorrência (0,40).

## Experimento

```bash
OMP_NUM_THREADS=16 .venv/bin/python -m benchlab_forecasting.hybrid experiment \
  --output-dir data/experiments/hybrid_2011_2026
```

Treino com emissões de 2011-02 a 2021-12, escolha do número de árvores por TIMSE na
validação (2022–2023) e teste em 2024-01 a 2026-10 (4.046 emissões a cada 6 h com
alvo). O teste não foi usado em nenhuma escolha. Depois de 2024-07-09 os alvos vêm
só do produto operacional, porque o Level 2 termina. Cada variante leva de 1 a 2 min
em CPU. A variante `mse_no_ch` está em `data/experiments/hybrid_2011_2026_mse_no_ch/`.

| Modelo | RMSE | 1–24 h | 25–48 h | 49–72 h | TIMSE total (M) | TIMSE mediano por emissão (mil) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistência | 97,8 | 60 | 101 | 122 | 64,0 | 556 |
| Recorrência de 27 dias | 107,8 | 107 | 108 | 109 | 59,9 | 929 |
| Híbrido MSE, sem buracos coronais | 78,6 | 57 | 83 | 92 | 81,2 | 349 |
| **Híbrido MSE** | **76,0** | 57 | **80** | **88** | 74,4 | 374 |
| Híbrido com pesos TIMSE | 109,5 | 92 | 114 | 121 | **49,6** | 736 |

- O **híbrido MSE** é a melhor previsão: erra menos que persistência e recorrência
  em todos os horizontes e tem o menor viés em todas as faixas de velocidade.
- Os **buracos coronais** reduzem o RMSE de 78,6 para 76,0 km/s e, quando o
  observado passa de 550 km/s, de 166,8 para 148,5 km/s (−11%).
- O **híbrido com pesos TIMSE** ganha no TIMSE total só porque prevê cerca de
  100 km/s acima no vento lento (viés de +140 km/s abaixo de 350 km/s).

## Comportamento do TIMSE

`timse_weights` reescreve o TIMSE como MSE ponderado: Σ wᵢ (ŷᵢ − yᵢ)², com
wᵢ = Σ_{j: v*ⱼ ≤ yᵢ} Δⱼ/|Sⱼ| (teste em `tests/test_timse.py`). A implementação
confere com o slide (`IMG_1046.PNG`). Calculado sobre todo o período de teste:

- 75–88% do TIMSE vem dos limiares acima de 700 km/s (1,7% dos pares, eventos como
  2024-05-12, 2025-06-01 e 2026-01-19/20);
- somar uma constante melhora o score: híbrido MSE + 100 km/s dá 45,5 M e + 200 km/s
  dá 33,3 M, com RMSE de 210 km/s;
- uma **previsão constante de 700 km/s dá 42,1 M**, melhor que qualquer modelo.

Calculado por emissão (72 valores) e depois resumido, a constante de 700 km/s fica
entre as piores (média 7.518 mil) e o híbrido MSE tem a melhor mediana. Um viés de
+50 km/s ainda melhora a média (de 3.177 para 2.353 mil), que continua dominada
pelos eventos extremos.

Antes de otimizar para o score oficial é preciso confirmar com a organização sobre
qual conjunto o TIMSE é calculado (todo o período, por emissão, por janela) e se há
alguma normalização. Do jeito que o slide está escrito, o agregado do período inteiro
recompensa previsões enviesadas para cima e depende de poucos eventos extremos.

## Figuras

```bash
.venv/bin/python scripts/plot_hybrid_results.py          # resumo do teste
E=data/experiments/hybrid_2011_2026
for o in 2024-04-25T00:00:00Z 2024-02-03T18:00:00Z 2024-05-10T00:00:00Z; do
  .venv/bin/python scripts/plot_single_forecast.py --predictions $E/test_predictions_all.parquet \
    --origin $o --models hybrid_mse hybrid_timse recurrence hybrid_mse_no_ch \
    --observations data/features/ace_speed_combined_2011_2026.parquet --output-dir $E/figures
done
```

Ficam em `data/experiments/hybrid_2011_2026/figures/` (PNG e PDF): `rmse_by_lead`,
`fixed_lead_window` (fev–mar/2024, 24 h e 72 h), `coronal_hole_lags`, `timse_behavior`
e três emissões com as 72 h observadas completas. A emissão de 03/02/2024 18 UTC foi
escolhida como a de maior ganho dos buracos coronais entre as chegadas de vento rápido
(subida > 150 km/s, pico < 800 km/s); nessas 41 emissões o ganho mediano é 5 km/s e há
ganho em 66% delas. 25/04/2024 é uma emissão típica e 10/05/2024 é a supertempestade de CMEs.
`ace_speed_combined_2011_2026.parquet` é a série de velocidade usada como alvo (Level 2
onde válido, operacional no resto).

Nas primeiras ~10 h de antecedência o híbrido erra mais que a persistência
(35 contra 21 km/s em 1 h); a partir de ~12 h ele é o melhor em todos os horizontes.

## Robustez da validação

Verificações feitas sobre os modelos salvos, sem retreinar:

- **Entradas só do produto operacional** (como em tempo real), em vez de Level 2 +
  operacional: RMSE do híbrido 76,0 → 76,2 km/s; persistência 97,8 → 98,2;
  recorrência 107,8 → 107,2. A troca de produto não muda as conclusões.
- **Alvo Level 2 × operacional:** até 2024-07-09 (54 mil pares) o híbrido tem
  68,2 km/s contra 81,9 da persistência; depois (167 mil pares, só operacional),
  78,4 contra 102,5.
- **Bootstrap em 39 blocos de 27 dias** (IC 95%, 300 reamostragens): ganho do
  híbrido sobre a persistência 21,8 km/s [16,1; 26,7]; sobre a recorrência 31,8
  [24,5; 39,2]; dos buracos coronais 2,6 [−0,1; 5,3] no geral e 4,3 [0,2; 8,0]
  entre 49 e 72 h. O ganho das imagens é pequeno e está no limite da significância.

Ainda não verificado: as variáveis de buraco coronal vêm do arquivo sinóptico com
latência suposta de 60 min; o produto NRT usado em operação (disponível desde
dez/2024) não foi comparado com o arquivo. Os modelos anteriores (Prophet ajustado,
Patch Transformer) não foram avaliados neste mesmo teste. Há uma única divisão
temporal, uma semente e hiperparâmetros fixos.

## LightGBM

`--model lightgbm` troca o `HistGradientBoostingRegressor` pelo LightGBM (extra
`boosting`) com os mesmos hiperparâmetros: `num_leaves` = `max_leaf_nodes`,
`min_child_samples` = `min_samples_leaf`, `reg_lambda` = `l2_regularization` e a
mesma taxa de aprendizado. O número de árvores é escolhido igual, pelo TIMSE de
validação. O treino é determinístico (`deterministic=True`, `force_row_wise=True`).

```bash
uv sync --locked --inexact --extra hybrid --extra boosting
.venv/bin/python -m benchlab_forecasting.hybrid experiment --model lightgbm \
  --variants mse mse_no_ch timse --output-dir data/experiments/hybrid_lightgbm_2011_2026
```

As três variantes rodam em cerca de 1 minuto (contra ~3 minutos do scikit-learn).
No mesmo teste:

| Biblioteca | Variante | RMSE | 1–24 h | 25–48 h | 49–72 h | TIMSE total (M) | TIMSE mediano por emissão (mil) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | MSE | 76,0 | 57,0 | 79,7 | 87,9 | 74,4 | 374 |
| LightGBM | MSE | 76,3 | 56,9 | 80,1 | 88,4 | 75,2 | 370 |
| XGBoost | MSE | 75,7 | 56,7 | 79,5 | 87,6 | 72,2 | 369 |
| CatBoost | MSE | 76,4 | 57,7 | 79,9 | 88,3 | 72,2 | 380 |
| NGBoost | MSE | 77,5 | 57,7 | 81,2 | 90,0 | 77,4 | — |
| scikit-learn | MSE sem buracos coronais | 78,6 | 56,6 | 82,6 | 92,2 | 81,2 | 349 |
| LightGBM | MSE sem buracos coronais | 78,4 | 56,5 | 82,3 | 92,0 | 79,4 | 340 |
| XGBoost | MSE sem buracos coronais | 79,0 | 56,8 | 83,0 | 92,9 | 82,3 | 358 |
| CatBoost | MSE sem buracos coronais | 79,2 | 57,5 | 83,3 | 92,7 | 82,1 | 361 |
| NGBoost | MSE sem buracos coronais | 78,9 | 56,9 | 82,9 | 92,5 | 81,4 | — |
| scikit-learn | pesos TIMSE | 109,5 | 91,8 | 113,8 | 120,9 | 49,6 | 736 |
| LightGBM | pesos TIMSE | 109,9 | 93,0 | 113,7 | 121,0 | 49,4 | 736 |
| XGBoost | pesos TIMSE | 108,8 | 90,7 | 112,6 | 120,9 | 49,2 | 729 |
| CatBoost | pesos TIMSE | 111,7 | 93,0 | 116,4 | 123,4 | 51,9 | 773 |
| NGBoost | pesos TIMSE | 95,5 | 74,4 | 100,5 | 108,2 | 57,5 | — |

A diferença de RMSE entre as bibliotecas (variante MSE) é −0,3 km/s, IC 95% [−0,9; 0,2]
no bootstrap em blocos de 27 dias: não há diferença. O ganho dos buracos coronais no
LightGBM é 2,1 km/s [−0,7; 4,9], coerente com o do scikit-learn.

### XGBoost

`--model xgboost` usa o `xgboost-cpu` (extra `boosting`, sem as bibliotecas de GPU)
com `tree_method="hist"` e crescimento por folha (`grow_policy="lossguide"`,
`max_leaves` = `max_leaf_nodes`, sem limite de profundidade), como o LightGBM. O
mínimo por folha vira `min_child_weight`, que com erro quadrático e pesos unitários é
o número de pares; com pesos TIMSE o significado muda um pouco. Como no LightGBM, o
modelo é treinado uma vez e prevê com as primeiras N árvores. O job SLURM roda as três
variantes em 46 s:

```bash
MODEL=xgboost sbatch scripts/slurm/hybrid_experiment.sbatch   # grava data/experiments/hybrid_xgboost_2011_2026
```

O XGBoost MSE é o melhor RMSE entre as bibliotecas na configuração padrão (o LightGBM ajustado, abaixo, chega a 75,3), mas a diferença é pequena: 0,6 km/s
[0,1; 1,1] melhor que o LightGBM e 0,3 [−0,1; 0,7] que o scikit-learn (bootstrap em
39 blocos de 27 dias). O ganho dos buracos coronais é 3,3 km/s [1,0; 5,5]. A troca de
biblioteca, como esperado, muda menos de 1 km/s.

### CatBoost

`--model catboost` (extra `boosting`) usa árvores por folha (`grow_policy="Lossguide"`,
`max_leaves`, `min_data_in_leaf`, `l2_leaf_reg`) em vez das árvores simétricas padrão
do CatBoost, para o tamanho da árvore significar o mesmo que nas outras bibliotecas.
`colsample` vira `rsm` e `subsample` usa amostragem de Bernoulli. O job leva 2 min 24 s:

```bash
MODEL=catboost sbatch scripts/slurm/hybrid_experiment.sbatch   # grava data/experiments/hybrid_catboost_2011_2026
```

O CatBoost MSE fica 0,7 km/s [0,2; 1,1] pior que o XGBoost e empata com o LightGBM
(−0,1 [−0,6; 0,4]) e o scikit-learn (−0,4 [−1,0; 0,2]). O ganho dos buracos
coronais é 2,8 km/s [0,2; 5,6]. A variante MSE parou no limite de 600 árvores, então
mais árvores ou uma taxa de aprendizado maior podem ajudar um pouco.

### NGBoost

`--model ngboost` prevê uma distribuição normal por par (média e desvio, colunas
`hybrid_*_scale` nas previsões). As árvores padrão do scikit-learn levavam 122 s por
iteração; `ngboost_trees.py` usa como base uma árvore do LightGBM discretizada uma vez,
o que dá 1,7 s por iteração com o mesmo algoritmo (gradiente natural e busca do passo).
O job leva 1 h 56 min:

```bash
MODEL=ngboost MAX_ITER=2000 sbatch --time=05:00:00 scripts/slurm/hybrid_experiment.sbatch
```

A média do NGBoost é a pior das cinco bibliotecas (77,5 km/s) e a variante MSE parou
no limite de 2000 iterações. Os intervalos são estreitos demais: com buracos coronais,
os intervalos de 50%, 80% e 95% cobrem 27%, 47% e 65% das observações (desvio mediano
27 km/s contra erro absoluto médio de 53). Sem buracos coronais a cobertura é 46%, 74%
e 88%. O desvio ordena bem o erro (o quintil de maior desvio tem erro médio 71 km/s,
o de menor, 41), mas precisa de recalibração antes de ser usado.

Nos nós de login, rode os testes com poucas threads. Sem isso, as bibliotecas abrem
uma thread por núcleo e o processo é morto pelo limite de 30 min de CPU:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python -m unittest tests.test_hybrid
```

## Busca de hiperparâmetros (LightGBM)

Busca aleatória de 40 configurações (`SEARCH_SPACE` em `hybrid.py`): taxa de
aprendizado, folhas, mínimo por folha, L2, fração de variáveis e de linhas. A
tentativa 0 é a configuração de referência. Cada uma é escolhida, com o número de
árvores, pelo RMSE na validação de 2022–2023; o teste é avaliado uma vez no fim. A
busca é retomável (`trials.jsonl`) e precisa rodar via SLURM, porque o nó de login
encerra processos após 30 min de CPU:

```bash
mkdir -p data/slurm_logs && sbatch scripts/slurm/tune_lightgbm.sbatch
```

O job 752793 (32 CPUs) levou 2 h 19 min. Resultados em
`data/experiments/hybrid_lightgbm_tuning/`.

- **Validação:** referência 68,33 km/s, melhor 67,88 km/s (`learning_rate=0.03`,
  63 folhas, `min_samples_leaf=500`, L2 1, `colsample=0.5`, sem subamostragem de
  linhas). As cinco melhores usam `colsample=0.5`, e quase todas usam 63 folhas. A
  diferença entre as tentativas é pequena: 34 das 40 ficam entre 67,9 e 68,5 km/s e a pior tem 69,1.
- **Teste:**

| Modelo | RMSE | 1–24 h | 25–48 h | 49–72 h | TIMSE total (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM ajustado | 75,3 | 57 | 79 | 87 | 74,6 |
| LightGBM referência (árvores por RMSE) | 75,6 | 56 | 79 | 88 | 74,9 |
| LightGBM ajustado sem buracos coronais | 78,6 | 58 | 82 | 92 | 81,7 |
| Persistência | 97,8 | 60 | 101 | 122 | 64,0 |

No bootstrap em 39 blocos de 27 dias, o ganho do ajuste sobre a referência é
0,3 km/s, IC 95% [−0,3; 1,0], ou seja, não significativo. O ganho dos buracos
coronais é 3,4 [1,4; 5,5], agora significativo. Escolher o número de árvores pelo
RMSE em vez do TIMSE já leva a referência de 76,3 para 75,6. Os hiperparâmetros estão
perto do limite do que essas variáveis permitem; ganhos maiores devem vir de dados
novos ou de outra formulação, não de mais busca.
