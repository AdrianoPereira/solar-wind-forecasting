# Outros modelos com attention

Três arquiteturas acrescentadas ao lado da [GRU com attention](attention_gru.md).
Todas reutilizam o fluxo de `lstm.py`: as mesmas sete variáveis com indicadores
de ausência, a janela causal com `available_at`, a normalização ajustada só no
treino, o MSE mascarado, a previsão como correção à última velocidade disponível
e o mesmo formato de checkpoint. A arquitetura é gravada no checkpoint, e carregar
um arquivo com a classe errada gera erro.

| Arquitetura | Módulo / CLI | Histórico | Tokens de entrada |
| --- | --- | ---: | --- |
| `attention_lstm` | `attention_lstm.py` / `attention_lstm_cli` | 168 h | estados horários da LSTM |
| `transformer` | `transformer.py` / `transformer_cli` | 168 h | 168 horas, um token por hora |
| `patch_transformer` | `transformer.py` / `patch_transformer_cli` | 720 h | 30 blocos de 24 h |

Nas três, cada uma das 72 horas previstas tem uma consulta aprendida, somada ao
último estado, que usa `MultiheadAttention` sobre todo o histórico codificado.
Como na GRU com attention, `residual_scale` soma o último estado ao contexto antes
da projeção de saída, e a camada de saída começa zerada (persistência).

- **LSTM com attention**: igual à GRU com attention, só troca o encoder. Os
  hiperparâmetros e os padrões são os mesmos.
- **Transformer**: projeção linear de cada hora, posição aprendida e `layers`
  camadas de self-attention com pré-normalização (`feedforward_size=64`,
  `layers=2`). Não há recorrência: a ordem vem só das posições aprendidas.
- **Patch Transformer**: cada token concatena as 24 horas de um dia com todas as
  variáveis (`patch_hours=24`). O último bloco termina na emissão. Com 720 h,
  o histórico contém as horas de 27 dias antes de todo o horizonte previsto
  (de `t0 − 647 h` a `t0 − 576 h`), de modo que a attention pode aprender a
  recorrência da rotação solar, que até agora é o baseline mais forte em TIMSE.
  `lookback_hours` precisa ser múltiplo de `patch_hours`.

`hidden_size` precisa ser divisível por `attention_heads`. Todos os campos da
configuração viram opções da CLI (`--layers`, `--feedforward-size`,
`--patch-hours`, `--residual-scale` etc.).

## Reproduzir os pilotos

Mesmo protocolo do piloto da GRU com attention: emissões de janeiro a agosto de
2019 para treino, época escolhida em setembro e avaliação das 28 emissões de
15–21 de novembro, com os mesmos checkpoints de LSTM/GRU e o Prophet ajustado.
Troque `MODELO` por `attention_lstm`, `transformer` ou `patch_transformer`, e use
um diretório novo e vazio ao repetir.

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked --extra forecast --extra lstm \
  python -m benchlab_forecasting.MODELO_cli experiment \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --train-start 2019-01-08 \
  --validation-start 2019-09-01 --validation-hours 336 \
  --test-start 2019-11-15 --test-hours 168 \
  --lstm-model data/experiments/lstm_2019/model.pt \
  --gru-model data/experiments/gru_2019/model.pt \
  --prophet-config data/experiments/prophet_tuning_2019/best_config.json \
  --output-dir data/experiments/MODELO_2019_nov_reproduced
```

Para prever a partir de um checkpoint, use o subcomando `forecast` como na
[GRU com attention](attention_gru.md#prever-a-partir-do-checkpoint).

## Resultado executado

Pilotos executados em 27/10/2026 com os padrões de cada arquitetura
(`residual_scale=0`, 2 threads em CPU). O arquivo SWEPAM do protocolo começa em
01/01/2019; com 720 h de histórico, o treino do Patch Transformer só pode começar
em **31/01/2019** (cerca de três semanas a menos de emissões de treino que os
demais). Os resultados ficam em `data/experiments/{attention_lstm,transformer,patch_transformer}_2019_nov/`.

Os modelos com cobertura completa foram comparados nos mesmos **1.778 pares**
(os mesmos da tabela da [GRU com attention](attention_gru.md#resultado-executado)):

| Modelo | RMSE (km/s) | TIMSE ((km/s)³) | Época | Parâmetros | Treino |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prophet ajustado anteriormente | 82,83 | 994.817 | — | — | — |
| GRU com attention, sem atalho | 93,22 | 5.016.157 | 35 | 11.169 | 60 s |
| **Transformer** | **72,64** | 5.039.833 | 11 | 29.569 | 108 s |
| Prophet original | 76,43 | 5.074.253 | — | — | — |
| **Patch Transformer (720 h)** | 88,32 | 5.168.391 | 10 | 35.457 | 10 s |
| LSTM | 74,78 | 5.234.528 | — | — | — |
| GRU | 77,79 | 5.748.262 | — | — | — |
| GRU com attention, `residual_scale=1` | 83,81 | 6.881.059 | 13 | 11.169 | 29 s |
| Persistência | 89,67 | 7.344.740 | — | — | — |
| **LSTM com attention** | 104,95 | 9.552.557 | 26 | 12.705 | 40 s |

Nos 1.495 pares comuns que incluem a recorrência de 27 dias, a ordem dos modelos
neurais se mantém; a recorrência (TIMSE 1.684.859) fica atrás só do Prophet ajustado.

- O **Transformer** tem o menor RMSE de todos os modelos neste recorte, mas seu
  TIMSE é praticamente igual ao da GRU com attention e ao do Prophet original.
- O **Patch Transformer** ainda não aproveita a recorrência: seu TIMSE fica longe
  da recorrência simples, e a validação parou cedo (época 10).
- A **LSTM com attention** ficou pior que a persistência neste recorte.
- Nenhuma arquitetura chega perto do TIMSE do Prophet ajustado. Todos os modelos
  neurais são treinados e selecionados por MSE, que não dá o peso extra que o
  TIMSE dá aos erros nas velocidades altas.

Os mesmos alertas da GRU com attention valem aqui: novembro de 2019 já foi
inspecionado, é uma única semana de um único ano e serve apenas como diagnóstico
exploratório, não como teste para escolher o modelo operacional. A inferência
mediana foi de cerca de 10 ms por emissão nas três arquiteturas, sem falhas.

## Figuras das séries temporais

`scripts/plot_best_model_timeseries.py` junta as previsões salvas de vários
experimentos (sem retreinar), ordena os modelos completos por TIMSE nos pares
comuns e desenha o melhor modelo, a melhor rede neural e a recorrência de 27 dias
contra as observações. As figuras (em inglês, PNG e PDF) ficam em
`data/experiments/attention_models_2019_nov_figures/`, com `ranking.csv` e `manifest.json`:

- `best_model_issuances`: as 28 trajetórias de 72 h de cada modelo sobre a série observada;
- `fixed_lead_timeseries`: previsões com 24, 48 e 72 h de antecedência, na hora em que se verificam;
- `selected_issuances`: seis emissões, com as 48 h observadas antes de cada uma.

```bash
E=data/experiments
uv run --locked --extra lstm --group notebooks python scripts/plot_best_model_timeseries.py \
  --predictions $E/transformer_2019_nov/test/predictions.parquet \
  --predictions $E/patch_transformer_2019_nov/test/predictions.parquet \
  --predictions $E/attention_lstm_2019_nov/test/predictions.parquet \
  --predictions $E/attention_gru_2019_nov/test/predictions.parquet \
  --output-dir $E/attention_models_2019_nov_figures
```

As 28 trajetórias do Prophet ajustado praticamente coincidem porque sua
configuração usa o plasma com defasagem de 648 h (27 dias): a previsão para cada
hora depende sobretudo de valores já conhecidos em todas as emissões. Na prática,
ele funciona como uma recorrência suavizada. Lacunas nas observações aparecem
como interrupções das linhas, sem interpolação.

## Treino em vários anos e uma previsão com série completa

Em 2019, o SWEPAM tem um único trecho com mais de 72 h seguidas de observação
(20–23/05, dentro do treino), então nenhuma emissão de teste tem as 72 h observadas
completas. O arquivo `ace_swepam_hourly_partial.parquet` (2000–2010, 98% válido)
permite isso. A GRU com attention, o Transformer e o Patch Transformer foram
treinados em 2003–2007 (7.264 emissões), com época escolhida em 2008 e teste em
28 emissões de 28/04 a 04/05/2010. O ano de 2009 não serve como validação porque
tem duas lacunas de velocidade maiores que 24 h, e a validação recusa emissões
inutilizáveis. O Prophet ajustado usa a configuração escolhida em 2019, sem novo ajuste.

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked --extra forecast --extra lstm \
  python -m benchlab_forecasting.MODELO_cli experiment \
  --data data/ace_hourly/ace_swepam_hourly_partial.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --train-start 2003-01-08 --validation-start 2008-01-01 --validation-hours 8784 \
  --test-start 2010-04-28 --test-hours 168 --threads 4 \
  --prophet-config data/experiments/prophet_tuning_2019/best_config.json \
  --output-dir data/experiments/MODELO_2003_2010
```

RMSE nas 28 emissões de teste: Patch Transformer 107,7 km/s, Transformer 117,2,
GRU com attention 127,2, Prophet ajustado 144,8, persistência 144,5 e Prophet
original 184,6. A melhor época foi a 3ª ou a 4ª nos três modelos, sinal de que a
taxa de aprendizado e a regularização ainda não foram ajustadas para esse volume de dados.

`scripts/plot_single_forecast.py` desenha uma emissão com as 72 h observadas
completas e recusa emissões com alvo ausente. A emissão de 02/05/2010 00 UTC tem
vento lento (305 km/s) seguido de um vento rápido de até 710 km/s, recorrente de
um evento de 03–05/04:

```bash
E=data/experiments
uv run --locked --group notebooks python scripts/plot_single_forecast.py \
  --predictions $E/patch_transformer_2003_2010/test/predictions.parquet \
  --predictions $E/transformer_2003_2010/test/predictions.parquet \
  --predictions $E/attention_gru_2003_2010/test/predictions.parquet \
  --origin 2010-05-02T00:00:00Z \
  --models prophet_tuned attention_gru recurrence transformer patch_transformer \
  --observations data/ace_hourly/ace_swepam_hourly_partial.parquet \
  --output-dir $E/single_forecast_2010
```

| Modelo | RMSE (km/s) | TIMSE ((km/s)³) |
| --- | ---: | ---: |
| Recorrência de 27 dias | 133,1 | 4.997.462 |
| Patch Transformer | 163,2 | 13.733.842 |
| Prophet ajustado | 192,5 | 18.012.783 |
| Transformer | 250,9 | 31.688.637 |
| GRU com attention | 274,7 | 37.471.180 |
| Persistência | 306,2 | 46.318.592 |

Os modelos com 168 h de histórico (Transformer e GRU com attention) não têm como
ver a rotação anterior e ficam próximos do vento lento. O Patch Transformer sobe
ao longo do horizonte, mas subestima o pico. A recorrência acerta a chegada do vento
rápido cerca de 6 h antes e superestima o primeiro pico. Uma única emissão não é
evidência estatística; ela ilustra o comportamento dos modelos.
