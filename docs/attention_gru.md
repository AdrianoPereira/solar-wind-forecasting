# GRU com attention por horizonte

`attention_gru.py` implementa uma GRU que conserva um estado para cada uma das
168 horas de entrada. Para cada uma das 72 horas previstas, uma consulta
aprendida usa `MultiheadAttention` sobre esses estados. A saída é uma correção
à última velocidade disponível, como nos modelos LSTM/GRU existentes.

O fluxo de dados, as sete variáveis (velocidade, densidade, temperatura, Bx, By,
Bz e Bt) e seus sete indicadores de ausência são os mesmos da LSTM/GRU. A
normalização é ajustada apenas no treino. A disponibilidade das entradas é
verificada em cada emissão; quando `available_at` não existe, a latência
histórica de 60 minutos é uma hipótese local. Nenhum alvo futuro entra nas
consultas. Os alvos ausentes são mascarados na perda.

`attention_heads=4`, `hidden_size=32` e `residual_scale=0` são os padrões.
`residual_scale=1` acrescenta o último estado da GRU ao contexto de cada
horizonte antes da projeção de saída. Esse parâmetro é salvo no checkpoint;
zero preserva a interpretação dos checkpoints do primeiro piloto.

## Reproduzir o piloto

O comando abaixo treina com emissões de janeiro–agosto de 2019, escolhe a época
em setembro e avalia 28 emissões de 15–21 de novembro. Os checkpoints de LSTM
e GRU foram treinados anteriormente, sem usar novembro. Troque o diretório de
saída por um caminho novo e vazio ao repetir o experimento.

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked --extra forecast --extra lstm \
  python -m benchlab_forecasting.attention_gru_cli experiment \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --train-start 2019-01-08 \
  --validation-start 2019-09-01 --validation-hours 336 \
  --test-start 2019-11-15 --test-hours 168 \
  --residual-scale 1 \
  --lstm-model data/experiments/lstm_2019/model.pt \
  --gru-model data/experiments/gru_2019/model.pt \
  --prophet-config data/experiments/prophet_tuning_2019/best_config.json \
  --output-dir data/experiments/attention_gru_2019_nov_residual_reproduced
```

O treino usa MSE mascarado e seleciona a época por RMSE de validação. Os
relatórios calculam TIMSE depois da previsão, pela fórmula publicada; a
equivalência com o avaliador completo da plataforma permanece pendente. Veja
[`validation.md`](validation.md). Cada execução registra argumentos, hashes de
fontes e dados, versões, época selecionada, cobertura, erros e latência. Os
resultados ficam em `data/experiments/attention_gru_2019_nov_residual/`.

## Resultado executado

Em cada rodada, os seis modelos com cobertura completa foram comparados nos
mesmos **1.778 pares com alvo observado**. As 28 emissões oferecem 2.016 pares possíveis;
nenhum dos modelos completos falhou. A recorrência tem cobertura de 85,42% e
seu TIMSE deve ser comparado somente nos 1.495 pares comuns aos sete modelos.

| Modelo | RMSE (km/s) | TIMSE ((km/s)³) |
| --- | ---: | ---: |
| GRU com attention, sem atalho | 93,22 | 5.016.157 |
| GRU com attention, `residual_scale=1` | 83,81 | 6.881.059 |
| LSTM | 74,78 | 5.234.528 |
| GRU | 77,79 | 5.748.262 |
| Persistência | 89,67 | 7.344.740 |
| Prophet original | 76,43 | 5.074.253 |
| Prophet ajustado anteriormente | 82,83 | 994.817 |

A versão sem atalho teve TIMSE menor que LSTM/GRU neste recorte, mas RMSE pior.
O atalho melhorou o RMSE da primeira versão, porém piorou seu TIMSE. Nenhuma
das duas superou o Prophet ajustado no TIMSE. A primeira execução foi inspecionada
antes de acrescentar o atalho; **novembro é um diagnóstico exploratório, não um
teste final intocado para escolher entre as versões**. Também se trata de um
único ano, sem evidência de ganho operacional futuro.

Com o atalho, a melhor época foi 13 (RMSE de validação **48,16 km/s**), com
11.169 parâmetros e treino de 28,8 s em CPU com duas threads. A inferência
mediana foi 10,5 ms por emissão, incluindo preparação da entrada; o percentil
95 foi 11,0 ms. O checkpoint recarregado produziu uma previsão de 72 horas
para 21/11/2019 18 UTC.

## Prever a partir do checkpoint

```bash
uv run --locked --extra lstm \
  python -m benchlab_forecasting.attention_gru_cli forecast \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --model data/experiments/attention_gru_2019_nov_residual/model.pt \
  --origin 2019-11-21T18:00:00Z \
  --output-dir data/experiments/attention_gru_forecast_new
```

O modelo ainda usa somente as 168 horas recentes; a recorrência de 27 dias
continua como baseline separado. O checkpoint de 2019 é experimental e não
substitui um modelo validado para inferência operacional no torneio.

Para testar combinações de tamanho, cabeças, atalho e outros parâmetros com
seleção separada do teste, veja a [busca de hiperparâmetros](attention_gru_tuning.md).
