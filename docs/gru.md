# GRU multivariada para previsão de 72 horas

`src/benchlab_forecasting/gru.py` implementa `GRUConfig`, `DirectGRU` e
`GRUForecaster`. A arquitetura usa uma GRU unidirecional de 32 unidades, dropout
de 0,1 e uma camada linear de 72 saídas. São **6.984 parâmetros**, contra 8.520
da LSTM com a mesma dimensão oculta; menos parâmetros não garante menor tempo
de execução no backend CPU.

As entradas são 168 horas de velocidade, densidade, temperatura, Bx, By, Bz e Bt,
com sete indicadores de ausência (14 canais). A rede aprende correções à última
velocidade disponível, começando na persistência. O loop de treino, perda MSE
mascarada, normalização, disponibilidade e pré-processamento são compartilhados
com a [LSTM](lstm.md). A implementação usa efetivamente `torch.nn.GRU`.

Não se consomem exógenas futuras. O forward-fill é limitado a seis horas e as
lacunas restantes recebem medianas calculadas no treino. Normalização também usa
somente treino. Alvos faltantes são excluídos da perda. A latência padrão é uma
hipótese de 60 minutos, substituída por `available_at` quando a fonte o fornece.

## Resultado executado em 05/10/2026

A validação selecionou a época **3**, com RMSE de **50,09 km/s**. O treino parou
na época 11, após oito épocas sem melhora. O modelo foi salvo antes de avaliar
as emissões de dezembro; não houve tuning posterior com esses resultados.

Nos mesmos **1.647 pares válidos** para todos os modelos:

| Modelo | RMSE (km/s) | MAE (km/s) |
|---|---:|---:|
| GRU | 67,08 | 51,75 |
| LSTM | 63,69 | 49,93 |
| Persistência | 87,01 | 69,70 |
| Recorrência de 27 dias | 56,76 | 48,42 |
| Prophet original | 62,67 | 54,51 |
| Prophet ajustado anteriormente | 92,51 | 84,75 |

A GRU foi melhor que a persistência, mas não superou LSTM, Prophet original ou
recorrência nesse recorte. Não há evidência de que deva substituir esses modelos.
A cobertura das previsões GRU foi 100%, sem falhas; a recorrência teve 88,19%.

Sem restringir pelos valores faltantes da recorrência, a interseção dos outros
cinco modelos contém **1.881 pares**. Nela, o RMSE foi 69,63 km/s na GRU, 66,13
na LSTM, 88,37 na persistência, 64,95 no Prophet original e 90,64 no ajustado.

A otimização com validação por época levou aproximadamente 8,27 segundos, sem
contar leitura e preparação das janelas. A mediana da inferência GRU, incluindo
preparação da entrada, foi 10,2 ms e o percentil 95 foi 12,0 ms. O pico RSS do
processo completo foi cerca de 433 MiB; não é a memória isolada da GRU.

## Treinar e comparar

A GRU reutiliza as dependências do extra `lstm`, incluindo PyTorch CPU; não exige
outra instalação quando a LSTM já funciona.

```bash
uv run --locked --extra lstm --extra forecast \
  python -m benchlab_forecasting.gru_cli experiment \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --train-start 2019-01-08 \
  --validation-start 2019-09-01 --validation-hours 336 \
  --test-start 2019-12-15 --test-hours 168 \
  --lstm-model data/experiments/lstm_2019/model.pt \
  --prophet-config data/experiments/prophet_tuning_2019/best_config.json \
  --output-dir data/experiments/gru_2019_new
```

Use um diretório novo. O experimento executado fica em
`data/experiments/gru_2019/`. O MAG parcial é usado explicitamente porque contém
o ano de 2019, enquanto o arquivo completo local cobre 2025.

O calendário é o mesmo da LSTM: 932 emissões de treino de janeiro a agosto,
56 de validação em setembro e 28 de comparação de 15–21/12/2019, com alvos até
24/12/2019 18 UTC. Emissões ocorrem a cada seis horas. Os alvos do treino
terminam antes da validação. A época é escolhida pela validação, com limite de
60 épocas e paciência de oito épocas sem melhora.

**Dezembro já foi avaliado em experimentos anteriores.** Esta execução é uma
comparação descritiva no mesmo período, não um novo teste final intocado. Não se
ajustam hiperparâmetros usando dezembro, mas a avaliação definitiva da escolha
entre arquiteturas exige outro período reservado.

`--lstm-model` carrega os pesos já selecionados da LSTM, sem treiná-la novamente.
O Prophet é reajustado em cada emissão com configuração previamente escolhida.
Os pesos recorrentes ficam fixos depois da validação. Os modelos são comparados
nos mesmos pares válidos; sobreposições são mantidas, sem assumir independência.

Sem `--lstm-model`, a LSTM não entra na comparação. Sem `--prophet-config`, não se
executa Prophet nem se precisa do extra `forecast`. Os argumentos de configuração
são os mesmos da LSTM, incluindo `--input-group speed|plasma|plasma_mag`,
`--hidden-size`, `--lookback-hours`, `--epochs`, `--learning-rate` e `--threads`.

## Prever com o checkpoint

```bash
uv run --locked --extra lstm python -m benchlab_forecasting.gru_cli forecast \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --model data/experiments/gru_2019/model.pt \
  --origin 2019-12-21T18:00:00Z \
  --output-dir data/experiments/gru_forecast_new
```

Saídas: `forecast.csv` e `report.json`. Cada emissão reconstrói seu histórico,
sem depender de estado recorrente de emissões anteriores. A inferência recusa
emissões anteriores à disponibilidade do checkpoint selecionado e entradas cuja
última velocidade disponível tenha mais de 24 horas.

```python
from benchlab_forecasting.gru import GRUForecaster, load_gru_data

data = load_gru_data(
    "data/ace_hourly/ace_swepam_hourly.parquet",
    "data/ace_mag_hourly/ace_mag_hourly_partial.parquet",
)
model = GRUForecaster.load("data/experiments/gru_2019/model.pt")
forecast = model.predict(data, "2019-12-21T18:00:00Z")
```

O checkpoint contém arquitetura, pesos, normalização, medianas e configuração.
Carregar um checkpoint GRU como LSTM, ou o inverso, gera um erro explícito.
Checkpoints antigos sem o campo de arquitetura continuam válidos como LSTM.

## Artefatos e gráficos

- `model.pt`: modelo selecionado antes da comparação em dezembro.
- `manifest.json`: configuração, calendário, versões e hashes de código e entradas.
- `training.json` e `training_history.csv`: época selecionada e curvas de treino.
- `test/predictions.parquet`: observações e previsões alinhadas dos modelos.
- `test/report.json`: métricas comuns, por horizonte, cobertura, falhas e custo.
- `test/comparison_without_recurrence.json`: comparação sem remover alvos por
  lacunas históricas da recorrência; sua amostra pode diferir do relatório principal.

```bash
uv run --locked --group notebooks python -m benchlab_forecasting.plotting \
  --experiment-dir data/experiments/gru_2019/test \
  --training-history data/experiments/gru_2019/training_history.csv \
  --training-label GRU
```

As figuras são em inglês e ficam em `test/figures/`, nos formatos PNG e PDF.

```bash
uv run --locked --extra lstm python -m unittest discover -s tests -p test_gru.py -v
```

Os testes verificam o encoder GRU real, inicialização na persistência, invariância
a dados futuros, serialização, distinção das arquiteturas, compatibilidade com
checkpoints LSTM antigos e execução da CLI com comparação à LSTM.

A suíte completa executada após a integração passou nos **48 testes**.
O checkpoint recarregado reproduziu a última trajetória da GRU; as previsões da
LSTM antiga permaneceram idênticas às salvas no experimento anterior.

A sensibilidade à latência foi medida somente na validação, mantendo pesos e
normalização fixos: RMSE de 50,09 / 49,38 / 53,42 km/s para latências de 1 / 3 /
6 horas. O diagnóstico está em `latency_sensitivity_validation.json`; não houve
nova seleção nem ajuste usando os alvos de dezembro.

Referência técnica: [GRU do PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html),
consultada em 05/10/2026.
