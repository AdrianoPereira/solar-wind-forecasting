# LSTM multivariada para velocidade do vento solar

Implementação em `src/benchlab_forecasting/lstm.py`, com CLI em `lstm_cli.py`.
O modelo recebe 168 horas de histórico e produz diretamente as próximas 72 horas.
São sete variáveis: velocidade, densidade, temperatura, Bx, By, Bz e Bt. Cada
variável tem um indicador de ausência, totalizando 14 canais de entrada.

A arquitetura tem uma LSTM unidirecional de 32 unidades, dropout de 0,1 na saída
e uma camada linear com 72 saídas. Possui 8.520 parâmetros. A previsão é a última
velocidade disponível mais uma correção aprendida para cada horizonte. A camada
de saída começa zerada, correspondendo inicialmente à persistência. O modelo
usa AdamW, clipping do gradiente e perda MSE somente nos alvos observados.

## Experimento executado

Artefatos em `data/experiments/lstm_2019/`:

- Treino: 932 emissões entre 08/01/2019 e 28/08/2019 18 UTC; os últimos alvos
  terminam em 31/08/2019 18 UTC, antes da validação.
- Validação: 56 emissões de 01/09 a 14/09/2019 18 UTC, usada para escolher a época.
- Teste: 28 emissões de 15/12 a 21/12/2019 18 UTC, com alvos até 24/12/2019 18 UTC.
- Horizonte: 72 horas; cadência: seis horas. Nenhuma emissão de treino foi descartada.

O trecho de 1–7 de dezembro, já avaliado durante o tuning do Prophet, não foi
reutilizado como teste final da LSTM. A rede permaneceu com pesos fixos após a
validação; o Prophet foi reajustado em cada emissão, usando configurações fixadas
anteriormente. Essa diferença de atualização faz parte da comparação.

A melhor época foi **4**, com RMSE de validação de **50,36 km/s**. A parada
antecipada encerrou na época 12, após oito épocas sem superar o melhor resultado.
O limite era 60 épocas; não houve busca de arquitetura baseada no teste final.

Comparação nos mesmos **1.881 pares válidos**, sem remover alvos por lacunas da
recorrência (`test/comparison_without_recurrence.json`):

| Modelo | RMSE (km/s) | MAE (km/s) |
|---|---:|---:|
| LSTM | 66,13 | 51,90 |
| Persistência | 88,37 | 70,91 |
| Prophet original | 64,95 | 56,88 |
| Prophet ajustado anteriormente | 90,64 | 82,32 |

A LSTM melhorou em relação à persistência, mas não superou o RMSE do Prophet
original nesse trecho. A comparação com recorrência está separada em
`test/report.json`, usando 1.647 pares comuns a todos os cinco modelos; a cobertura
da recorrência foi 88,19%, contra 100% dos outros modelos. Não compare seus scores
diretamente com a tabela acima, pois as amostras são diferentes.

Na interseção de 1.647 pares, o RMSE foi 63,69 km/s na LSTM, 87,01 na persistência,
56,76 na recorrência, 62,67 no Prophet original e 92,51 no Prophet ajustado. Assim,
a recorrência teve o menor erro entre os modelos nesses pares comuns.

A etapa de otimização, incluindo validação por época, levou aproximadamente
2,54 segundos em CPU com duas threads; isso exclui carregamento e preparação
das janelas. A inferência mediana foi aproximadamente 9,3 ms por emissão, incluindo
preparação da entrada. O pico RSS do processo completo foi cerca de 515 MiB,
incluindo dados, treinamento e comparações com Prophet; não é a memória isolada
da rede. Não houve falhas nas emissões do teste.

Este é um piloto em um único ano e período curto. O teste agora foi avaliado;
novas decisões de arquitetura devem usar validação e reservar outro teste final.

## Instalação e reprodução

PyTorch é opcional: não é importado ao usar apenas os modelos anteriores. O extra
`lstm` usa o índice oficial de wheels CPU e está registrado no `uv.lock`.

```bash
uv sync --locked --inexact --extra forecast --extra lstm

uv run --locked --extra forecast --extra lstm \
  python -m benchlab_forecasting.lstm_cli experiment \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --train-start 2019-01-08 \
  --validation-start 2019-09-01 --validation-hours 336 \
  --test-start 2019-12-15 --test-hours 168 \
  --prophet-config data/experiments/prophet_tuning_2019/best_config.json \
  --output-dir data/experiments/lstm_2019_new
```

O MAG parcial é usado explicitamente porque contém 2019, enquanto o arquivo MAG
completo local cobre 2025. Nenhum download adicional de dados é necessário.
Use sempre um diretório novo. Sem `--prophet-config`, a avaliação compara somente
LSTM, persistência e recorrência, sem precisar instalar o extra `forecast`.

Os argumentos permitem mudar `--lookback-hours`, `--hidden-size`, `--dropout`,
`--epochs`, `--patience`, `--learning-rate`, `--weight-decay`, `--batch-size`,
`--threads` e `--seed`. `--input-group speed` usa apenas velocidade;
`--input-group plasma` acrescenta densidade e temperatura; `plasma_mag` é o padrão.

## Prever com o modelo salvo

```bash
uv run --locked --extra lstm python -m benchlab_forecasting.lstm_cli forecast \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --model data/experiments/lstm_2019/model.pt \
  --origin 2019-12-21T18:00:00Z \
  --output-dir data/experiments/lstm_forecast_new
```

O comando salva `forecast.csv` e `report.json`. A rede é recarregada com pesos e
pré-processamento salvos, sem novo treino. Não mantém estado recorrente entre
emissões. A emissão precisa ser posterior à disponibilidade dos alvos usados
para selecionar o checkpoint. O arquivo de entrada deve cobrir o histórico
necessário; nenhum valor observado depois da emissão é consumido.

```python
from benchlab_forecasting.lstm import LSTMForecaster, load_lstm_data

data = load_lstm_data(
    "data/ace_hourly/ace_swepam_hourly.parquet",
    "data/ace_mag_hourly/ace_mag_hourly_partial.parquet",
)
model = LSTMForecaster.load("data/experiments/lstm_2019/model.pt")
forecast = model.predict(data, "2019-12-21T18:00:00Z")
```

## Disponibilidade e dados ausentes

Cada janela é reconstruída na grade horária até a emissão. Se houver
`available_at` em uma fonte, ele define a disponibilidade histórica; caso contrário,
assume-se `timestamp + latency_minutes`, com padrão de 60 minutos. Não se deve
usar horário de coleta como horário de disponibilidade histórica. Status e
sentinelas são tratados pelos leitores ACE existentes.

Valores não disponíveis na emissão são mascarados antes de preencher lacunas.
O forward-fill é limitado a seis horas e usa apenas o passado. Lacunas restantes
recebem medianas aprendidas nas janelas de treino. Médias e desvios usados na
normalização também são calculados apenas no treino. Os indicadores preservam
a ausência original, inclusive quando houve preenchimento. A última velocidade
realmente disponível deve ter idade de até 24 horas; caso contrário, a emissão
falha explicitamente e fica registrada na avaliação.

Não há imputação de alvos. O cálculo da perda ignora alvos ausentes e o conjunto
de treino exclui alvos publicados depois do corte de validação. As janelas de
treino são purgadas para não atravessar a validação. As sobreposições de previsões
são mantidas como pares emissão/horizonte; não são observações independentes.

Um diagnóstico adicional manteve pesos e normalização fixos e variou somente a
latência das entradas na validação: RMSE de 50,36 / 50,05 / 53,95 km/s para
latências de 1 / 3 / 6 horas. O resultado está em
`latency_sensitivity_validation.json`. Não houve nova seleção nem uso adicional
dos alvos do teste nesse diagnóstico.

## Arquivos e gráficos

- `model.pt`: pesos, configuração, normalização, medianas e histórico do treino.
- `manifest.json`: calendários, parâmetros, versões e hashes de código e entradas.
- `training.json` e `training_history.csv`: seleção da época e curvas de treino.
- `test/predictions.parquet`: previsões e observações alinhadas.
- `test/report.json`: métricas dos cinco modelos, cobertura, erros e custo.
- `test/comparison_without_recurrence.json`: comparação na amostra maior.
- `forecast_last_origin/`: previsão reproduzida pela CLI a partir do checkpoint.

Gráficos em inglês, incluindo curva de aprendizado:

```bash
uv run --locked --group notebooks python -m benchlab_forecasting.plotting \
  --experiment-dir data/experiments/lstm_2019/test \
  --training-history data/experiments/lstm_2019/training_history.csv
```

Os arquivos PNG/PDF ficam em `test/figures/`. A curva de treino agrega perdas
durante as atualizações de cada época; a curva de validação usa os pesos ao final
da época, com dropout desativado.

## Verificação

Os **44 testes** do projeto passaram, incluindo sete testes novos de LSTM para
causalidade, disponibilidade atrasada, purga de alvos, perda mascarada, invariância
a mudanças futuras, checkpoint e CLI. A previsão real do checkpoint recarregado
reproduziu a trajetória da última emissão do teste.

```bash
uv run --locked --extra lstm python -m unittest discover -s tests -p test_lstm.py -v
```

Referências consultadas em 05/10/2026: [LSTM do PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)
e [instalação CPU](https://pytorch.org/get-started/locally/).
