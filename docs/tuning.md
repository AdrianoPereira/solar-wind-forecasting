# Tuning do Prophet com exógenas e Fourier

O pacote aceita densidade e temperatura ACE/SWEPAM (`plasma`) e também Bx, By,
Bz e Bt ACE/MAG (`plasma_mag`). Os coeficientes são ajustados junto da tendência
e sazonalidade. As imagens AIA não entram nesta implementação.

## Resultado executado em 05/10/2026

A busca em `data/experiments/prophet_tuning_2019/` avaliou 20 configurações em
12 emissões de validação: quatro por dia em 01/06, 01/08 e 01/10/2019. A seleção
usou somente essas previsões. O teste final teve 28 emissões, de 01/12/2019 00 UTC
a 07/12/2019 18 UTC, com alvos até 10/12/2019 18 UTC. Horizonte de 72 horas e
cadência de seis horas. Nenhum dos 240 ajustes de validação falhou.

A configuração selecionada foi `plasma_fourier_lag_648`: densidade e temperatura
com lag de 648 horas, período de 27 dias, Fourier de ordem 3, priors de sazonalidade
e regressores iguais a 1, tendência constante e janela de treino de 180 dias.
MAG foi testado, mas não entrou na configuração vencedora.

Na validação, o RMSE foi 66,88 km/s no Prophet original, 60,79 km/s com Fourier
de ordem 3 sem exógenas e 60,02 km/s na configuração selecionada. Portanto, o
ganho incremental das exógenas sobre Fourier foi pequeno nesse recorte. Os
240 ajustes de seleção somaram aproximadamente 35,6 segundos de execução.

Comparação no teste final, nos mesmos **1.857 pares válidos** de atual, ajustado e
persistência (`test/comparison_without_recurrence.json`):

| Modelo | RMSE (km/s) | MAE (km/s) |
|---|---:|---:|
| Prophet original | 71,12 | 66,10 |
| Prophet ajustado | 49,47 | 39,88 |
| Persistência | 35,32 | 27,29 |

O RMSE do Prophet caiu aproximadamente **30,45%**, mas ainda ficou acima da
persistência. A cobertura de previsões desses três modelos foi 100%. Incluindo a
recorrência, a interseção tem 1.691 pares e os RMSEs são 71,31 / 49,78 / 35,89 /
57,15 km/s, respectivamente (atual / ajustado / persistência / recorrência).
As lacunas da recorrência explicam a mudança de amostra.

Na hora 72, na comparação sem recorrência, o RMSE foi 63,78 km/s no original,
45,93 km/s no ajustado e 47,90 km/s na persistência. Isso não altera a conclusão
de que a persistência foi melhor no agregado das 72 horas.

A sensibilidade à latência foi avaliada somente na validação, mantendo fixa a
configuração selecionada, sem nova seleção: RMSE de 60,02 / 60,23 / 60,53 km/s
para latências assumidas de 1 / 3 / 6 horas. O relatório está em
`latency_sensitivity_validation.json`.

Trata-se de uma busca inicial em três blocos curtos de um único ano, não de uma
garantia de melhora em outros regimes ou no score oficial. O teste final já foi
usado para esta avaliação; uma nova seleção requer outro período final reservado.

## Causalidade das entradas

Para o alvo `t`, cada regressor usa a observação em `t - lag`. O lag precisa cobrir
o horizonte mais a latência assumida: com 72 horas e uma hora de latência, o mínimo
é 73 horas. A busca usa 96 ou 648 horas. Não se usam as exógenas realmente observadas
nas próximas 72 horas. Defasagens longas podem perder informação recente.

O preenchimento usa apenas observações anteriores, por no máximo seis horas.
Lacunas restantes recebem a mediana do treino, com indicadores de ausência quando
variam no treino. Colunas constantes são removidas; variáveis completamente
ausentes no treino fazem o candidato falhar. O Prophet ajusta a padronização apenas
no treino de cada emissão. Alvos ausentes não são preenchidos.

Se a fonte tiver `available_at`, ele deve indicar disponibilidade histórica do
valor, não data de coleta. Sem ele, assume-se latência configurável de 60 minutos.
Além do corte na emissão, cada valor deve estar disponível até
`source_time + lag - horizon`; essa regra conservadora também vale no treino.
Os arquivos históricos não demonstram a latência operacional.

## Reproduzir a busca

```bash
uv run --locked --extra forecast python -m benchlab_forecasting.tuning \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --validation-starts 2019-06-01 2019-08-01 2019-10-01 \
  --validation-hours 24 --step-hours 6 \
  --test-start 2019-12-01 --test-hours 168 \
  --output-dir data/experiments/prophet_tuning_2019_new
```

Use um diretório novo por execução. São 240 ajustes de seleção e 56 no teste,
além dos baselines. O plasma completo local cobre 2019; o MAG completo cobre 2025.
Por isso o comando usa explicitamente o MAG parcial, cujo trecho de 2019 contém
8.760 registros e aproximadamente 92,1% de componentes magnéticas válidas.

A busca inclui o modelo original, Fourier sem exógenas, exógenas sem Fourier e
combinações. Testa ordens 1/3/5/8, períodos de 26/27/28 dias, priors de sazonalidade
0,1/1, priors dos regressores 0,1/1, tendência constante ou linear, changepoint
priors 0,01/0,05 e janelas de 90/180 dias. É uma lista limitada de configurações,
não o produto cartesiano de todos os parâmetros.

A ordem K representa K pares seno/cosseno, ou 2K termos de Fourier. Ordem 3 para
27 dias inclui harmônicas de 27, 13,5 e 9 dias. Não se acrescenta uma sazonalidade
independente de 13,5 dias para evitar duplicar termos já incluídos.

O critério local é menor RMSE nos mesmos pares de validação. Candidatos com qualquer
previsão faltante são inelegíveis; falhas e cobertura ficam registradas. A seleção
é salva antes de avaliar o teste. Durante o teste walk-forward, observações do
início do teste podem entrar em refits posteriores quando disponíveis, mantendo
fixa a configuração. Os intervalos de alvos não podem se sobrepor entre blocos
de validação ou entre validação e teste. Dentro dos blocos, as sobreposições são
mantidas, sem supor independência estatística ou estimar intervalos de confiança.

Para uma busca própria, informe `--candidates arquivo.json`, por exemplo:

```json
[
  {"name": "prophet_current", "group": "none", "config": {}},
  {"name": "custom_exogenous", "group": "plasma_mag", "config": {
    "rotation_days": 27, "fourier_order": 3,
    "seasonality_prior_scale": 0.1, "regressor_prior_scale": 0.1,
    "exogenous_lag_hours": 96, "train_days": 180
  }}
]
```

Inclua `prophet_current` para preservar a referência original. Se ausente, o
primeiro candidato é a referência no teste. Todos devem compartilhar o horizonte.

## Artefatos

- `manifest.json`: configurações, calendário, versões e hashes dos dados e código.
- `diagnostics.json`: falhas, cobertura e tempo por ajuste de validação.
- `leaderboard.csv`: ranking agregado e RMSE por bloco de validação.
- `validation_predictions.parquet` e `validation_report.json`: previsões e métricas.
- `best_config.json`: configuração escolhida exclusivamente na validação.
- `latency_sensitivity_validation.json`: diagnóstico adicional com configuração
  fixa e latências de 1/3/6 horas, sem reutilizar o teste final.
- `test/predictions.parquet` e `test/report.json`: comparação dos quatro modelos,
  métricas por horizonte, cobertura e falhas.
- `test/comparison_without_recurrence.json`: comparação sem remover alvos por
  lacunas da recorrência, usando atual, ajustado e persistência.
- `test/model_last_origin.json`: modelo ajustado na última emissão, com as features
  futuras necessárias para reproduzir aquela previsão.

Os nomes das exógenas seguem o esquema ACE nativo. Modelos antigos sem regressores
continuam carregáveis; alterar a emissão exige novo ajuste.

Validação do código: 25 testes de previsão, validação e tuning passaram,
incluindo alinhamento dos lags, disponibilidade atrasada, limite de preenchimento,
imputação só no treino, invariância a alterações de dados futuros, serialização
com Prophet real e separação entre seleção e teste. Uma execução adicional pela
CLI reproduziu a trajetória selecionada da última emissão do teste, e um modelo
salvo no esquema anterior foi carregado com sucesso.

## Usar a configuração selecionada

```bash
uv run --locked --extra forecast benchlab-forecast forecast \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --exogenous plasma --exogenous-lag-hours 648 \
  --rotation-days 27 --fourier-order 3 \
  --seasonality-prior-scale 1 --regressor-prior-scale 1 \
  --train-days 180 --growth flat --origin 2019-12-15T00:00:00Z \
  --output-dir data/experiments/prophet_tuned_forecast
```

Ou carregue a configuração para uso pela API:

```python
import json
from benchlab_forecasting import ForecastConfig, ProphetForecaster, load_series
from benchlab_forecasting.features import load_regressors

selected = json.load(open("data/experiments/prophet_tuning_2019/best_config.json"))
series = load_series("data/ace_hourly/ace_swepam_hourly.parquet")
regressors = None if selected["group"] == "none" else load_regressors(
    "data/ace_hourly/ace_swepam_hourly.parquet",
    "data/ace_mag_hourly/ace_mag_hourly_partial.parquet",
    selected["group"],
)
model = ProphetForecaster(ForecastConfig(**selected["config"]))
forecast = model.fit(series, "2019-12-15", regressors=regressors).predict()
```

Para gerar os gráficos em inglês:

```bash
uv run --locked --extra forecast --group notebooks \
  python -m benchlab_forecasting.plotting \
  --experiment-dir data/experiments/prophet_tuning_2019/test
```

Referências consultadas em 05/10/2026: [regressores, Fourier e regularização do
Prophet](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html)
e [validação temporal e tuning](https://facebook.github.io/prophet/docs/diagnostics.html).
