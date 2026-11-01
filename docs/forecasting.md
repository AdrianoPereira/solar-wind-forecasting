# Previsão horária com Prophet

O pacote `benchlab_forecasting` oferece API Python e CLI para experimentos locais
de velocidade do vento solar. Implementa Prophet, persistência e recorrência de
27 dias, com previsões de 1 a 72 horas e backtest walk-forward. Não substitui o
container de submissão nem o avaliador oficial do BenchLab.

## Instalação

Na raiz do projeto:

```bash
uv sync --locked --extra forecast
uv run --locked --extra forecast benchlab-forecast --help
```

As dependências de previsão são opcionais. Para combinar com notebooks, acrescente
`--group notebooks`. O pacote também pode ser instalado com `pip install '.[forecast]'`.

## Previsão

Exemplo com dados que cubram o período anterior a março de 2019:

```bash
uv run --locked --extra forecast benchlab-forecast forecast \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --origin 2019-03-01T00:00:00Z \
  --train-days 180 --horizon 72 \
  --output-dir data/experiments/prophet_20190301
```

Cada experimento exige diretório vazio. Saídas:

- `forecast.csv`: `timestamp_utc` e `speed_km_s`, de `t0+1h` a `t0+horizon`.
- `model.json`: Prophet serializado com configuração, emissão e janela usada.
- `report.json`: argumentos, versão das dependências, SHA-256 dos dados e metadados.

`--model persistence` usa a última velocidade válida disponível.
`--model recurrence` usa a velocidade observada exatamente 27 dias antes de cada
hora prevista. Valores históricos ausentes permanecem ausentes; a CLI de previsão
retorna código 1 se houver previsões faltantes.

Prophet usa tendência constante (`--growth flat`) por padrão. Compare com
`--growth linear`. As sazonalidades diária, semanal e anual estão desativadas.
A sazonalidade solar é uma hipótese opcional: ative com `--rotation-days 27` e
ajuste `--fourier-order 3`. Ela exige pelo menos duas rotações entre o primeiro e o
último ponto válido de treino. Uma sazonalidade fixa de 27 dias não representa
perfeitamente a evolução solar. Não há ganho de precisão pressuposto.

Previsões negativas do Prophet são limitadas a zero. Intervalos probabilísticos
estão desativados; o pacote entrega previsões pontuais. Por padrão é univariado.
Densidade, temperatura e MAG podem entrar como regressores defasados com
`--exogenous plasma` ou `--exogenous plasma_mag`; consulte o
[tuning com exógenas e Fourier](tuning.md). Imagens ainda não entram como
regressores. As defasagens respeitam o horizonte e a latência configurada.

## Avaliação cronológica

```bash
uv run --locked --extra forecast benchlab-forecast backtest \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --start 2019-03-01T00:00:00Z --end 2019-03-02T18:00:00Z \
  --step-hours 6 --train-days 180 --horizon 72 \
  --models prophet persistence recurrence \
  --output-dir data/experiments/backtest_201903
```

Início e fim são emissões inclusivas, não datas dos alvos. O arquivo precisa
cobrir até `end + horizon`. Cada emissão refaz o ajuste usando somente seu passado
disponível. Esse exemplo executa oito ajustes Prophet; comece com poucas emissões
antes de ampliar o período. A janela móvel de treino limita o custo por ajuste.

`predictions.parquet` registra emissão, instante-alvo, horizonte, observação e
previsões de cada modelo. O relatório registra:

- RMSE e MAE sobre todos os pares válidos de cada modelo (`available`).
- Comparação sobre os mesmos pares válidos para todos os modelos (`common`).
- Métricas comuns por hora e por faixas 1–24, 25–48 e 49–72 h.
- Cobertura das previsões, emissões completas, quantidade de alvos válidos e falhas.

Falhas de uma emissão/modelo ficam registradas, com previsões nulas. O processo
continua para permitir diagnóstico e retorna código 1 se houve exceção ou se algum
modelo não tem pares avaliáveis. Lacunas parciais da recorrência são indicadas na
cobertura, sem serem convertidas em persistência silenciosamente. Se a interseção
comum for vazia, suas métricas são nulas; não compare diretamente resultados
`available` com coberturas diferentes.

O relatório também inclui uma aproximação TIMSE provisória e o status da
especificação oficial; consulte o [módulo de validação](validation.md), que permite
reavaliar previsões salvas sem treinar modelos.

As métricas são locais, sem pesos oficiais. Previsões sobrepostas são mantidas
como pares emissão/horizonte, sem supor independência estatística. Reserve um
período final separado da seleção de configurações. O backtest não faz busca de
hiperparâmetros nem estima intervalos de confiança.

## Entrada, qualidade e disponibilidade

Aceita Parquet ou CSV, por padrão nas colunas `timestamp_utc` e `speed_km_s`.
Personalize com `--time-column` e `--target-column`. As unidades esperadas são km/s.
O pacote é voltado a velocidade não negativa, não a séries genéricas com sinal.

- Horários sem fuso são interpretados como UTC; horários com fuso são convertidos.
- Timestamps devem ser únicos e na hora cheia. Não há reamostragem implícita.
- Valores negativos, não finitos, `status != 0` ou `is_valid == false` viram nulos.
- Não há interpolação nem imputação de alvos. Prophet recebe somente pontos válidos.
- Exige no mínimo 48 observações válidas (`--min-points`) e última observação com
  idade de até 24 horas (`--max-age-hours`).

Sem metadados de publicação, considera-se disponibilidade em `timestamp + 60 min`.
`--latency-minutes` altera essa **hipótese**, que deve refletir também a convenção
do timestamp e o fechamento da média horária. Ela não comprova disponibilidade
histórica de produtos revisados. Teste a sensibilidade à latência.

Se houver uma coluna real de publicação, informe `--available-column available_at`.
Ela substitui a latência fixa; linhas sem disponibilidade conhecida são excluídas
do treino. Observação e publicação devem estar em ou antes de `t0`. A coluna deve
corresponder à versão do valor disponível na época, não à data de coleta atual.
Os rótulos de avaliação podem ser posteriores; nunca são usados no ajuste daquela
emissão. O produto ACE exato do ranking ainda precisa ser reconciliado com o arquivo
operacional horário utilizado neste projeto.

## API Python

```python
from benchlab_forecasting import ForecastConfig, ProphetForecaster, load_series, backtest

data = load_series("data/ace_hourly/ace_swepam_hourly.parquet")
config = ForecastConfig(horizon=72, train_days=180, rotation_days=27)
model = ProphetForecaster(config).fit(data, "2019-03-01T00:00:00Z")
forecast = model.predict()
model.save("/tmp/solar_prophet.json")
restored = ProphetForecaster.load("/tmp/solar_prophet.json")
same_forecast = restored.predict()

predictions, report = backtest(
    data, "2019-03-01T00:00:00Z", "2019-03-02T18:00:00Z", config
)
```

O modelo salvo reproduz a previsão de sua emissão original. Para outra emissão,
ajuste novamente com `.fit(data, novo_t0)`. A API espera o DataFrame normalizado
por `load_series`, com `ds` UTC e `y` tratado.

## Testes

```bash
uv run --locked --extra forecast python -m unittest discover -s tests -p 'test_forecasting.py' -v
```

Testes sintéticos exercitam o backend Prophet real, corte de dados futuros,
latência/publicação, serialização, alinhamento da recorrência, alvos ausentes,
comparação comum e CLI. Não dependem de APIs remotas.

Referências consultadas em 03/10/2026:
[API Prophet](https://facebook.github.io/prophet/docs/quick_start.html),
[serialização e tendência constante](https://facebook.github.io/prophet/docs/additional_topics.html),
[sazonalidades e regressores](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html).
