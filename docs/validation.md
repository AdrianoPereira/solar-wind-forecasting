# Módulo independente de validação

`benchlab_forecasting.validation` avalia previsões de qualquer modelo, sem treinar
Prophet. O backtest existente usa esse mesmo módulo para produzir seus relatórios.

## TIMSE: fórmula publicada

A fórmula da imagem fornecida pelo usuário (slide 29) está implementada em
`timse(y, prediction)`. A [FAQ dos participantes](https://docs.google.com/document/d/1nLT8KiXn2hRiyoFS9neTDoJuG75ptVTH5VUYa-D8DRw/edit),
consultada em **05/10/2026**, esclarece que os limiares são as velocidades
observadas únicas, ordenadas, e que não existe ponderação adicional por horizonte.

\[
S_j = \{i : y_i \ge v_j^*\},\qquad
M_j = \frac{1}{|S_j|}\sum_{i\in S_j}(y_i-\hat y_i)^2,
\qquad
\mathrm{TIMSE} = \sum_{j=1}^{m-1}(v_{j+1}^*-v_j^*)M_j.
\]

A integração usa retângulos à esquerda, sem normalização pela amplitude, sem
raiz e sem multiplicador de cobertura. Com velocidades em km/s, a unidade é
**(km/s)³**. Menor é melhor. Empates entram juntos na mesma cauda; o último limiar
não tem intervalo à direita. Ordenação e somas acumuladas custam O(n log n).
`timse_profile()` expõe limiares, tamanho das caudas, MSE, larguras, contribuições
e integral acumulada para auditar o cálculo.

Alvos não finitos são excluídos. Qualquer previsão ausente em um alvo válido torna
o score estrito `timse` nulo. Essa é uma convenção local: o tratamento de previsões
faltantes na plataforma ainda não está confirmado. `timse_common` usa exatamente
os mesmos pares válidos de todos os modelos. Cobertura é informada separadamente.
Quando há somente uma velocidade observada única, a soma vazia vale zero;
`degenerate=True` sinaliza que isso não significa previsão perfeita. Sem alvos
avaliáveis, o resultado é nulo.

A matemática publicada está implementada; a equivalência com o avaliador completo
da plataforma ainda não foi certificada. `official_timse()` permanece reservado
para esse avaliador e lança `NotImplementedError`; use **`timse()`** para calcular
a fórmula. `challenge.official_score` permanece nulo. A FAQ consultada tem SHA-256
`1183984ad86855f38df6dddc9fe5ab00bc7e74b3158daaa6deab21b1993be325`.

`timse_proxy()` foi preservado para ler/reproduzir os experimentos históricos.
Ele calcula a média dos MSEs das caudas em cada observação, incluindo limiares
repetidos, em (km/s)². **É diferente da fórmula publicada** e não deve ser usado
como TIMSE. Relatórios antigos não foram alterados retroativamente.

Veja [validação TIMSE com figuras](timse.md) para o comando, os resultados e o
relatório HTML. Os gráficos, eixos, legendas e tabelas estão em inglês.

## Validar previsões já salvas

```bash
uv run --locked --extra forecast python -m benchlab_forecasting.validation \
  --predictions data/experiments/prophet_backtest_smoke_20190301/predictions.parquet \
  --models prophet persistence recurrence \
  --start 2019-03-01T00:00:00Z --end 2019-03-01T06:00:00Z \
  --step-hours 6 --horizon 72 \
  --output data/experiments/validation_20190301.json
```

O comando aceita CSV/Parquet no formato do backtest: `origin_utc`, `timestamp_utc`,
`horizon`, `y` e uma coluna por modelo. O arquivo de saída deve ser novo. Para uma
previsão isolada, acrescente emissão, horizonte e observações ao `forecast.csv`
antes de avaliar; essas informações não podem ser inferidas do valor previsto.

Forneça `--start` e `--end` para contabilizar também emissões completamente ausentes
do arquivo. Sem essas opções, o calendário considera apenas emissões presentes.
Linhas faltantes são reconstruídas como previsões ausentes. Um alvo conhecido em
outra emissão é reutilizado no mesmo timestamp; não se interpolam observações.
Se uma emissão inteira e seus alvos estão ausentes, ela reduz cobertura, mas não
gera erros numéricos inventados. O comando retorna 1 se não há pares comuns
avaliáveis; cobertura parcial continua explicitamente registrada.

O relatório contém MSE, RMSE, MAE e viés médio (`previsão - observação`), métricas
por hora, faixas 1–24/25–48/49–72 h e mês do alvo. Resultados `available` usam os
pares disponíveis de cada modelo; `common` permite comparação nos mesmos pares.
Sobreposições são mantidas como pares emissão/horizonte, uma convenção local.

`complete_origin_fraction` mede a proporção de trajetórias com valores finitos;
não comprova entrega no prazo, aceitação da plataforma ou uptime oficial. O módulo
valida a consistência temporal, duplicatas, velocidades negativas/infinitas e alvos
conflitantes. Ele não substitui o validador de submissão da plataforma.

```python
import pandas as pd
from benchlab_forecasting.validation import evaluate_predictions

predictions = pd.read_parquet("data/experiments/prophet_backtest_smoke_20190301/predictions.parquet")
report = evaluate_predictions(predictions, ["prophet", "persistence", "recurrence"])
print(report["metrics"]["prophet"]["common"])
print(report["metrics"]["prophet"]["timse_common"])
```

## Diferentes divisões temporais

`temporal_splits` gera índices posicionais para amostras supervisionadas indexadas
por **emissão**. Não recebe linhas individuais de previsões empilhadas. Estratégias:

- `holdout`: um único bloco futuro reservado.
- `expanding`: blocos futuros sucessivos, com treino crescente desde o início.
- `rolling`: blocos futuros sucessivos, com janela limitada de emissões de treino.

```python
import pandas as pd
from benchlab_forecasting.validation import temporal_splits

origins = pd.date_range("2019-01-01", "2019-12-28", freq="6h", tz="UTC")
folds = temporal_splits(
    origins, "2019-07-01", strategy="rolling", n_splits=3,
    train_hours=90 * 24, validation_hours=7 * 24,
    step_hours=7 * 24, horizon=72, gap_hours=0,
)
for fold in folds:
    train_origins = origins[fold.train_indices]
    validation_origins = origins[fold.validation_indices]
    # Ajuste modelo, normalização e seleção de variáveis exclusivamente no treino.
    # Emita previsões para validation_origins respeitando disponibilidade de dados.
```

Blocos de validação usam `[início, fim)`. Amostras de treino são admitidas somente
quando `emissão + horizonte + gap < início_da_validação`, removendo alvos que
invadiriam a validação. O gap adicional é opcional; a purga do horizonte é sempre
aplicada. `train_hours` conta para trás desde o início da validação, antes da purga.
Os blocos de emissões não se sobrepõem, embora seus alvos de 72 h possam se sobrepor.
Não trate folds ou previsões sobrepostas como amostras independentes ao estimar
incerteza. Os índices não executam treinamento automaticamente; o backtest Prophet
continua sendo walk-forward com ajuste por emissão sobre observações disponíveis.

Reserve um holdout final fora da seleção de configurações. Nenhuma divisão
aleatória é oferecida, pois ela quebraria a ordem temporal deste problema.

## Testes

```bash
uv run --locked --extra forecast python -m unittest discover -s tests -p 'test_validation.py' -v
```

Incluem cálculos manuais das métricas, empates, ênfase nos erros de velocidades
altas, lacunas, calendários incompletos, alinhamento, comparação comum e purga de
alvos nos três tipos de divisão.

Testes da fórmula TIMSE e geração dos artefatos: `python -m unittest discover -s tests -p "test_timse.py" -v`.
