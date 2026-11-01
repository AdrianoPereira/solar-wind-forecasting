# Gráficos dos resultados

O módulo lê os resultados já salvos, sem treinar modelos. Todos os títulos,
eixos e legendas são em inglês. Funciona sem interface gráfica e salva PNG e
PDF por padrão; também aceita SVG. Matplotlib já pertence ao grupo `notebooks`.

Previsão individual do Prophet, comparada às observações disponíveis:

```bash
uv run --locked --extra forecast --group notebooks \
  python -m benchlab_forecasting.plotting \
  --experiment-dir data/experiments/prophet
```

O arquivo `report.json` fornece a emissão, nome do modelo e caminho dos dados
observados. Use `--observations data/ace_hourly/ace_swepam_hourly.parquet` se os
dados mudaram de lugar. Flags inválidas e lacunas permanecem ausentes, sem
interpolação. Sem arquivo de observações disponível, somente a previsão é traçada.

Para os resultados do backtest:

```bash
uv run --locked --extra forecast --group notebooks \
  python -m benchlab_forecasting.plotting \
  --experiment-dir data/experiments/prophet_backtest_smoke_20190301
```

As figuras são salvas na subpasta `figures/` do experimento:

- `forecast_trajectory`: observações e previsões de uma emissão; por padrão, a última.
- `model_comparison`: RMSE, MAE e viés (previsão menos observação).
- `error_by_horizon`: RMSE por hora e por faixas de horizonte.
- `forecast_coverage`: cobertura de valores e proporção de emissões completas.
- `observed_vs_predicted`: densidade de pares observados e previstos e linha de igualdade.
- `error_distribution`: distribuição dos erros.

A previsão individual gera apenas `forecast_trajectory`. Para selecionar outra
emissão do backtest ou mudar a exportação, acrescente:

```bash
--origin 2019-03-01T00:00:00Z --formats png svg --dpi 240 \
  --output-dir data/experiments/figures_custom
```

Para experimentos LSTM, `--training-history caminho/training_history.csv` também
gera `learning_curve`, com curvas de treino/validação e época selecionada.
Para GRU, acrescente `--training-label GRU`.

Executar novamente atualiza as figuras com os mesmos nomes. Os arquivos de
previsões e relatórios de entrada não são alterados.

As métricas são recalculadas pelo módulo de validação existente, usando apenas
os pares emissão/horizonte válidos **comuns a todos os modelos**. Sobreposições
são mantidas; os gráficos não representam amostras estatisticamente independentes
nem estimam incerteza. Sem pares comuns, os painéis de erro indicam ausência de
dados ou ficam sem curva, preservando o gráfico de cobertura.

O calendário completo usa `start`, `end` e `step_hours` dos argumentos do relatório,
quando presentes. Sem esses metadados, a cobertura considera somente as emissões
presentes no arquivo. Cobertura mede completude das previsões salvas, não uptime
da plataforma. Os gráficos apresentam diagnósticos locais, sem score oficial.
