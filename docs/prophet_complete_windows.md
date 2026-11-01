# Prophet nas janelas ACE completas

Três estratégias Prophet usam, respectivamente, 3, 7 ou 14 dias de histórico
sem lacunas para prever a velocidade do vento solar nas 72 horas seguintes.
O script lê `data/training_windows/{3d,7d,14d}_72h/` e ajusta um Prophet
separado em cada origem de previsão. Não concatena janelas sobrepostas nem
usa seus alvos futuros no ajuste daquela origem.

```bash
uv run --locked --extra forecast python scripts/train_prophet_complete_windows.py
```

Cada pasta em `data/models/prophet_complete_windows/` contém:

- `model.json`: Prophet ajustado na última janela disponível daquela duração;
- `latest_forecast.parquet`: previsão de 72 horas nessa origem e observação
  histórica correspondente;
- `validation_flat.parquet` e `validation_linear.parquet`: previsões de seleção;
- `test.parquet`: previsões e persistência no teste posterior;
- `manifest.json`: configuração, datas, versão, hash dos dados e métricas.

O modelo salvo pertence à origem registrada no manifesto; para outra origem,
o Prophet deve ser ajustado novamente com a janela de entrada correspondente.
O script aceita `--validation-origins`, `--test-origins`, `--seed`, `--data-dir`,
`--source` e `--output-dir`.

Somente a velocidade SWEPAM (`inputs[:, :, 0]`) entra no Prophet. Os outros oito
campos têm histórico na janela, mas seus valores nas 72 horas futuras não são
conhecidos em `t0`; incluí-los como regressores contemporâneos vazaria o futuro.
As entradas vão exatamente de `t0-L` a `t0-1 h`. O Prophet prevê `t0+1` a
`t0+72 h`, sem usar dados interpolados. O modelo limita velocidades previstas
negativas a zero.

Foram comparadas duas configurações simples, com tendência `flat` ou `linear`.
Ambas desativam sazonalidades diária, semanal e anual, mudanças de tendência
(`n_changepoints=0`) e amostragem de incerteza. A escolha usa RMSE na validação.
As mesmas 24 origens elegíveis nas três durações foram escolhidas sem sobrepor
seus alvos: 12 para validação cronológica (2003–2008) e 12 posteriores para
teste (2009–2010). A persistência, que repete a última velocidade observada,
é avaliada exatamente nas mesmas horas do teste.

| Histórico | Tendência escolhida | RMSE Prophet (km/s) | RMSE persistência (km/s) | RMSE Prophet na hora 72 (km/s) |
| --- | --- | ---: | ---: | ---: |
| 3 dias | flat | 79,24 | 63,74 | 93,03 |
| 7 dias | flat | 73,11 | 63,74 | 82,65 |
| 14 dias | linear | 89,85 | 63,74 | 118,18 |

Cada RMSE global usa 12 × 72 = 864 pares observação–previsão. Neste teste,
**nenhuma das três estratégias Prophet superou a persistência**. As janelas
completas terminam em 2011 (e as de 14 dias em 2010); portanto esta avaliação
pequena e antiga não demonstra desempenho no período recente. Para ampliar a
avaliação após 2011, é preciso primeiro construir exemplos com uma política
explícita para entradas e alvos ausentes ou validar um produto ACE alternativo.

As figuras do teste ficam em `data/models/prophet_complete_windows/figures/`:
[comparação de RMSE](../data/models/prophet_complete_windows/figures/comparacao_rmse_teste.png),
[RMSE por horizonte](../data/models/prophet_complete_windows/figures/rmse_por_horizonte.png)
e [exemplo de previsão de 72 horas](../data/models/prophet_complete_windows/figures/exemplo_previsoes_72h.png).
Elas usam somente os `test.parquet` das três durações e podem ser regeneradas com:

```bash
uv run --locked --extra forecast python scripts/plot_prophet_complete_windows.py
```

O modelo JSON usa a [serialização oficial do Prophet](https://facebook.github.io/prophet/docs/additional_topics.html).
As [instruções oficiais para dados horários](https://facebook.github.io/prophet/docs/non-daily_data.html)
embasam a grade `ds` horária usada no ajuste e na previsão.
