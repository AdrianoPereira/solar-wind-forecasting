# Validação TIMSE com figuras

A fórmula do slide fornecido está implementada em
[`validation/metrics.py`](../src/benchlab_forecasting/validation/metrics.py).
O gerador [`timse_report.py`](../src/benchlab_forecasting/validation/timse_report.py)
avalia previsões salvas de qualquer modelo e produz figuras e relatório em inglês.

Abra o [relatório HTML completo](../data/experiments/timse_validation_2019_final/report.html).
Ele incorpora as imagens e pode ser compartilhado como um único arquivo.

## Fórmula e interpretação

Para os limiares **únicos e ordenados** das velocidades observadas:

\[
S_j=\{i:y_i\ge v_j^*\},\quad
M_j=\frac{1}{|S_j|}\sum_{i\in S_j}(y_i-\hat y_i)^2,\quad
\mathrm{TIMSE}=\sum_{j=1}^{m-1}(v_{j+1}^*-v_j^*)M_j.
\]

Integração por retângulos à esquerda, sem dividir pela amplitude, sem extrair raiz
e sem ponderação adicional por horizonte. A unidade é **(km/s)³** e menor é melhor.
A função `timse_proxy` dos relatórios antigos é outra métrica, mantida somente para
reprodutibilidade; nenhum dos gráficos novos usa essa aproximação.

```python
from benchlab_forecasting.validation import timse, timse_profile

# MSEs das caudas: 7.5, 12.5, 16; larguras: 2, 3, 0.
assert timse([1, 1, 3, 6], [2, 3, 6, 10]) == 52.5
profile = timse_profile(observed, predicted)
# profile: threshold, tail_n, tail_mse, width, contribution, cumulative_timse.
```

Empates entram juntos nas caudas. O último limiar não tem intervalo à direita.
Com apenas um limiar, a fórmula resulta em zero e o relatório marca `degenerate`;
isso não evidencia previsão perfeita. Sem pares avaliáveis, o score é indefinido.

## Reproduzir

No ambiente existente:

```bash
MPLCONFIGDIR=/tmp/benchlab-matplotlib .venv/bin/python \
  -m benchlab_forecasting.validation.timse_report \
  --predictions data/experiments/gru_2019/test/predictions.parquet \
  --models gru lstm persistence recurrence prophet_current prophet_tuned \
  --start 2019-12-15T00:00:00Z --end 2019-12-21T18:00:00Z \
  --step-hours 6 --horizon 72 \
  --output-dir data/experiments/timse_validation_2019_reproduced \
  --formats png pdf
```

A saída precisa ser um diretório novo ou vazio. Aceita CSV e Parquet com
`origin_utc`, `timestamp_utc`, `horizon`, `y` e uma coluna por modelo. Para instalar
o ambiente em outra máquina: `uv sync --locked --extra forecast --group notebooks`.
Nenhum modelo é retreinado por esse comando.

O calendário explícito contabiliza emissões completamente ausentes. Sem `--start`
e `--end`, só é possível reconstruir as emissões presentes no arquivo. Alvos
faltantes não são interpolados; observações conhecidas no mesmo timestamp em outra
emissão são reutilizadas. O relatório conserva as sobreposições entre previsões
como pares emissão/horizonte e registra essa convenção local.

## Resultados executados

Emissões de **15–21/12/2019**, 28 origens a cada 6 horas, horizonte de 72 horas.
São 2.016 pares esperados e 1.881 pares com alvo observado. A janela já foi explorada
nos experimentos anteriores; esta avaliação é retrospectiva, sem nova seleção de
configurações e sem constituir um novo holdout independente.
As velocidades observadas vão de 319,9 a 534,0 km/s; este recorte não valida o
comportamento em eventos acima dessa faixa. Os pares comuns dos seis modelos
correspondem a 189 timestamps de alvo distintos, e os dos cinco completos a 214.

Comparação dos seis modelos nos mesmos **1.647 pares**:

| Modelo | TIMSE (km/s)³ | Cobertura de previsões |
|---|---:|---:|
| Recorrência de 27 dias | 607.624,80 | 88,19% |
| Prophet ajustado | 905.833,46 | 100% |
| Prophet atual | 1.664.874,23 | 100% |
| LSTM | 1.764.448,12 | 100% |
| GRU | 2.022.667,11 | 100% |
| Persistência | 2.687.954,04 | 100% |

Comparação separada dos modelos com cobertura completa, nos **1.881 pares**:

| Modelo | TIMSE (km/s)³ | RMSE (km/s) |
|---|---:|---:|
| Prophet ajustado | 846.850,01 | 90,64 |
| Prophet atual | 1.705.457,09 | 64,95 |
| LSTM | 1.839.864,28 | 66,13 |
| GRU | 2.106.172,20 | 69,63 |
| Persistência | 2.758.719,96 | 88,37 |

O Prophet ajustado apresenta o menor TIMSE entre os modelos completos nesta janela,
apesar do RMSE maior. A curva de MSE das caudas mostra erros menores nas velocidades
mais altas desse conjunto. A ordenação pelo TIMSE pode, portanto, diferir da
ordenação pelo RMSE. Isso não garante desempenho superior no período do torneio.

A recorrência não pode ser declarada vencedora operacional com base nesse recorte:
a comparação comum exclui seus alvos sem previsão. Seu score estrito permanece
indefinido. Os valores de grupos diferentes não devem ser comparados diretamente,
pois os alvos e seus pesos mudam. O relatório informa também cobertura entre pares
com alvo observado, distinta da cobertura de toda a grade de previsões.

## Figuras e tabelas

Para cada grupo de modelos, foram geradas cinco figuras, em PNG e PDF:

1. **Ranking and coverage**: TIMSE nos pares comuns e cobertura na grade completa.
2. **Tail MSE**: erros acima de cada limiar e quantidade de pares na cauda.
3. **Threshold contributions**: integral acumulada e contribuições agrupadas apenas para visualização.
4. **By horizon**: TIMSE por antecedência e número de pares comuns.
5. **By origin**: TIMSE por emissão e número de pares comuns.

Os limiares são recalculados dentro de cada horizonte, emissão ou faixa. Os scores
desses subconjuntos **não somam** o score global; a faixa de velocidades também
pode mudar. Os CSVs registram limites observados, contagens e casos degenerados.

Arquivos na pasta do relatório:

- `report.html`: figuras e tabelas em inglês, sem dependências de imagens externas.
- `report.json`: fórmula, fontes, calendário, contagens, versões, hashes do código e dados.
- `*_scores.csv`: TIMSE comum, TIMSE estrito, RMSE, cobertura e emissões completas.
- `*_thresholds.csv`: cada limiar, tamanho de cauda, MSE, largura e contribuição.
- `*_groups.csv`: resultados por hora, emissão e faixas 1–24/25–48/49–72 h.
- `*_common_pairs.parquet`: pares exatos utilizados, permitindo auditoria.

`all_models` inclui os seis modelos; `complete_models` inclui apenas modelos com
previsões finitas em toda a grade esperada. Essa segunda comparação é criada
automaticamente quando existem modelos completos e incompletos.

## Escopo e verificação

A imagem fornecida define a matemática, e a
[FAQ dos participantes](https://docs.google.com/document/d/1nLT8KiXn2hRiyoFS9neTDoJuG75ptVTH5VUYa-D8DRw/edit)
consultada em 05/10/2026 confirma os limiares únicos e a ausência de pesos adicionais
por horizonte. O avaliador de referência completo e o tratamento de previsões
faltantes na plataforma ainda precisam ser verificados. Por isso `timse()` calcula
a fórmula publicada, enquanto `official_timse()` continua reservado ao avaliador
certificado. Não foi inventada penalidade para lacunas ou multiplicador de uptime.

A suíte completa passou com **55 testes**, incluindo cálculos manuais, empates,
limiares irregulares, comparação com soma direta, escala cúbica, lacunas, alinhamento,
casos degenerados e geração de figuras/HTML. Os gráficos foram inspecionados.

```bash
MPLCONFIGDIR=/tmp/benchlab-matplotlib OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -v
```

As previsões sobrepostas não são independentes. Não foram calculados intervalos
de confiança nem feitas afirmações de significância a partir desses pares.
Para selecionar novos hiperparâmetros com TIMSE, use blocos cronológicos de
validação e reserve outro período final; veja [divisões temporais](validation.md).
