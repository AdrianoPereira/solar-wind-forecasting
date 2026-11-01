# Busca de hiperparâmetros da GRU com attention

`benchlab_forecasting.attention_gru_tuning` executa uma busca limitada e
reproduzível. Cada candidato é treinado em origens anteriores a um bloco de
parada antecipada. O checkpoint selecionado pela parada antecipada fica fixo
durante todos os blocos de validação externa. A configuração vencedora é salva
em `best_config.json` **antes** de avaliar o teste final. Nenhum alvo ausente é
interpolado, e treino, parada antecipada, validação e teste têm intervalos de
alvo separados pelo horizonte de previsão.

`--metric timse` (padrão) usa a [fórmula publicada](validation.md) nos mesmos
pares observados para todos os candidatos elegíveis. `--metric rmse` é uma
alternativa. Só entram no ranking candidatos com previsões finitas em toda a
grade de validação; cobertura e falhas permanecem registradas. Os scores de
TIMSE de blocos individuais servem para diagnóstico: a soma ou média deles não
é o TIMSE global, pois os limiares são recalculados em cada bloco.

## Executar a busca padrão

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/benchlab-matplotlib \
uv run --locked --extra forecast --extra lstm \
  python -m benchlab_forecasting.attention_gru_tuning \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --train-start 2019-01-08 \
  --early-stop-start 2019-08-01 --early-stop-hours 168 \
  --validation-starts 2019-09-01 2019-10-01 \
  --validation-hours 168 \
  --test-start 2019-10-15 --test-hours 168 \
  --epochs 30 --patience 6 --metric timse \
  --lstm-model data/experiments/lstm_2019/model.pt \
  --gru-model data/experiments/gru_2019/model.pt \
  --output-dir data/experiments/attention_gru_tuning_2019_reproduced
```

Use sempre um diretório de saída novo e vazio. O padrão compara três versões:
attention com 32 unidades e 4 cabeças, a mesma com atalho do último estado da
GRU e uma versão com 16 unidades e 2 cabeças. Os demais hiperparâmetros são
iguais. O comando aceita até 32 candidatos em um arquivo JSON:

```json
[
  {"name": "small", "config": {"hidden_size": 16, "attention_heads": 2}},
  {"name": "regularized", "config": {"dropout": 0.2}}
]
```

Passe esse arquivo com `--candidates caminho/candidates.json`. Cada objeto
sobrescreve os campos de `AttentionGRUConfig`; todos devem usar o mesmo
horizonte. `--epochs`, `--patience`, `--seed` e `--threads` definem os valores
base. O treinamento da rede ainda minimiza MSE e escolhe a época pelo RMSE do
bloco de parada antecipada; `--metric` escolhe **entre configurações** nos
blocos posteriores. Isso evita usar o mesmo bloco para escolher época e
hiperparâmetros, mas não equivale a otimizar diretamente TIMSE nos pesos.

Também é possível criar uma grade cartesiana com `--search-space
caminho/search_space.json`, em lugar de `--candidates`:

```json
{
  "hidden_size": [16, 32],
  "attention_heads": [2, 4],
  "residual_scale": [0.0, 1.0]
}
```

Esse exemplo produz oito tentativas `trial_001` a `trial_008`, em ordem
determinística. A grade aceita até 32 combinações e rejeita configurações
inválidas, como tamanho oculto não divisível pelo número de cabeças. Horizonte,
latência assumida, semente e orçamento de épocas permanecem fixos na grade.

## Busca executada em 2019

Foram usadas 808 emissões de treino, 28 de parada antecipada em agosto, dois
blocos de 28 emissões em setembro e outubro para seleção e 28 emissões de teste
de 15–21/10/2019. Todos os candidatos cobriram 100% da grade de validação.
Nos pares observados comuns aos candidatos:

| Candidato | RMSE validação (km/s) | TIMSE validação ((km/s)³) |
| --- | ---: | ---: |
| `attention_small` | 53,27 | 1.861.534 |
| `attention_residual` | 49,81 | 2.495.674 |
| `attention` | 51,17 | 2.551.141 |

Pelo critério escolhido, venceu `attention_small`. Seu checkpoint está em
`data/experiments/attention_gru_tuning_2019/candidates/attention_small/model.pt`.
A melhor época foi 8, com 3.793 parâmetros. Cada candidato foi treinado uma
vez e avaliado com pesos fixos nos blocos de seleção e teste.

No teste, os modelos com cobertura completa foram comparados nos mesmos
**1.537 pares observados**. O vencedor teve RMSE de **34,67 km/s** e TIMSE de
**2.444.263 (km/s)³**; a LSTM teve 33,76 e 1.917.970, e a GRU teve 29,00 e
2.387.353. A recorrência cobriu 91,57% da grade e não entra nessa comparação.
O vencedor produziu as 72 horas esperadas, sem falhas, e sua inferência mediana
foi de cerca de 9,8 ms por emissão em CPU.

Esses resultados são um piloto limitado a 2019. As escolhas de arquitetura já
foram influenciadas por análises anteriores de novembro, portanto outubro não
é um teste independente de todo o processo de pesquisa. A busca demonstra o
pipeline de seleção, mas ainda não comprova ganho da attention nem qualifica o
modelo para a submissão operacional.

O diretório de saída registra `manifest.json`, `leaderboard.csv`,
`validation_predictions.parquet`, relatórios, diagnósticos e um checkpoint por
candidato. `data/` está no `.gitignore`; preserve os artefatos que precisar
distribuir ou reproduza a busca com o comando acima.

## Busca extensa com Optuna

`benchlab_forecasting.attention_gru_optuna` usa o amostrador TPE do Optuna para
testar 100 configurações por padrão. O espaço inclui número de cabeças, tamanho
oculto, janela de entrada, grupo de variáveis, dropout, tamanho do lote, taxa de
aprendizado, decaimento de pesos e escala do atalho residual. Cada tentativa
usa o mesmo treino e o mesmo bloco de parada antecipada. A escolha entre
tentativas usa somente os blocos posteriores de validação; o teste reservado é
avaliado uma vez com `--finalize`, depois que a busca termina.

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked --extra lstm --extra hpo \
  python -m benchlab_forecasting.attention_gru_optuna \
  --data data/ace_hourly/ace_swepam_hourly.parquet \
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet \
  --train-start 2019-01-08 \
  --early-stop-start 2019-08-01 --early-stop-hours 168 \
  --validation-starts 2019-09-01 2019-10-01 \
  --validation-hours 336 \
  --test-start 2019-11-01 --test-hours 168 \
  --epochs 30 --patience 6 --metric timse \
  --trials 100 --threads 2 --finalize \
  --lstm-model data/experiments/lstm_2019/model.pt \
  --gru-model data/experiments/gru_2019/model.pt \
  --output-dir data/experiments/attention_gru_optuna_2019
```

`--train-start` define a primeira emissão de treino. O treino termina antes de
`--early-stop-start`, deixando o horizonte de 72 horas livre entre os alvos dos
dois blocos. `--early-stop-hours`, `--validation-hours` e `--test-hours`
definem a duração das respectivas janelas de emissões; `--step-hours` define a
cadência, de seis horas por padrão. As quatro etapas precisam estar em ordem e
sem sobreposição dos alvos. O período de 2019 acima é exploratório: análises
anteriores do projeto já influenciaram as escolhas do modelo e dos períodos.

Para continuar uma busca interrompida, execute o mesmo comando com `--resume`.
`--trials` significa o total pretendido; pode ser aumentado para ampliar a
busca. A retomada confere os períodos, arquivos de entrada, espaço, versões e
código usados antes de acrescentar tentativas. `study.db` guarda o estado do
Optuna; `leaderboard.csv` mostra os scores, e `trials/` contém os checkpoints,
históricos e previsões. Sem `--finalize`, o teste continua reservado. Não use
`--finalize` novamente depois que `test/` tiver sido criado.

## Executar em várias GPUs

O paralelismo da busca usa **um processo e uma tentativa por GPU**. Cada processo
treina uma configuração completa em sua GPU; as tentativas compartilham o banco
Optuna e os artefatos do mesmo diretório. O modelo e seus lotes são movidos para
CUDA durante o treino. Checkpoints são gravados com tensores em CPU, e a
inferência carregada continua em CPU.
Aqui, tuning significa busca de hiperparâmetros: cada tentativa inicia os pesos
do zero. O fluxo não continua o treinamento de um checkpoint anterior.

O `pyproject.toml` deste projeto aponta `torch` para o índice **CPU**. No ambiente
da máquina com GPUs, instale nesse mesmo `.venv` uma build CUDA compatível com o
driver usando o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).
Depois, use `uv run --no-sync`, pois uma sincronização pelo lock atual repõe a
build CPU. Confirme antes de começar:

```bash
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.is_available())'
```

Exemplo com GPUs físicas 0 e 1 e 100 tentativas no total:

```bash
OUT=data/experiments/attention_gru_optuna_2019_gpus
ARGS=(
  --data data/ace_hourly/ace_swepam_hourly.parquet
  --mag-data data/ace_mag_hourly/ace_mag_hourly_partial.parquet
  --train-start 2019-01-08
  --early-stop-start 2019-08-01 --early-stop-hours 168
  --validation-starts 2019-09-01 2019-10-01 --validation-hours 336
  --test-start 2019-11-01 --test-hours 168
  --epochs 30 --patience 6 --metric timse
  --trials 100 --threads 2 --device cuda
  --output-dir "$OUT"
)
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m benchlab_forecasting.attention_gru_optuna "${ARGS[@]}" --prepare
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 \
    uv run --no-sync python -m benchlab_forecasting.attention_gru_optuna \
    "${ARGS[@]}" --resume --worker-id "$gpu" > "$OUT/worker_$gpu.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m benchlab_forecasting.attention_gru_optuna \
  "${ARGS[@]}" --resume --finalize
```

Em cada processo, a GPU visível vira `cuda:0`. `--trials` é o total compartilhado,
não a quantidade por GPU. A última chamada seleciona o melhor checkpoint e
avalia o teste reservado uma vez, após todos os processos terminarem. Para
retomar após interrupção, use os mesmos argumentos com `--resume`; se houver
tentativas `RUNNING` de processos que já morreram, execute uma vez com
`--resume --recover-running` **somente depois** de parar todos os workers.
Essas tentativas passam a `FAIL` e contam no total; aumente `--trials` caso
queira repô-las. Mantenha estudo e workers no mesmo sistema de arquivos com
suporte a `flock` e SQLite. Compare tempo e uso de GPU antes de ampliar o
número de processos: o modelo é pequeno e o preparo das janelas ainda usa CPU.

Um JSON passado por `--space` altera o espaço de busca. Listas indicam escolhas
discretas; `low` e `high` indicam intervalos numéricos:

```json
{
  "hidden_size": [16, 32, 64],
  "attention_heads": [2, 4],
  "learning_rate": {"low": 0.0003, "high": 0.003, "log": true},
  "dropout": [0.0, 0.1, 0.2, 0.3]
}
```

O orçamento `--epochs` é o máximo por tentativa. A parada antecipada reduz esse
limite quando o RMSE interno deixa de melhorar. A busca otimiza TIMSE ou RMSE
**entre configurações**; o ajuste dos pesos ainda usa MSE. Uma busca longa pode
ajustar demais as decisões aos blocos de validação, então um resultado promissor
precisa de avaliação posterior em outro período.
