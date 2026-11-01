# Solar Wind Forecasting · FDL BenchLab 2026

Previsão horária da velocidade do vento solar medida pelo ACE em L1, de 1 a 72 horas
à frente. É a nossa participação no
[FDL BenchLab Solar Wind Prediction Tournament 2026](overvie.md): a organização roda um
container Docker sem estado a cada 6 horas, em CPU, com 15 minutos de limite, e a
métrica oficial é o TIMSE.

O melhor modelo é um **híbrido de gradient boosting**: um único modelo para as 72 horas,
que combina três fontes de informação, todas cortadas em `t0 − 1 h`:

- **plasma em L1 (ACE):** última velocidade e sua idade, médias de 6, 24 e 72 h,
  tendência, densidade e temperatura;
- **recorrência de 27 dias:** velocidade uma e duas rotações solares antes de cada hora
  prevista;
- **buracos coronais (SDO/AIA 193 Å):** fração de área escura em seis faixas de 15° de
  longitude, em imagens de 0, 24 e 48 h antes.

![RMSE por antecedência no teste](docs/presentation/figures/rmse_by_lead.png)

## Resultados

Período de teste: jan/2024 a out/2026, com 4.105 emissões a cada 6 h. Esse período
não foi usado em nenhum treino nem escolha de modelo.

| Modelo | RMSE (km/s) | RMSE em 72 h |
| --- | ---: | ---: |
| **Híbrido, LightGBM ajustado** | **75,3** | **91** |
| Híbrido, XGBoost | 75,7 | — |
| Híbrido, scikit-learn HGB | 76,0 | — |
| Híbrido, LightGBM ajustado, sem buracos coronais | 78,6 | 96 |
| Persistência | 97,8 | 129 |
| Recorrência de 27 dias | 107,8 | 110 |

- O erro fica 23% abaixo da persistência no geral e 30% abaixo em 72 h.
- A persistência só é melhor nas primeiras ~10 h.
- Os buracos coronais reduzem o RMSE em 3,4 km/s (IC 95% 1,4–5,5) e em 5,5 km/s em 72 h.
- Cinco bibliotecas de boosting (scikit-learn, LightGBM, XGBoost, CatBoost e NGBoost)
  ficam a menos de 2 km/s umas das outras.
- **Atenção ao TIMSE:** somado no período inteiro, ele recompensa previsões enviesadas
  para cima. Uma constante de 700 km/s tem TIMSE melhor que o de qualquer modelo. Ainda
  falta confirmar com a organização como o TIMSE é agregado.

Detalhes em [docs/hybrid.md](docs/hybrid.md). A apresentação dos resultados está em
[docs/presentation/benchlab_results.pptx](docs/presentation/benchlab_results.pptx).

## Estrutura

```
src/benchlab_forecasting/  pacote: dados, features, modelos, validação e TIMSE
  hybrid.py                modelo híbrido (experiment e tune)
  coronal_holes.py         máscara e frações de buraco coronal nas imagens AIA
  validation.py            métricas, TIMSE e bootstrap em blocos
scripts/                   download de dados, construção de features e figuras
scripts/slurm/             jobs SLURM para treino e busca de hiperparâmetros
tests/                     testes unitários (unittest)
notebooks/                 exploração dos dados ACE e das imagens AIA
docs/                      documentação de cada modelo e experimento (em português)
docs/presentation/         apresentação, figuras e scripts que as geram
overvie.md                 resumo do torneio, regras e plano
```

`data/` não vai para o repositório (são cerca de 96 GB). Todos os arquivos dela são
recriados pelos scripts abaixo.

## Instalação

São necessários Python 3.11 ou mais novo (o projeto usa 3.14, em `.python-version`) e
o [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:AdrianoPereira/solar-wind-forecasting.git
cd solar-wind-forecasting
uv sync --locked --extra hybrid --extra boosting --group images --group notebooks
```

| Extra ou grupo | Para que serve |
| --- | --- |
| `--extra hybrid` | modelo híbrido com scikit-learn |
| `--extra boosting` | LightGBM, XGBoost, CatBoost e NGBoost |
| `--group images` | leitura das imagens FITS do AIA (astropy) |
| `--group notebooks` | Jupyter, matplotlib e scipy (figuras) |
| `--extra forecast` | Prophet |
| `--extra lstm` / `--extra hpo` | LSTM, GRU e Transformers em CPU (PyTorch); Optuna |

Para instalar mais de um conjunto em momentos diferentes, use `uv sync --locked --inexact ...`.
Assim os pacotes já instalados não são removidos.

## Como rodar o modelo híbrido

Todos os comandos rodam na raiz do repositório.

**1. Baixar os dados** (ACE Level 2 e operacional, e AIA 193 Å a cada 6 h desde 2011).
O download é retomável e leva horas:

```bash
mkdir -p data/download_logs
nohup bash scripts/download_training_data.sh > data/download_logs/latest.out 2>&1 &
```

As variáveis opcionais (`START_YEAR`, `END_DATE`, `AIA_JOBS`, `SKIP_AIA=1`…) estão no
cabeçalho do script. Os downloads individuais estão em
[scripts/README.md](scripts/README.md).

**2. Construir as features:**

```bash
.venv/bin/python scripts/build_coronal_hole_features.py --workers 16   # data/features/coronal_holes_0193_6h.parquet
.venv/bin/python scripts/build_speed_series.py                          # série de velocidade usada nas figuras
```

**3. Treinar e avaliar.** Cada comando treina as variantes escolhidas, seleciona o número
de árvores na validação (2022–2023) e avalia uma vez no teste:

```bash
export OMP_NUM_THREADS=16
# scikit-learn (padrão): variantes com MSE, sem buracos coronais e com pesos TIMSE
.venv/bin/python -m benchlab_forecasting.hybrid experiment \
  --variants mse mse_no_ch timse --output-dir data/experiments/hybrid_2011_2026
# outra biblioteca: --model lightgbm | xgboost | catboost | ngboost
.venv/bin/python -m benchlab_forecasting.hybrid experiment --model xgboost \
  --variants mse mse_no_ch timse --output-dir data/experiments/hybrid_xgboost_2011_2026
```

Cada experimento grava `report.json` (métricas e intervalos de confiança),
`test_predictions.parquet` e os modelos (`model_*.pkl`).

**4. Busca de hiperparâmetros** (40 tentativas, retomável). Ela gera o LightGBM ajustado
em `data/experiments/hybrid_lightgbm_tuning/`:

```bash
.venv/bin/python -m benchlab_forecasting.hybrid tune --model lightgbm --trials 40 --threads 32 \
  --output-dir data/experiments/hybrid_lightgbm_tuning
```

**5. Figuras e apresentação:**

```bash
.venv/bin/python docs/presentation/make_figures.py   # figuras da apresentação, em docs/presentation/figures/
```

O `.pptx` é gerado por `docs/presentation/build.js`, com `pptxgenjs`. Para refazê-lo:

1. Rode `npm install` em `docs/presentation/`.
2. Rode `node build.js`, com `PPTX_SKILL` apontando para a pasta que contém `scripts/apply_theme.js`. Esse script aplica as cores do tema e não faz parte deste repositório.

### Em cluster com SLURM (SDumont)

Nos nós de login, qualquer processo é encerrado depois de 30 minutos de CPU. Sem
`OMP_NUM_THREADS`, as bibliotecas abrem uma thread por núcleo e o processo morre.
Treinos longos devem ir por SLURM. Ajuste conta e partição nos arquivos `.sbatch`.

```bash
mkdir -p data/slurm_logs
MODEL=catboost sbatch scripts/slurm/hybrid_experiment.sbatch   # VARIANTS, MAX_ITER e OUTPUT_DIR são opcionais
sbatch scripts/slurm/tune_lightgbm.sbatch
```

## Testes

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Alguns testes precisam dos extras opcionais (`boosting`, `lstm`, `forecast`). Os
demais rodam só com `hybrid`.

## Outros modelos testados

Antes do híbrido testamos Prophet, LSTM, GRU, modelos com attention e Transformers.
A documentação de cada um:

- [Prophet e baselines](docs/forecasting.md), [tuning do Prophet](docs/tuning.md) e
  [Prophet em janelas completas](docs/prophet_complete_windows.md); CLI `benchlab-forecast`
  (`uv sync --locked --extra forecast`).
- [LSTM](docs/lstm.md), [GRU](docs/gru.md), [GRU com attention](docs/attention_gru.md),
  [busca da GRU com attention](docs/attention_gru_tuning.md) e
  [LSTM com attention, Transformer e Patch Transformer](docs/attention_models.md).
- [Validação temporal](docs/validation.md), [TIMSE com figuras](docs/timse.md) e
  [gráficos de resultados](docs/plotting.md).

## Submissão ao torneio

O kit oficial (`benchlab-starter-kit/`) é um repositório separado da organização e está
fora deste repositório. Ele contém o token da equipe em `.team`, que nunca deve ser
versionado. Clone o kit ao lado do projeto, conforme as instruções da organização.

O container de submissão com o modelo híbrido ainda está em preparação.
