# Histórico horário do vento solar

## Download paralelo para o modelo híbrido

`download_training_data.sh` baixa em paralelo, a partir de 2011, tudo o que o
modelo de buraco coronal + recorrência + L1 precisa. Cada job tem log próprio em
`data/download_logs/<execução>/` e o resumo fica em `status.tsv`:

- `download_ace_level2.py`: ACE Level 2 horário do CDAWeb via HAPI (`AC_H2_SWE`,
  que termina em 2024-07-09, e `AC_H2_MFI`), um mês por requisição, em
  `data/ace_level2/`. Cobre horas em que o produto operacional tem status não
  nominal; serve para treino e auditoria, não como disponibilidade em tempo real.
- ACE operacional (SWEPAM e MAG) em `data/ace_hourly_<início>_<fim>/` e
  `data/ace_mag_hourly_<início>_<fim>/`, sem sobrescrever os Parquet atuais.
- SDO/AIA 193 Å a cada 6 h em `data/aia_synoptic/`, um cliente por ano e no
  máximo `AIA_JOBS` (3) ao mesmo tempo. O cache existente de 2020–2025 é
  reaproveitado e as falhas anteriores são tentadas de novo.

```bash
mkdir -p data/download_logs
nohup bash scripts/download_training_data.sh > data/download_logs/latest.out 2>&1 &
```

Variáveis: `START_YEAR` (2011), `END_DATE` (ontem, UTC), `AIA_JOBS`,
`AIA_WORKERS` (2), `AIA_WAVELENGTH` (193), `CA_FILE`, `SKIP_AIA=1`, `SKIP_ACE=1`.
Repetir o comando retoma pelo cache. O código de saída é 1 se algum job falhar ou
deixar cobertura incompleta. O cache da AIA guarda também candidatos rejeitados:
um ano ocupa de 10 a 20 GB.

## Imagens solares SDO/AIA

`download_aia_images.py` baixa FITS sinópticos de 1024 × 1024 pixels do
[arquivo JSOC](https://jsoc1.stanford.edu/data/aia/synoptic/), preservando os bytes
originais. A [NASA descreve as opções de acesso](https://sdo.gsfc.nasa.gov/data/dataaccess.php).
O nível de processamento é lido de `LVL_NUM`: a amostra histórica de março/2024
verificada durante o desenvolvimento tem nível 1.5. O script não assume que o
histórico revisado seja idêntico ao produto NRT nem aplica calibração adicional.

```bash
uv sync --locked --group images

# Estimate the number of images and storage without downloading anything.
uv run --group images python scripts/download_aia_images.py \
  --start 2024-01-01 --end 2024-12-31 --dry-run

# Small pilot: four forecast origins per day, AIA 193 angstroms.
uv run --group images python scripts/download_aia_images.py \
  --start 2024-03-01 --end 2024-03-07 --wavelength 193 \
  --cadence-hours 6 --workers 2 \
  --ca-file /etc/pki/tls/certs/ca-bundle.crt
```

As datas são obrigatórias e inclusivas em UTC. Por padrão, o canal é 193 Å,
a cadência é de seis horas e o destino é `data/aia_synoptic/`.
Um ano de 365 dias contém 1.460 emissões nessa cadência: cerca de 1,8 GB,
estimados com uma amostra de 1,25 MB, antes de candidatos rejeitados e metadados.
Tamanhos reais variam por canal e imagem. Não é necessário baixar a cadência
integral do arquivo. `--dry-run` não acessa a rede nem escreve arquivos.

Para cada emissão, o script escolhe a imagem válida mais recente com horário
real `DATE-OBS` anterior ou igual à emissão menos `--latency-minutes` (60 por
padrão). Busca até `--max-age-minutes` (60) antes desse limite. Portanto, os
arquivos da primeira emissão podem pertencer ao dia anterior ao `--start`.
Essa latência é uma hipótese configurável; a data real de publicação histórica
não é conhecida e fica nula. Não use a data de coleta atual como disponibilidade
histórica em um backtest. O horário do nome do arquivo tem resolução de minutos;
a aceitação final usa o cabeçalho, incluindo segundos.

Valida FITS, dimensões, instrumento, canal, pixels finitos, exposição positiva
e `QUALITY == 0` (política conservadora). Procura outro candidato anterior se uma
imagem não passar. Não substitui por JPEG, não preenche lacunas e não redimensiona.
Imagens válidas examinadas, mas fora do limite temporal, podem permanecer no cache;
use o relatório para identificar as imagens efetivamente selecionadas.

Saídas:

- `raw/YYYY/MM/DD/*.fits`: imagens originais.
- Arquivos `.json` ao lado dos FITS: URL, SHA-256, tamanho, coleta, observação,
  qualidade, exposição, nível de processamento, fração de pixels finitos e WCS.
- `reports/START_END_CHANNEL_CADENCEh.json`: um resultado por emissão, imagem
  selecionada, idade, erros e cobertura. Repetir a mesma combinação de período,
  canal e cadência atualiza esse relatório; copie-o antes de comparar políticas.

`--refresh` refaz os downloads; o cache normal verifica URL, hash e FITS.
Os índices remotos ainda são consultados para selecionar os candidatos.
`--timeout`, `--attempts`, `--ca-file`, `--base-url` (HTTPS) e `--output-dir`
são configuráveis. Os workers são threads locais; este script não usa scheduler
Dask. Execute um único cliente por destino. Cobertura incompleta retorna código 1
e mantém os arquivos válidos para retomada.

Falhas agora exibem o motivo no terminal. HTTP 503 indica indisponibilidade
temporária do servidor; o downloader espaça o início das requisições em pelo
menos um segundo, usa até cinco tentativas por padrão, recuo exponencial e
respeita `Retry-After`. O intervalo de espera é compartilhado entre os workers.
Se persistir, use `--workers 1` e retome mais tarde, sem `--refresh`.

Alguns FITS históricos contêm `nan`/`-nan` sem aspas em campos auxiliares
(`OSCNMEAN`, `OSCNRMS`, `RSUN_LF`, `X0_LF`, `Y0_LF`, `GCIEC_*`, `HCIEC_*`).
O leitor compartilhado `aia_fits.py` trata apenas esses campos como indefinidos
em memória e registra cada ocorrência em `header_normalizations`. Os FITS
originais permanecem intactos. Validação estrutural e de qualidade, exposição,
horário, canal e WCS continua ativa. O notebook visualizador usa o mesmo leitor.

## Campo magnético ACE/MAG

O script separado baixa o produto operacional horário `YYYYMM_ace_mag_1h.txt`
do [arquivo mensal NASA/SolarSoft](https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/).
O cabeçalho da fonte identifica coordenadas **GSM**, campo em **nT** e ângulos
em **graus**. Não é o produto científico Level 2.

```bash
uv run python scripts/download_ace_mag_hourly.py \
  --start 2024-01-01 --end 2024-12-31 --workers 4 \
  --ca-file /etc/pki/tls/certs/ca-bundle.crt
```

Sem datas, busca de 2020-01-01 até hoje em UTC. Escolha o mesmo intervalo
do SWEPAM para combinar os produtos por `timestamp_utc`.
O argumento `--ca-file` é opcional quando as CAs padrão já são suficientes.

Saídas em `data/ace_mag_hourly/`, separadas dos dados de plasma:

- `ace_mag_hourly.parquet`: `timestamp_utc`, `status`, `is_valid`,
  `bx_gsm_nt`, `by_gsm_nt`, `bz_gsm_nt`, `bt_nt`,
  `latitude_gsm_deg`, `longitude_gsm_deg` e `source_file`.
- Colunas `_raw`: valores originais, inclusive sentinelas.
- `raw/`: arquivos mensais e metadados de origem, coleta e SHA-256.
- `download_report.json`: período, cobertura, validade e falhas mensais.

Status diferente de zero e sentinela `-999.9` viram nulos nas colunas tratadas.
Componentes Bx/By/Bz negativos são válidos; Bt negativo e ângulos fora dos
intervalos do produto são anulados. `is_valid` exige Bx, By, Bz e Bt válidos;
os ângulos são avaliados separadamente. Bt é preservado como fornecido pela fonte,
sem recalculá-lo a partir dos componentes médios. Não há imputação.

Cache, `--refresh`, `--scheduler`, `--timeout`, `--attempts` e `--output-dir`
funcionam como no script SWEPAM abaixo. Em falhas, salva
`ace_mag_hourly_partial.parquet` sem sobrescrever o arquivo completo.
Um arquivo completo indica ausência de falhas mensais de aquisição, não
ausência de lacunas ou medições inválidas na fonte.

## Plasma ACE/SWEPAM

Na raiz do projeto:

```bash
uv sync --locked
uv run python scripts/download_ace_hourly.py --workers 4
```

O período padrão começa em **2020-01-01** e termina na data atual em UTC,
limitado à disponibilidade da fonte. Para outro intervalo (datas inclusivas):

```bash
uv run python scripts/download_ace_hourly.py --start 2020-01-01 --end 2020-12-31
```

Fonte: [arquivo mensal ACE/SWEPAM NOAA/SWPC hospedado pela NASA](https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/),
produto `YYYYMM_ace_swepam_1h.txt` de médias horárias operacionais.
Não é o produto científico Level 2; a correspondência com o alvo oficial
do BenchLab ainda precisa ser confirmada.

Saídas em `data/ace_hourly/` (configurável com `--output-dir`):

- `ace_swepam_hourly.parquet`: timestamps UTC, velocidade em km/s,
  densidade de prótons em cm⁻³, temperatura em K e status da fonte.
- `raw/`: arquivos originais com metadados de URL, coleta e SHA-256.
- `download_report.json`: cobertura, horas ausentes e falhas por mês.

As colunas com sufixo `_raw` preservam os valores originais, incluindo
sentinelas. Nas colunas tratadas, status diferente de zero ou valores negativos
viram nulos. `is_valid` indica especificamente a validade da velocidade.
Não há interpolação nem preenchimento de horas ausentes.

Meses já baixados e fechados são reutilizados; use `--refresh` para atualizá-los.
Se algum mês falhar, o script salva `ace_swepam_hourly_partial.parquet`, registra
a falha e retorna código 1. Execute novamente para tentar os meses faltantes.
Mesmo sem falhas de download, consulte o relatório para identificar lacunas
na própria fonte. Produtos históricos não comprovam disponibilidade em tempo real.

### Recuperar apenas lacunas do histórico ACE

Use o CSV produzido por `audit_ace_missing.py` para selecionar os meses a consultar:

```bash
python3 scripts/audit_ace_missing.py
python3 scripts/download_missing_ace.py --dry-run
python3 scripts/download_missing_ace.py
```

Na auditoria, os dois instrumentos são comparados até o mês mais recente em
cache de qualquer um deles, para revelar arquivos ausentes no fim da série.

A fonte fornece arquivos **mensais**, portanto uma lacuna de uma hora exige
consultar o mês inteiro. Por padrão, o script consulta apenas meses cujo arquivo
local está ausente ou que têm horas sem registro. Para tentar novamente também
registros com status ou valor inválido, acrescente `--retry-invalid`; isso pode
envolver centenas de meses sem que a fonte ofereça valores melhores.

Os arquivos obtidos ficam em `data/quality/ace_recovery_cache/`, com URL e
SHA-256. Apenas valores que estavam ausentes no CSV de auditoria e que são
válidos na nova cópia vão para `data/quality/ace_recovered_values.csv`.
Execuções posteriores reutilizam o cache e preservam valores recuperados em
execuções anteriores. `data/quality/ace_recovery_report.json` informa os meses
consultados, falhas e valores ainda não resolvidos. Os arquivos brutos originais
e os Parquet consolidados não são alterados. Use `--instrument swepam` ou
`--instrument mag` para restringir o produto, e `--refresh` para baixar novamente
meses já presentes no cache de recuperação.
Se necessário, `--ca-file` aceita um arquivo PEM de CAs confiáveis.

Os achados da verificação de disponibilidade após 2011 estão em
`data/quality/ace_availability_post2011.md`.

### Dataset ACE interpolado

```bash
.venv/bin/python scripts/build_interpolated_ace.py
```

O script reconstrói a grade horária conjunta de ACE/SWEPAM e ACE/MAG a partir
dos arquivos mensais locais e incorpora as células recuperadas em
`data/quality/ace_recovered_values.csv`. Grava
`data/quality/ace_interpolated_all_internal.parquet` e um relatório JSON com
contagens por campo. Cada variável numérica é interpolada linearmente entre
observações válidas, inclusive através de lacunas longas. A longitude usa o
menor arco circular ao cruzar 0°/360°. Não há extrapolação antes da primeira
observação nem depois da última.

Cada campo tem colunas `*_is_recovered`, `*_is_interpolated` e
`*_interpolation_gap_hours`; o último valor registra o tamanho **total** da
lacuna preenchida, em horas. `swepam_original_status` e
`mag_original_status` preservam o status anterior à recuperação/interpolação.
Valores interpolados são sintéticos e usam uma observação futura como âncora:
não os trate como alvos observados em avaliação nem como entradas disponíveis
em tempo real. Os arquivos brutos e Parquet originais permanecem intactos.

Para reproduzir as figuras de comparação:

```bash
.venv/bin/python scripts/plot_ace_interpolation.py
```

As imagens ficam em `data/quality/figures/`: cobertura anual original versus
interpolada, a maior lacuna de SWEPAM, a maior lacuna de MAG e um exemplo com
valores recuperados. O arquivo `ace_figures_manifest.json` identifica os
períodos plotados. Preto representa medições originais válidas, laranja a série
com interpolação e verde os valores recuperados da fonte.

Para exportar janelas completas para treinamento com 3, 7 e 14 dias de entrada
e 72 horas de velocidade futura observada:

```bash
.venv/bin/python scripts/build_complete_ace_windows.py
```

As saídas estão em `data/training_windows/`, com instruções em
`data/training_windows/README.md`. A seleção exige todas as nove entradas
ACE/SWEPAM + ACE/MAG e todos os 72 alvos sem lacunas, excluindo células
interpoladas. Medições recuperadas do mesmo arquivo operacional são aceitas,
mas nenhuma das janelas selecionadas as utiliza. Por ser uma regra estrita,
não há janelas depois de 2011 no histórico atual.

### Erro de certificado SSL

Se aparecer `CERTIFICATE_VERIFY_FAILED`, informe um arquivo PEM de CAs confiáveis
do sistema ou fornecido pela administração da rede. Por exemplo, em sistemas
que disponibilizam o pacote neste caminho:

```bash
uv run python scripts/download_ace_hourly.py --workers 4 \
  --ca-file /etc/pki/tls/certs/ca-bundle.crt
```

O script mantém a verificação SSL e adiciona esses certificados às CAs padrão.
O conteúdo PEM é enviado aos workers, sem exigir o mesmo caminho em cada nó.
Se o erro persistir, solicite à administração a CA do proxy institucional.
Falhas de certificado não são repetidas automaticamente; após corrigir a CA,
execute novamente o comando para recuperar os meses faltantes.

## Execução distribuída

PyArrow grava o Parquet e Dask Distributed executa uma tarefa por mês.
`--workers 4` cria quatro workers locais com uma thread cada, no mesmo processo.
A escrita do Parquet consolidado acontece no cliente após a coleta dos meses.

Para distribuir entre máquinas, instale o mesmo projeto com `uv sync --locked`
em todos os nós e use a mesma versão de Python. No nó coordenador:

```bash
uv run dask scheduler --host 0.0.0.0
```

Em cada nó de execução (substitua `HOST` pelo endereço do coordenador):

```bash
uv run dask worker tcp://HOST:8786 --nworkers 2 --nthreads 1
```

No cliente:

```bash
uv run python scripts/download_ace_hourly.py --scheduler tcp://HOST:8786
```

Os workers precisam de acesso à internet e ao diretório de saída compartilhado,
no mesmo caminho absoluto do cliente, com permissão de escrita. O script envia
seu módulo aos workers. Use um único cliente por diretório de saída.
No modo remoto, a quantidade de downloads simultâneos depende dos workers do
cluster; `--workers` controla apenas o modo local. Use o cluster em rede privada.

Referência: [cliente Dask Distributed](https://distributed.dask.org/en/stable/client.html).
