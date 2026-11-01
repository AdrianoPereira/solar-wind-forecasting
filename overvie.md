# FDL BenchLab Solar Wind Prediction AI Tournament 2026

> **Atualização de métrica em 05/10/2026:** a fórmula publicada do TIMSE foi
> implementada a partir do slide fornecido e da FAQ. As referências
> abaixo a RMSE ponderado por horizonte e multiplicador de uptime são históricas
> e não devem orientar o score atual. Veja [fórmula e validação com figuras](docs/timse.md).

> Documento consolidado em 24 de agosto de 2026 (America/Sao_Paulo).  
> Idioma: português do Brasil.  
> Escopo: informações públicas do torneio, análise técnica e plano recomendado para a equipe.

## Resumo executivo

O FDL BenchLab Solar Wind Prediction AI Tournament 2026 é uma competição operacional de aprendizado de máquina para prever a velocidade do vento solar ambiente a montante da Terra com 72 horas de antecedência. A cada 6 horas, o sistema da organização executará um container Docker congelado de cada equipe; cada execução deverá produzir uma previsão horária para as 72 horas seguintes. A avaliação será feita contra dados futuros e não vistos do satélite ACE, em uma janela ao vivo anunciada para novembro de 2026 a janeiro de 2027.

O desafio não é apenas obter baixo erro em um conjunto histórico. O ranking combina precisão, por meio de RMSE ponderado com maior peso próximo da 72ª hora, e confiabilidade operacional, por meio de um fator de uptime. A submissão precisa buscar automaticamente os dados públicos disponíveis no instante da inferência, funcionar sem estado persistente, executar em CPU e tolerar atrasos e lacunas sem falhar.

O prêmio geral anunciado soma US$ 20.000: US$ 9.000, US$ 5.000, US$ 3.000, US$ 2.000 e US$ 1.000 para as cinco primeiras colocações. A organização também prevê wildcards para regimes/eventos específicos, além de possibilidades de artigo conjunto, apresentações, apoio de viagem, colaboração de pesquisa e implantação operacional.

Em 24/08/2026, a especificação pública ainda está incompleta. Embora a página chame o arquivo vinculado de “briefing pack”, o documento acessível é um `Benchlab_One_pager`; o briefing técnico completo está anunciado para 01/09/2026. Portanto, detalhes como fórmula exata dos pesos, esquema de saída, limites de CPU/memória/tempo, produto ACE definitivo e regras exatas do code freeze ainda precisam ser confirmados.

## 1. O problema

O vento solar é um fluxo contínuo de plasma que sai do Sol e influencia o ambiente espacial próximo à Terra. Mudanças em sua velocidade e estrutura afetam a magnetosfera e podem contribuir para condições adversas de clima espacial, relevantes para satélites, comunicações, navegação, redes elétricas e missões tripuladas.

O torneio busca melhorar a previsão operacional da velocidade do vento solar ambiente, não apenas demonstrar desempenho retrospectivo. Segundo a descrição oficial:

- alvo: velocidade volumétrica (*bulk speed*) horária do vento solar medida pelo ACE;
- localização: a montante da Terra, nas proximidades do ponto L1;
- horizonte: 72 horas;
- resolução da saída: horária;
- frequência de execução: uma nova previsão a cada 6 horas;
- avaliação: dados futuros, reais e não vistos;
- ambiente: container Docker executado em infraestrutura CPU da organização;
- duração: anunciada como três meses e, na página, como novembro de 2026 a janeiro de 2027.

A página caracteriza o objetivo principal como previsão do vento solar ambiente, e não de eventos transientes impulsivos. CMEs, HSSs e diferentes regimes podem, porém, gerar avaliações wildcard separadas.

## 2. Como o ciclo operacional funciona

| Etapa | Responsabilidade da equipe | Responsabilidade da organização |
|---|---|---|
| Treinamento | Treinar e validar com computação própria | Fornecer dados, baselines e template anunciados no briefing |
| Empacotamento | Entregar container, código, dependências e README | Disponibilizar o template e o harness de avaliação |
| Pré-deploy | Garantir execução automática por um único comando | Validar o pipeline entre 1 e 10 de novembro |
| Code freeze | Congelar pesos, código e configuração conforme a regra final | Preservar as submissões para o período ao vivo |
| Inferência | Buscar apenas dados disponíveis no instante do disparo e produzir a trajetória prevista | Disparar o container a cada 6 horas em CPU |
| Avaliação | Não há acesso aos alvos futuros | Comparar previsões com ACE, computar métricas e uptime |
| Acompanhamento | Observar previsões e classificação | Publicar leaderboard ao vivo |

“Atômica” e “stateless” significam que cada execução deve ser autocontida. O modelo não deve depender de memória de uma execução anterior nem de dados que só se tornaram disponíveis depois do horário de emissão da previsão.

## 3. Cronograma público

| Período/data | Marco anunciado | Situação em 24/08/2026 |
|---|---|---|
| Julho–agosto de 2026 | Pré-registro e vagas limitadas | Aberto; o one-pager informa abertura em 01/08 |
| 01/09/2026 | Briefing oficial / liberação do pacote | Futuro |
| Setembro–outubro de 2026 | Construção local do protótipo | Futuro |
| 01/11/2026 | Data mostrada na página como “Submission” | Precisa ser reconciliada com a janela seguinte |
| 01–10/11/2026 | Countdown, confirmação e validação obrigatória do pipeline | Janela de pré-deploy |
| 11/11/2026 | Lançamento e início da avaliação ao vivo | Anunciado na linha do tempo da página |
| Nov/2026–jan/2027 | Execuções a cada 6 horas | A página usa esse intervalo; o one-pager diz “três meses completos” |
| A definir | Encerramento, wildcards e anúncio dos vencedores | Ainda não publicado |

Há uma ambiguidade operacional importante: a página exibe “Submission: November 1, 2026”, mas também reserva 1–10 de novembro para validação e indica lançamento em 11 de novembro. A equipe deve tratar 1º de novembro como prazo interno de entrega e usar os dias seguintes apenas para compatibilização com o harness, até que o briefing esclareça o que pode ser alterado.

## 4. Participação, equipes e propriedade intelectual

### Confirmado

- Participação mundial, individual ou em equipes de até 4 pessoas.
- Público: estudantes, pesquisadores, acadêmicos, engenheiros, cientistas de dados e profissionais da indústria.
- Não é necessário conhecimento prévio em física solar; visão computacional, séries temporais, aprendizado de máquina e abordagens informadas por física são bem-vindos.
- O pré-registro é obrigatório para avançar à fase operacional.
- O limite previsto é de 100 equipes na etapa anterior ao pré-deploy.
- A preferência será dada primeiro a equipes maiores e depois por ordem de chegada.
- As equipes mantêm a propriedade intelectual de seus códigos e modelos.
- A submissão deve ser open source sob licença MIT.

### Pontos que precisam de esclarecimento

As duas fontes oficiais usam formulações diferentes para exclusões. O one-pager exclui pessoal ou pesquisadores afiliados à CU Boulder e à Trillium Technologies. A página é mais detalhada: exclui funcionários, organizadores, juízes e pessoas com acesso privilegiado à infraestrutura ou à avaliação, incluindo afiliados da Trillium e certos tipos de afiliados da CU Boulder. Pessoas ligadas a essas instituições devem pedir confirmação por escrito antes de investir no projeto.

Também existe uma tensão de redação na licença: MIT normalmente permite uso comercial, mas o one-pager associa a liberação a benchmarking não comercial e validação científica. É necessário confirmar se toda a submissão será MIT sem restrição adicional, quais “configurações” precisam ser publicadas e se pesos treinados, dados derivados e segredos de acesso entram no escopo.

O formulário público de pré-registro solicita nome, e-mail, instituição/afiliação, país, área de experiência e nível de formação ou vínculo profissional. Não é necessário login apenas para visualizar/preencher o formulário; login no Google é oferecido para salvar o progresso.

## 5. Prêmios e oportunidades

| Categoria geral | Prêmio anunciado |
|---|---:|
| 1º lugar | US$ 9.000 |
| 2º lugar | US$ 5.000 |
| 3º lugar | US$ 3.000 |
| 4º lugar | US$ 2.000 |
| 5º lugar | US$ 1.000 |
| Total geral | US$ 20.000 |

As categorias wildcard e seus valores serão definidos no lançamento. A organização informa que as janelas serão escolhidas por critérios científicos predefinidos, como limiares de velocidade ou intervalos identificados independentemente para CMEs, HSSs e regimes de vento solar; não serão escolhidas a partir do leaderboard. Como não é possível saber quais fenômenos ocorrerão depois do congelamento do código, a organização se reserva o direito de estender o período de avaliação e ajustar os bônus.

Oportunidades mencionadas para as melhores equipes:

- colaboração para implantação operacional;
- coautoria de artigo técnico com o comitê organizador;
- apresentação em conferências e possível apoio de viagem;
- continuidade em atividades de pesquisa;
- integração do modelo de melhor desempenho ao conjunto operacional no SWx-TREC e no SpaceML.

A redação pública não deixa totalmente claro se os wildcards são adicionais aos US$ 20.000 ou uma alocação dentro desse fundo. Isso deve ser confirmado no lançamento.

## 6. Dados

### 6.1 Alvo oficial

O leaderboard usará velocidade volumétrica horária do vento solar do satélite ACE em tempo real. O one-pager não identifica ainda:

- instrumento/produto e nível de processamento exatos;
- campo/coluna e unidade oficiais, embora velocidade do vento solar normalmente seja expressa em km/s;
- política de qualidade e flags;
- latência admissível para considerar uma observação “disponível”;
- regra de agregação quando previsões emitidas a cada 6 horas se sobrepõem;
- se a trajetória terá exatamente 72 valores ou inclui os dois extremos temporais.

Para desenvolvimento histórico, a NASA informa que os dados ACE/SWEPAM Level 2 incluem velocidade de fluxo e que médias horárias estão disponíveis pelo ACE Science Center e CDAWeb. A NOAA também publica um feed JSON horário do ACE/SWEPAM. O produto definitivo do torneio deve ser confirmado antes de congelar a preparação dos rótulos.

### 6.2 Entradas permitidas

É permitido usar qualquer dado público de heliofísica que possa ser acessado programaticamente no instante da inferência. A organização cita GONG, SDO e SOHO. Modelos fundacionais pré-treinados ou ajustados são permitidos quando os pesos-base são públicos.

Fontes candidatas, a validar contra o template e as regras de rede:

| Fonte | Uso possível | Risco operacional |
|---|---|---|
| ACE/SWEPAM histórico | Rótulo de treinamento, persistência e recorrência | Produto do ranking ainda não especificado |
| SDO/AIA | Imagens EUV para buracos coronais e estruturas solares | Volume, staging e latência; preferir produtos NRT apropriados |
| SDO/HMI | Magnetogramas e mapas do campo fotosférico | Staging do JSOC pode não caber no tempo de execução |
| GONG | Magnetogramas/mapas sinópticos com cobertura terrestre distribuída | Lacunas, atualização e escolha do produto |
| SOHO | Modalidade/fallback de observação solar | Cadência e compatibilidade histórica |
| WSA/CCMC | Previsão física de referência ou feature de um híbrido | API, cobertura, licença/citação e latência devem ser confirmadas |
| NOAA/SWPC | Feeds operacionais do clima espacial | Não assumir que todo feed é aceito como alvo oficial |

O acesso perto de tempo real merece atenção. A própria documentação do SDO diferencia dados de ciência, que podem exigir staging, de produtos Near Real Time recuperáveis automaticamente. Uma solução que dependa de exportação assíncrona do JSOC pode ser ótima para treinamento e inadequada para o loop operacional.

### 6.3 Lacunas

O one-pager declara que:

- períodos sem dado-alvo serão removidos da avaliação;
- dados de entrada ausentes dentro de uma janela de previsão serão preenchidos à frente a partir do último valor disponível;
- o modelo deve suportar latência ou lacunas sem encerrar com erro.

Ainda não está claro se o *forward-fill* é feito pelo harness apenas para entradas fornecidas pela organização ou se cada equipe deve aplicá-lo às fontes que busca. A implementação deve possuir política própria, explícita e testada de fallback.

## 7. Métrica e avaliação

A métrica principal anunciada é RMSE ponderado sobre toda a janela, com maior peso para a precisão no marco crítico de 72 horas. Em termos gerais, uma métrica desse tipo tem a forma:

\[
\operatorname{wRMSE} =
\sqrt{\frac{\sum_i w_i(\hat{v}_i-v_i)^2}{\sum_i w_i}}
\]

Essa expressão serve apenas para explicar a família da métrica. A organização ainda não publicou os pesos \(w_i\), a unidade de agregação, o tratamento de previsões sobrepostas nem a direção e normalização do score mostrado no leaderboard.

O score final será ajustado por um fator de uptime baseado na proporção de execuções bem-sucedidas. A fórmula exata também não foi publicada. Na prática, uma melhora pequena no RMSE não compensa uma arquitetura que falha em buscar dados ou excede limites de execução. Disponibilidade é parte do modelo competitivo.

Os wildcards poderão avaliar separadamente vento lento, vento rápido, CMEs, HSSs ou outros regimes. O desempenho geral, entretanto, é descrito como previsão do vento solar ambiente ao longo do período inteiro.

## 8. Requisitos de submissão

### Obrigatórios segundo o material atual

- container Docker totalmente implantável e compatível com o template fornecido;
- todo o código-fonte necessário;
- pesos/configurações exigidos pela licença final;
- dependências declaradas e reproduzíveis;
- README claro;
- execução automática por um único comando;
- busca automática das dependências de dados;
- uso apenas do que estava disponível no timestamp da inferência;
- funcionamento em CPU;
- aprovação na semana obrigatória de pré-deploy, de 1 a 10 de novembro;
- robustez a lacunas e latência.

### Ainda não publicado

- imagem-base e contrato exato do container;
- arquitetura e número de CPUs, memória, disco, GPU inexistente/indisponível e limite de tempo;
- acesso à internet, DNS, domínios liberados e política de credenciais;
- tamanho máximo da imagem e dos artefatos;
- diretórios de entrada/saída e esquema do arquivo de previsão;
- política de logs, códigos de saída, retries e timeout;
- determinismo, sementes e uso de aleatoriedade;
- possibilidade de atualizar imagem/pesos durante o pré-deploy;
- momento exato do code freeze e correções permitidas depois dele;
- política para indisponibilidade da infraestrutura da organização.

## 9. Análise competitiva

### 9.1 O que provavelmente decidirá o ranking

1. **Generalização temporal real.** O conjunto de avaliação está no futuro. Splits aleatórios e normalização calculada com todo o histórico produzem resultados enganosos.
2. **Desempenho no fim do horizonte.** A hora 72 terá peso elevado; otimizar apenas os primeiros passos pode piorar a métrica oficial.
3. **Cobertura de regimes.** Modelos excessivamente suaves podem ter RMSE médio razoável, mas perder HSSs e mudanças rápidas, prejudicando wildcards e possivelmente o score geral.
4. **Uptime.** APIs externas, staging, downloads volumosos e dependências frágeis podem reduzir diretamente o resultado final.
5. **Custo de inferência em CPU.** Arquiteturas visuais grandes precisam de redução, quantização, embeddings compactos ou um caminho de fallback.

### 9.2 Baselines mínimos

Antes de arquiteturas complexas, a equipe deve implementar e medir:

- média/climatologia por regime ou época;
- persistência recente;
- recorrência solar próxima de 27 dias;
- combinação linear ou gradient boosting de persistência, recorrência e features temporais;
- referência física WSA, se as regras permitirem usar a saída e sua disponibilidade operacional for suficiente.

Um baseline simples e extremamente confiável deve permanecer dentro do container como fallback. Uma execução válida com previsão menos precisa tende a ser preferível a uma falha total, dada a penalidade de uptime.

### 9.3 Arquitetura recomendada

Uma estratégia de risco controlado é um ensemble em camadas:

1. **Camada resiliente:** recorrência de 27 dias, climatologia robusta e últimas observações válidas.
2. **Modelo tabular:** árvores de boosting ou rede temporal compacta com lags, estatísticas móveis, tendência, fase de rotação solar e disponibilidade das fontes.
3. **Modelo solar:** encoder compacto de imagens EUV/magnetogramas para representar buracos coronais e a configuração do disco solar.
4. **Componente físico:** previsão WSA ou features derivadas de mapas coronais, se permitido.
5. **Gating/ensemble:** pesos dependentes da qualidade dos dados, do regime e do horizonte.

O componente visual deve provar ganho fora da amostra e não apenas em um split aleatório. Caso o download ou a inferência visual falhe, o ensemble deve degradar automaticamente para a camada tabular/resiliente.

### 9.4 Validação correta

- usar *walk-forward validation* com cortes cronológicos;
- manter buffers entre treino, validação e teste para evitar vazamento por janelas sobrepostas;
- reproduzir disparos a cada 6 horas e horizontes de 72 horas;
- registrar o “tempo de disponibilidade” dos dados, não apenas o timestamp físico da observação;
- ajustar preprocessamento somente no trecho de treino de cada fold;
- simular lacunas, dados atrasados, HTTP 429/5xx, timeouts e respostas malformadas;
- reportar RMSE por horizonte, regime e período, além do agregado;
- comparar com persistência, recorrência e uma referência física;
- executar a imagem Docker repetidamente em uma máquina CPU limpa.

### 9.5 Métricas locais sugeridas

Enquanto os pesos oficiais não forem publicados, acompanhar:

- RMSE geral e por horizonte: 1–24 h, 25–48 h e 49–72 h;
- RMSE específico em 72 h;
- MAE e correlação, apenas como diagnóstico;
- erro de pico e de tempo de chegada para HSSs;
- métricas separadas para vento lento/rápido e intervalos CME/HSS;
- taxa de execuções válidas, latência p50/p95/p99 e consumo máximo de memória;
- frequência de cada caminho de fallback.

Após o briefing, deve-se reproduzir exatamente a fórmula oficial em um avaliador local e usar o uptime no critério de seleção de modelos.

## 10. Plano de execução recomendado

### 24–31 de agosto

- concluir o pré-registro e definir equipe de até 4 pessoas;
- atribuir responsáveis por dados/ciência, modelagem, infraestrutura e validação;
- construir ingestão histórica do ACE e documentar proveniência/flags;
- implementar persistência e recorrência de 27 dias;
- criar um simulador *walk-forward* sem vazamento;
- testar acesso programático às fontes candidatas.

### 1–14 de setembro

- revisar o briefing técnico linha a linha;
- adaptar o repositório ao template Docker oficial;
- congelar a definição do rótulo e do protocolo local de avaliação;
- obter um baseline ponta a ponta executável por um comando;
- resolver com a organização todas as perguntas da seção 12.

### 15 de setembro–10 de outubro

- desenvolver modelos tabulares e temporais;
- avaliar recorrência, features coronais, WSA e imagens solares;
- medir ganho incremental sob validação cronológica;
- construir fallback e observabilidade.

### 11–25 de outubro

- selecionar/combinar modelos pelo score operacional esperado;
- testar lacunas, latência, rede lenta e indisponibilidade de fonte;
- reduzir tempo, memória e tamanho do container;
- auditar licenças, referências e ausência de segredos.

### 26 de outubro–1º de novembro

- fazer ensaio de code freeze;
- reproduzir a imagem a partir do zero;
- validar o comando único e o esquema de saída;
- finalizar README, inventário de dados/modelos e checksums.

### 1–10 de novembro

- usar a janela de pré-deploy para compatibilidade com o harness;
- não introduzir mudanças de modelagem sem nova validação completa;
- guardar logs e evidências de cada execução de qualificação;
- congelar a versão final conforme instrução da organização.

## 11. Estrutura sugerida do repositório

```text
solar-wind-hackathon/
├── README.md
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── configs/
│   └── inference.yaml
├── src/solar_wind/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── inference.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── operational/
├── artifacts/
│   ├── model.*
│   └── checksums.txt
├── scripts/
│   ├── train.py
│   ├── backtest.py
│   └── run_inference.py
└── docs/
    ├── data-provenance.md
    └── model-card.md
```

Essa estrutura é uma recomendação, não um requisito oficial. Ela deve ser ajustada ao template liberado pela organização.

## 12. Perguntas prioritárias para o briefing de 1º de setembro

1. Qual produto ACE/SWEPAM, versão, campo, unidade, flags e latência compõem o alvo oficial?
2. A saída contém 72 pontos ou inclui ambos os extremos? Quais timestamps e timezone são exigidos?
3. Como previsões sobrepostas, emitidas a cada 6 horas, são agregadas no score?
4. Qual é a fórmula completa do RMSE ponderado e quais são os pesos por horizonte?
5. Como o uptime entra no score e o que caracteriza uma execução bem-sucedida?
6. Quais são CPU, memória, disco, timeout, tamanho da imagem e política de concorrência?
7. O container terá internet aberta? Quais domínios, protocolos, quotas e credenciais são permitidos?
8. Quem executa o *forward-fill* e quais limites/flags devem ser respeitados?
9. Qual é o prazo efetivo: 1º ou 10 de novembro? O que pode mudar durante o pré-deploy?
10. Qual é a data final da avaliação, considerando “novembro–janeiro” e “três meses completos”?
11. Wildcards são adicionais aos US$ 20.000? Quais critérios e valores serão usados?
12. Qual é o escopo exato da licença MIT para código, configurações, pesos e artefatos?
13. Quais afiliações com CU Boulder tornam uma pessoa inelegível?
14. Quais baselines serão fornecidos e quais modelos/serviços externos são aceitos?
15. Como falhas da própria infraestrutura ou indisponibilidade geral de uma fonte afetam uptime?

## 13. Checklist de prontidão

- [ ] Pré-registro concluído por todos os procedimentos exigidos.
- [ ] Elegibilidade e afiliações confirmadas.
- [ ] Produto-alvo ACE confirmado contra a especificação final.
- [ ] Pipeline sem vazamento temporal auditado.
- [ ] Baselines de persistência e recorrência implementados.
- [ ] Avaliador local idêntico ao oficial.
- [ ] Inferência completa em CPU dentro dos limites.
- [ ] Container reproduzível e executável por um comando.
- [ ] Nenhum segredo ou dado privado dentro da imagem.
- [ ] Downloads possuem timeout, retry limitado, validação e fallback.
- [ ] Lacunas e respostas corrompidas são testadas.
- [ ] Uma falha de modalidade não derruba toda a previsão.
- [ ] Logs permitem diagnosticar fonte, versão, latência e fallback.
- [ ] Licenças e citações de dados/modelos estão documentadas.
- [ ] README e model card estão completos.
- [ ] Imagem final possui versão, checksum e tag imutável.
- [ ] Ensaio de 72 horas ou mais sem falha concluído antes do pré-deploy.

## 14. Organização e contatos

O BenchLab é uma iniciativa da Trillium Technologies/Frontier Development Lab. A competição é apresentada em parceria com Queen Mary University of London e CU Boulder, com apoio do NASA Grant `80NSSC25K7178`. O anúncio à comunidade também descreve o BenchLab como iniciativa do FDL em parceria com CU Boulder, Queen Mary University of London e NASA.

Contatos publicados no one-pager:

- Enrico Camporeale: `enrico.camporeale@colorado.edu`
- Mike Heyns: `mike.heyns@trillium.tech`

## 15. Fontes

### Fontes oficiais do torneio

1. [Página oficial — Solar Wind Prediction 2026](https://fdl.ai/benchlab-solar-wind-prediction-2026)
2. [One-pager / convite da sessão](https://docs.google.com/document/d/1meZTeEyrPd_o0MPo37X8JYVY_udQWs7IzRlPui6paOg/edit?usp=sharing)
3. [Formulário de pré-registro](https://docs.google.com/forms/d/e/1FAIpQLSetVWZyGT_FDZ473j-f-ak5kEd608A1CObXkfudwiKmxSrM2g/viewform)
4. [Visão geral do FDL BenchLab](https://fdl.ai/benchlab)
5. [SWx-TREC](https://swx-trec.com/)
6. [Anúncio à comunidade — UK Solar Physics](https://uksolphys.org/general-news/live-solar-wind-prediction-tournament-and-prize-fund/)

### Referências técnicas para preparação

7. [NASA Open Data — ACE/SWEPAM 64-second Level 2](https://data.nasa.gov/dataset/ace-solar-wind-electron-proton-alpha-monitor-swepam-64-second-level-2-data)
8. [NOAA SWPC — diretório JSON horário do ACE/SWEPAM](https://services.swpc.noaa.gov/json/ace/swepam/)
9. [NASA SDO — formas de acesso a dados e produtos NRT](https://sdo.gsfc.nasa.gov/data/dataaccess.php)
10. [NASA CCMC — WSA Dashboard e API](https://ccmc.gsfc.nasa.gov/community-tools/WSA-Dashboard/)
11. [NASA CCMC — SIML-HSS](https://ccmc.gsfc.nasa.gov/models/SIML-HSS~1/)
12. [NASA CCMC — AmbSoWi-ML](https://ccmc.gsfc.nasa.gov/models/AmbSoWi-ML~1/)
13. [FDL 2025 — Decoding Solar Wind Structure](https://fdl.ai/heliolab25)
14. [SuryaBench — artigo e dados de benchmark](https://www.nature.com/articles/s41597-026-06552-5)

As referências 7–14 são recomendações técnicas e não definem as regras do torneio. Em caso de conflito, prevalecem a página, o briefing e as instruções operacionais publicadas pela organização.

## Conclusão

A competição premia um sistema de previsão operacional, não apenas um modelo. A proposta mais forte será a que combinar validação temporal rigorosa, bom desempenho no horizonte de 72 horas, representação útil do estado solar, baselines físicos/recorrentes e uma camada de engenharia capaz de produzir previsões válidas mesmo com dados incompletos. Até 01/09/2026, o trabalho de maior retorno é montar a equipe, concluir o pré-registro, estabelecer um baseline sem vazamento e preparar uma arquitetura de ingestão resiliente. A especificação final deve então substituir todas as suposições marcadas neste documento.
