# Modelo com física: buracos coronais + propagação HUX (proposta, não implementada)

Status: **ideia registrada em 04/11/2026 para retomar se necessário**. Nada disto está no
código nem na submissão.

## Por que não um PINN "puro"

Um PINN (*physics-informed neural network*; Raissi et al., 2019) treina uma rede para
satisfazer uma equação diferencial em uma solução específica. Isso não ajuda aqui por três
motivos:

- **Custo:** cada emissão (a cada 6 h) exigiria treinar de novo, o que é lento e instável.
- **Não há o que ganhar na propagação:** a parte que tem equação (o vento indo do Sol até
  L1) já tem solução numérica barata e exata.
- **O erro está em outro lugar:** o grosso do erro dos modelos físicos vem da condição de
  contorno, isto é, da velocidade com que o vento sai do Sol. Uma restrição de equação na
  função de perda não resolve isso.

A alternativa recomendada é **física diferenciável**: a física entra como uma camada fixa
dentro do modelo, e só uns poucos parâmetros são ajustados aos dados.

## A física

1. **Saída do Sol (~0,1 UA).** Vento rápido sai dos buracos coronais. Há duas formas de
   estimar a velocidade de saída:
   - relação empírica com a área de buraco coronal, por exemplo `v ≈ 350 + 800·A` km/s
     (Vršnak, Temmer & Veronig, 2007);
   - WSA, a partir do campo magnético (Arge & Pizzo, 2000), que exige magnetogramas.
2. **Rotação.** O Sol gira cerca de 13,2° por dia (período sinódico de ~27,3 dias). Uma
   estrutura a leste do meridiano central chega a L1 mais tarde. É o atraso de ~1 dia por
   faixa de 15° visto em `docs/hybrid.md`.
3. **Propagação até 1 UA com HUX** (*Heliospheric Upwind eXtrapolation*; Riley & Lionello,
   2011). É uma equação do tipo Burgers em raio e longitude:
   `∂v/∂r = (Ω_rot / v) · ∂v/∂φ`.
   Ela reproduz o vento rápido alcançando o lento e a compressão nas regiões de interação.
   Resolve-se com diferenças finitas em milissegundos. Há implementação aberta em Python
   (HUXt; Owens et al., 2020).

## Desenho proposto

```
frações de buraco coronal por longitude (já calculadas, AIA 193 Å)
   └─► camada de saída: v0(φ) = v_min + (v_max − v_min) · f(A(φ); θ)     [poucos parâmetros]
          └─► HUX diferenciável de 0,1 UA a 1 UA, com a rotação solar
                 └─► série de 72 h em L1 (previsão física)
                        └─► nova feature do LightGBM (stacking), ou modelo isolado
```

- **Entrada:** as frações de buraco coronal por faixa de longitude, que já existem em
  `data/features/coronal_holes_0193_6h.parquet` (`ch_t25/t35_lat60_lon…`, de −60° a +30°).
  - Para cobrir a longitude toda, junte as imagens das últimas emissões, rotacionadas para
    a mesma referência, montando um mapa de Carrington aproximado.
  - Longitudes sem observação recebem a velocidade lenta (`v_min`).
- **Parâmetros a ajustar:** `v_min`, `v_max`, a forma de `f`, o limiar de área e, se
  preciso, uma aceleração residual além de 0,1 UA. Também pode entrar um deslocamento de
  longitude que absorva o erro de mapeamento.
- **Treino:** de ponta a ponta contra a velocidade do ACE, com os mesmos cortes do híbrido
  (treino 2011-02 a 2021-12, validação 2022–2023, teste 2024-01 a 2026-10 avaliado uma
  vez). Usar PyTorch, que já é extra do projeto (`--extra lstm`), com o HUX escrito em
  tensores.
- **Corte de dados:** usar só imagens até `t0 − 1 h`, como em `build_pairs`
  (`src/benchlab_forecasting/hybrid.py`).
- **Integração:** acrescentar a previsão física por hora de antecedência como coluna em
  `build_pairs` e retreinar o LightGBM ajustado. No container, o HUX roda em CPU em
  milissegundos e não precisa de nenhum dado novo.

## Opção mais completa (WSA)

Trocar a camada empírica pelo WSA com magnetogramas em tempo real, de GONG (NSO) ou HMI
(JSOC). É mais fiel à física, mas exige:

- um pipeline novo de dados: mapas sinópticos, extrapolação PFSS e fator de expansão;
- domínios novos no `submission.json`;
- validação de que o produto em tempo real equivale ao de arquivo.

## Expectativa

- **Sozinho:** modelos empíricos de buraco coronal com HUX costumam errar 90–100 km/s em
  3–4 dias. É pior que o híbrido atual, que erra 87 km/s em 49–72 h (ver o benchmark no
  `README.md`).
- **Como feature do boosting:** ganho provável de **1 a 3 km/s** nas antecedências longas,
  concentrado nas chegadas de correntes rápidas. Não é um salto.
- **Valor além do RMSE:**
  - interpretabilidade, com velocidade de saída e tempo de trânsito explícitos;
  - comportamento mais físico em regimes pouco vistos no treino.

## Critérios para adotar

1. O ganho de RMSE no teste deve ser positivo, com IC 95% do bootstrap em blocos de 27 dias
   acima de zero (`validation.py`).
2. O RMSE em 49–72 h não pode piorar.
3. O tempo do container deve continuar bem abaixo dos 15 minutos.
4. A paridade entre o pipeline ao vivo e o offline deve ser verificada, como foi feito
   para a submissão.
5. **Nunca trocar a submissão perto do prazo:** a nova versão precisa passar de novo pelo
   dry run até ficar ELIGIBLE.

## Referências

- Arge, C. N. & Pizzo, V. J. (2000). *Improvement in the prediction of solar wind
  conditions using near-real time solar magnetic field updates.* JGR 105(A5), 10465.
- Owens, M. et al. (2020). *A computationally efficient, time-dependent model of the solar
  wind for use as a surrogate to three-dimensional numerical magnetohydrodynamic
  simulations* (HUXt). Solar Physics 295, 43.
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed neural
  networks.* Journal of Computational Physics 378, 686.
- Riley, P. & Lionello, R. (2011). *Mapping solar wind streams from the Sun to 1 AU: a
  comparison of techniques.* Solar Physics 270, 575.
- Vršnak, B., Temmer, M. & Veronig, A. M. (2007). *Coronal holes and solar wind high-speed
  streams: I. Forecasting the solar wind parameters.* Solar Physics 240, 315.
