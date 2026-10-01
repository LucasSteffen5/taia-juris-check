# Relatório da execução ft-20260930-v2-balanced-campaign

## Resultado e seleção

O checkpoint de seed 42 selecionado é `s3_1_7e4cf94f_o00_w` (`/runs/ft-20260930-v2-balanced-campaign/candidates/s3_1_7e4cf94f_o00_w`). Seu score ponderado na validação local foi **0.6754**, contra **0.6363** do checkpoint original, ganho de **+0.0391** (+6.14%). O score N2 foi 0.6379, contra 0.5979 no baseline. Os valores `tau` não aumentaram em nenhum nível; a taxa `inventada → real` não aumentou. As configurações foram ordenadas pela média entre seeds, com desempate por score N2 e precisão de spans; a elegibilidade foi exigida para todas as seeds. Elegível para uso explícito: **true**. O pipeline padrão permanece no checkpoint original.

| Rodada | Score final | Score N2 | Precisão de spans | Elegível |
|---|---:|---:|---:|:---:|
| original_lr_5e-06 | 0.6523 | 0.6124 | 0.8237 | não |
| original_lr_1e-05 | 0.6547 | 0.6213 | 0.8204 | sim |
| original_lr_2e-05 | 0.6582 | 0.6216 | 0.8418 | sim |
| original_lr_3e-05 | 0.6571 | 0.6196 | 0.8337 | sim |
| v1_lr_5e-06 | 0.6582 | 0.6214 | 0.8086 | sim |
| v1_lr_1e-05 | 0.6611 | 0.6246 | 0.8136 | sim |
| v1_lr_2e-05 | 0.6593 | 0.6253 | 0.8106 | sim |
| v1_lr_3e-05 | 0.6611 | 0.6226 | 0.8325 | sim |
| s2_1_a4bafc99_a00_r1 | 0.6611 | 0.6246 | 0.8319 | sim |
| s2_1_a4bafc99_a00_r3 | 0.6519 | 0.6133 | 0.8343 | não |
| s2_1_a4bafc99_a00_r5 | 0.6605 | 0.6229 | 0.8420 | sim |
| s2_1_a4bafc99_a10_r1 | 0.6592 | 0.6216 | 0.8262 | sim |
| s2_1_a4bafc99_a10_r3 | 0.6649 | 0.6289 | 0.8397 | sim |
| s2_1_a4bafc99_a10_r5 | 0.6533 | 0.6168 | 0.8244 | não |
| s2_1_a4bafc99_a25_r1 | 0.6529 | 0.6161 | 0.8217 | não |
| s2_1_a4bafc99_a25_r3 | 0.6608 | 0.6229 | 0.8361 | sim |
| s2_1_a4bafc99_a25_r5 | 0.6539 | 0.6163 | 0.8378 | sim |
| s2_2_807e8395_a00_r1 | 0.6626 | 0.6244 | 0.8210 | sim |
| s2_2_807e8395_a00_r3 | 0.6578 | 0.6212 | 0.8233 | sim |
| s2_2_807e8395_a00_r5 | 0.6617 | 0.6246 | 0.8154 | sim |
| s2_2_807e8395_a10_r1 | 0.6625 | 0.6263 | 0.8321 | sim |
| s2_2_807e8395_a10_r3 | 0.6580 | 0.6222 | 0.8253 | sim |
| s2_2_807e8395_a10_r5 | 0.6616 | 0.6252 | 0.8141 | sim |
| s2_2_807e8395_a25_r1 | 0.6631 | 0.6261 | 0.8205 | sim |
| s2_2_807e8395_a25_r3 | 0.6592 | 0.6242 | 0.8222 | sim |
| s2_2_807e8395_a25_r5 | 0.6616 | 0.6249 | 0.8128 | sim |
| s3_1_7e4cf94f_o00_w | 0.6754 | 0.6379 | 0.8481 | sim |
| s3_1_7e4cf94f_o00_c | 0.6576 | 0.6180 | 0.8293 | sim |
| s3_1_7e4cf94f_o01_w | 0.6601 | 0.6229 | 0.8402 | sim |
| s3_1_7e4cf94f_o01_c | 0.6579 | 0.6201 | 0.8106 | sim |
| s3_1_7e4cf94f_o03_w | 0.6557 | 0.6166 | 0.8488 | sim |
| s3_1_7e4cf94f_o03_c | 0.6591 | 0.6250 | 0.8384 | sim |
| s3_2_f42dbbb6_o00_w | 0.6593 | 0.6226 | 0.8221 | sim |
| s3_2_f42dbbb6_o00_c | 0.6590 | 0.6215 | 0.8311 | sim |
| s3_2_f42dbbb6_o01_w | 0.6640 | 0.6253 | 0.8348 | sim |
| s3_2_f42dbbb6_o01_c | 0.6586 | 0.6226 | 0.8293 | sim |
| s3_2_f42dbbb6_o03_w | 0.6582 | 0.6218 | 0.8196 | sim |
| s3_2_f42dbbb6_o03_c | 0.6582 | 0.6216 | 0.8222 | sim |
| s4_1_52b24416_seed17 | 0.6619 | 0.6243 | 0.8397 | sim |
| s4_1_52b24416_seed2026 | 0.6639 | 0.6243 | 0.8418 | sim |
| s4_2_b2f6dcdc_seed17 | 0.6601 | 0.6237 | 0.8251 | sim |
| s4_2_b2f6dcdc_seed2026 | 0.6587 | 0.6226 | 0.8346 | sim |

## Estabilidade entre seeds

A seleção de configuração usa a média do score entre seeds; o pacote exporta o checkpoint da seed 42. O desvio padrão abaixo é populacional.

| Configuração | Seeds | Score médio | Desvio padrão | N2 médio | Precisão spans média | Todos elegíveis |
|---|---|---:|---:|---:|---:|:---:|
| `original_5e-06_a0.25_r5_o0.0_window` | 42 | 0.6523 | 0.0000 | 0.6124 | 0.8237 | não |
| `original_1e-05_a0.25_r5_o0.0_window` | 42 | 0.6547 | 0.0000 | 0.6213 | 0.8204 | sim |
| `original_2e-05_a0.25_r5_o0.0_window` | 42 | 0.6582 | 0.0000 | 0.6216 | 0.8418 | sim |
| `original_3e-05_a0.25_r5_o0.0_window` | 42 | 0.6571 | 0.0000 | 0.6196 | 0.8337 | sim |
| `v1_5e-06_a0.25_r5_o0.0_window` | 42 | 0.6582 | 0.0000 | 0.6214 | 0.8086 | sim |
| `v1_1e-05_a0.25_r5_o0.0_window` | 42 | 0.6611 | 0.0000 | 0.6246 | 0.8136 | sim |
| `v1_2e-05_a0.25_r5_o0.0_window` | 42 | 0.6593 | 0.0000 | 0.6253 | 0.8106 | sim |
| `v1_3e-05_a0.25_r5_o0.0_window` | 42 | 0.6611 | 0.0000 | 0.6226 | 0.8325 | sim |
| `a4bafc99_a00_r1` | 42 | 0.6611 | 0.0000 | 0.6246 | 0.8319 | sim |
| `a4bafc99_a00_r3` | 42 | 0.6519 | 0.0000 | 0.6133 | 0.8343 | não |
| `a4bafc99_a00_r5` | 42 | 0.6605 | 0.0000 | 0.6229 | 0.8420 | sim |
| `a4bafc99_a10_r1` | 42 | 0.6592 | 0.0000 | 0.6216 | 0.8262 | sim |
| `a4bafc99_a10_r3` | 42 | 0.6649 | 0.0000 | 0.6289 | 0.8397 | sim |
| `a4bafc99_a10_r5` | 42 | 0.6533 | 0.0000 | 0.6168 | 0.8244 | não |
| `a4bafc99_a25_r1` | 42 | 0.6529 | 0.0000 | 0.6161 | 0.8217 | não |
| `a4bafc99_a25_r3` | 42 | 0.6608 | 0.0000 | 0.6229 | 0.8361 | sim |
| `a4bafc99_a25_r5` | 42 | 0.6539 | 0.0000 | 0.6163 | 0.8378 | sim |
| `807e8395_a00_r1` | 42 | 0.6626 | 0.0000 | 0.6244 | 0.8210 | sim |
| `807e8395_a00_r3` | 42 | 0.6578 | 0.0000 | 0.6212 | 0.8233 | sim |
| `807e8395_a00_r5` | 42 | 0.6617 | 0.0000 | 0.6246 | 0.8154 | sim |
| `807e8395_a10_r1` | 42 | 0.6625 | 0.0000 | 0.6263 | 0.8321 | sim |
| `807e8395_a10_r3` | 42 | 0.6580 | 0.0000 | 0.6222 | 0.8253 | sim |
| `807e8395_a10_r5` | 42 | 0.6616 | 0.0000 | 0.6252 | 0.8141 | sim |
| `807e8395_a25_r1` | 42 | 0.6631 | 0.0000 | 0.6261 | 0.8205 | sim |
| `807e8395_a25_r3` | 42 | 0.6592 | 0.0000 | 0.6242 | 0.8222 | sim |
| `807e8395_a25_r5` | 42 | 0.6616 | 0.0000 | 0.6249 | 0.8128 | sim |
| `a4bafc99_a10_r3_o00_window` | 42, 17, 2026 | 0.6671 | 0.0059 | 0.6289 | 0.8432 | sim |
| `a4bafc99_a10_r3_o00_context` | 42 | 0.6576 | 0.0000 | 0.6180 | 0.8293 | sim |
| `a4bafc99_a10_r3_o01_window` | 42 | 0.6601 | 0.0000 | 0.6229 | 0.8402 | sim |
| `a4bafc99_a10_r3_o01_context` | 42 | 0.6579 | 0.0000 | 0.6201 | 0.8106 | sim |
| `a4bafc99_a10_r3_o03_window` | 42 | 0.6557 | 0.0000 | 0.6166 | 0.8488 | sim |
| `a4bafc99_a10_r3_o03_context` | 42 | 0.6591 | 0.0000 | 0.6250 | 0.8384 | sim |
| `807e8395_a25_r1_o00_window` | 42 | 0.6593 | 0.0000 | 0.6226 | 0.8221 | sim |
| `807e8395_a25_r1_o00_context` | 42 | 0.6590 | 0.0000 | 0.6215 | 0.8311 | sim |
| `807e8395_a25_r1_o01_window` | 42, 17, 2026 | 0.6609 | 0.0022 | 0.6238 | 0.8315 | sim |
| `807e8395_a25_r1_o01_context` | 42 | 0.6586 | 0.0000 | 0.6226 | 0.8293 | sim |
| `807e8395_a25_r1_o03_window` | 42 | 0.6582 | 0.0000 | 0.6218 | 0.8196 | sim |
| `807e8395_a25_r1_o03_context` | 42 | 0.6582 | 0.0000 | 0.6216 | 0.8222 | sim |


## Métricas da classificação por nível

F1 por classe (`real`, `inventada`, `incompleta`) e score local por nível:

| Conjunto | Nível | Score | Macro F1 | Real | Inventada | Incompleta |
|---|:---:|---:|---:|---:|---:|---:|
| baseline · validação local | N1 | 0.7131 | 0.6621 | 0.8772 | 0.5664 | 0.5427 |
| baseline · validação local | N2 | 0.5979 | 0.5612 | 0.6383 | 0.5074 | 0.5379 |
| selecionado · validação local | N1 | 0.7505 | 0.6969 | 0.8772 | 0.6119 | 0.6016 |
| selecionado · validação local | N2 | 0.6379 | 0.5987 | 0.6667 | 0.5263 | 0.6032 |
| selecionado · teste local | N1 | 0.7910 | 0.7328 | 0.8435 | 0.7227 | 0.6322 |
| selecionado · teste local | N2 | 0.6088 | 0.5674 | 0.5283 | 0.5641 | 0.6099 |

## Auditoria de spans

Precisão/recall usam IoU 0,5; a versão tipada exige também o tipo correto. A tabela separa regex, NER bruto, fusão e resultado após resolução, e conta falsos positivos em documentos sem citações.

| Conjunto | Etapa | Precisão | Recall | Precisão tipada | Recall tipada | Acertos/ouro | FP docs negativos |
|---|---|---:|---:|---:|---:|---:|---:|
| validação local | fused | 0.8656 | 0.6982 | 0.2543 | 0.2051 | 953/1365 | 0 |
| validação local | fused · N1 | 0.8653 | 0.6857 | 0.2546 | 0.2018 | 469/684 | — |
| validação local | fused · N2 | 0.8658 | 0.7107 | 0.2540 | 0.2085 | 484/681 | — |
| validação local | fused · jurisprudencia | 0.7018 | 0.4904 | — | — | 280/571 | — |
| validação local | fused · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |
| validação local | ner | 0.8481 | 0.7077 | 0.2511 | 0.2095 | 966/1365 | 0 |
| validação local | ner · N1 | 0.8479 | 0.6930 | 0.2469 | 0.2018 | 474/684 | — |
| validação local | ner · N2 | 0.8483 | 0.7225 | 0.2552 | 0.2173 | 492/681 | — |
| validação local | ner · jurisprudencia | 0.6651 | 0.5009 | — | — | 286/571 | — |
| validação local | ner · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |
| validação local | regex | 0.9314 | 0.1392 | 0.5049 | 0.0755 | 190/1365 | 0 |
| validação local | regex · N1 | 0.9370 | 0.1740 | 0.4094 | 0.0760 | 119/684 | — |
| validação local | regex · N2 | 0.9221 | 0.1043 | 0.6623 | 0.0749 | 71/681 | — |
| validação local | regex · jurisprudencia | 0.8583 | 0.1804 | — | — | 103/571 | — |
| validação local | regex · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |
| validação local | resolved | 0.8656 | 0.6982 | 0.2543 | 0.2051 | 953/1365 | 0 |
| validação local | resolved · N1 | 0.8653 | 0.6857 | 0.2546 | 0.2018 | 469/684 | — |
| validação local | resolved · N2 | 0.8658 | 0.7107 | 0.2540 | 0.2085 | 484/681 | — |
| validação local | resolved · jurisprudencia | 0.7018 | 0.4904 | — | — | 280/571 | — |
| validação local | resolved · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |
| teste local | fused | 0.8818 | 0.8304 | 0.3203 | 0.3016 | 1082/1303 | 0 |
| teste local | fused · N1 | 0.8856 | 0.8300 | 0.3268 | 0.3063 | 542/653 | — |
| teste local | fused · N2 | 0.8780 | 0.8308 | 0.3138 | 0.2969 | 540/650 | — |
| teste local | fused · jurisprudencia | 0.7602 | 0.7264 | — | — | 393/541 | — |
| teste local | fused · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |
| teste local | ner | 0.8554 | 0.8081 | 0.2900 | 0.2740 | 1053/1303 | 0 |
| teste local | ner · N1 | 0.8459 | 0.8070 | 0.2905 | 0.2772 | 527/653 | — |
| teste local | ner · N2 | 0.8651 | 0.8092 | 0.2895 | 0.2708 | 526/650 | — |
| teste local | ner · jurisprudencia | 0.7083 | 0.6599 | — | — | 357/541 | — |
| teste local | ner · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |
| teste local | regex | 0.8530 | 0.2717 | 0.4747 | 0.1512 | 354/1303 | 0 |
| teste local | regex · N1 | 0.8846 | 0.3522 | 0.3885 | 0.1547 | 230/653 | — |
| teste local | regex · N2 | 0.8000 | 0.1908 | 0.6194 | 0.1477 | 124/650 | — |
| teste local | regex · jurisprudencia | 0.7695 | 0.3641 | — | — | 197/541 | — |
| teste local | regex · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |
| teste local | resolved | 0.8818 | 0.8304 | 0.3203 | 0.3016 | 1082/1303 | 0 |
| teste local | resolved · N1 | 0.8856 | 0.8300 | 0.3268 | 0.3063 | 542/653 | — |
| teste local | resolved · N2 | 0.8780 | 0.8308 | 0.3138 | 0.2969 | 540/650 | — |
| teste local | resolved · jurisprudencia | 0.7602 | 0.7264 | — | — | 393/541 | — |
| teste local | resolved · legislacao | 0.0000 | 0.0000 | — | — | 0/0 | — |

## Dados e recursos

- Dados locais: 412 documentos de treino, 62 de validação e 56 de teste. Os dez documentos oficiais autorizados entraram somente no treino; 16 ficaram reservados para score final quando solicitado.
- GPU: NVIDIA GeForce RTX 3060; pico alocado: 2.06 GiB; pico reservado: 2.61 GiB.
- Batch por dispositivo: 16; acumulação: 1; batch efetivo: 16; FP16: True; gradient checkpointing: True.
- Rodada selecionada: LR 0.0000300, 720 passos, época 4.50, loss treino 0.0035 e loss de validação 0.0796; runtime de treino 911.5 s.
- Tempo contabilizado em preparação e treino: 33837.4 s, sem limite fixo de GPU.

| Etapa | Segundos |
|---|---:|
| `fine_tuning_cv_1_52b24416_fold0` | 526.4 |
| `fine_tuning_cv_1_52b24416_fold1` | 574.5 |
| `fine_tuning_cv_1_52b24416_fold2` | 729.3 |
| `fine_tuning_cv_2_b2f6dcdc_fold0` | 620.5 |
| `fine_tuning_cv_2_b2f6dcdc_fold1` | 635.0 |
| `fine_tuning_cv_2_b2f6dcdc_fold2` | 626.0 |
| `fine_tuning_original_lr_1e-05` | 950.4 |
| `fine_tuning_original_lr_2e-05` | 634.1 |
| `fine_tuning_original_lr_3e-05` | 748.0 |
| `fine_tuning_original_lr_5e-06` | 946.7 |
| `fine_tuning_s2_1_a4bafc99_a00_r1` | 623.1 |
| `fine_tuning_s2_1_a4bafc99_a00_r3` | 572.3 |
| `fine_tuning_s2_1_a4bafc99_a00_r5` | 580.5 |
| `fine_tuning_s2_1_a4bafc99_a10_r1` | 632.1 |
| `fine_tuning_s2_1_a4bafc99_a10_r3` | 583.4 |
| `fine_tuning_s2_1_a4bafc99_a10_r5` | 843.8 |
| `fine_tuning_s2_1_a4bafc99_a25_r1` | 674.7 |
| `fine_tuning_s2_1_a4bafc99_a25_r3` | 933.9 |
| `fine_tuning_s2_1_a4bafc99_a25_r5` | 562.7 |
| `fine_tuning_s2_2_807e8395_a00_r1` | 559.8 |
| `fine_tuning_s2_2_807e8395_a00_r3` | 581.8 |
| `fine_tuning_s2_2_807e8395_a00_r5` | 569.9 |
| `fine_tuning_s2_2_807e8395_a10_r1` | 567.7 |
| `fine_tuning_s2_2_807e8395_a10_r3` | 564.4 |
| `fine_tuning_s2_2_807e8395_a10_r5` | 618.5 |
| `fine_tuning_s2_2_807e8395_a25_r1` | 561.9 |
| `fine_tuning_s2_2_807e8395_a25_r3` | 564.0 |
| `fine_tuning_s2_2_807e8395_a25_r5` | 612.5 |
| `fine_tuning_s3_1_7e4cf94f_o00_c` | 711.2 |
| `fine_tuning_s3_1_7e4cf94f_o00_w` | 911.5 |
| `fine_tuning_s3_1_7e4cf94f_o01_c` | 906.1 |
| `fine_tuning_s3_1_7e4cf94f_o01_w` | 846.4 |
| `fine_tuning_s3_1_7e4cf94f_o03_c` | 1012.7 |
| `fine_tuning_s3_1_7e4cf94f_o03_w` | 1014.0 |
| `fine_tuning_s3_2_f42dbbb6_o00_c` | 709.9 |
| `fine_tuning_s3_2_f42dbbb6_o00_w` | 565.1 |
| `fine_tuning_s3_2_f42dbbb6_o01_c` | 842.7 |
| `fine_tuning_s3_2_f42dbbb6_o01_w` | 668.9 |
| `fine_tuning_s3_2_f42dbbb6_o03_c` | 1091.2 |
| `fine_tuning_s3_2_f42dbbb6_o03_w` | 789.9 |
| `fine_tuning_s4_1_52b24416_seed17` | 560.2 |
| `fine_tuning_s4_1_52b24416_seed2026` | 1015.1 |
| `fine_tuning_s4_2_b2f6dcdc_seed17` | 670.0 |
| `fine_tuning_s4_2_b2f6dcdc_seed2026` | 668.6 |
| `fine_tuning_v1_lr_1e-05` | 626.1 |
| `fine_tuning_v1_lr_2e-05` | 627.1 |
| `fine_tuning_v1_lr_3e-05` | 580.2 |
| `fine_tuning_v1_lr_5e-06` | 759.0 |
| `prepare_and_teacher_inference` | 63.4 |


## Validação interna por contexto

Os grupos do treino local foram divididos em três folds disjuntos. Esses resultados são diagnósticos e não foram usados para escolher o checkpoint.

| Configuração | Score médio | Desvio padrão |
|---|---:|---:|
| `807e8395_a25_r1_o01_window` | 0.6857 | 0.0364 |
| `a4bafc99_a10_r3_o00_window` | 0.6874 | 0.0344 |


## Incerteza descritiva por contexto

Bootstrap por grupos de contexto: 16 grupos com citações; 1 grupos sem citações foram excluídos apenas desta média diagnóstica. Média dos scores por contexto: 0.6507; intervalo bootstrap de 95%: 0.5629–0.7370 (1000 réplicas, seed 42). O score oficial/local do conjunto completo continua sendo a métrica principal.

## Teste local e procedência

O score final local do checkpoint selecionado, avaliado uma vez após a seleção, foi **0.6695** (N1 0.7910; N2 0.6088). Este teste local não é score oficial do desafio.

Hash dos pesos selecionados: `52b24416c23a645db9aa49e83cf59e14aa8b225ae396f01d48415d50ae67945c`. Hiperparâmetros, defaults resolvidos, tokenizer, arquitetura, ambiente e configurações efetivas estão em `training_config.resolved.json`; hashes e métricas estruturadas estão em `model_report.json` e `provenance.json`.

## Avaliação oficial final

Após congelar a seleção local, original, v1 e v2 foram avaliados nos mesmos 16 documentos oficiais reservados. As 119 anotações conferiram com os textos normalizados para LF antes da inferência. A fórmula oficial inclui bônus de calibração e pode resultar em score acima de 1.

| Modelo | Score oficial | N1 | N2 |
|---|---:|---:|---:|
| original | 1.082503 | 1.093750 | 1.076880 |
| v1 | 1.082703 | 1.093855 | 1.077127 |
| v2 | 1.082643 | 1.093820 | 1.077055 |

O maior score observado foi o do **v1**. A diferença v2 − v1 foi **-0.000060**. Os três modelos tiveram F1 por classe idêntico neste subconjunto; a diferença do score decorreu do bônus de calibração. Troca automática no pipeline da banca recomendada: **não**. Esse resultado não foi usado para escolher outro checkpoint nem ajustar hiperparâmetros. Métricas completas, IDs, hashes, análise de erros e auditoria de duplicatas estão em `official_evaluation.json` e `official_comparison.json`.

A tentativa anterior com cópias CRLF foi invalidada. `evaluation_correction.json` registra o motivo e os hashes; esses scores não participaram da comparação final.

