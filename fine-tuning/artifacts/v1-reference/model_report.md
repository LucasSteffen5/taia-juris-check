# Relatório da execução ft-20260930-prod2

## Resultado e seleção

O checkpoint selecionado é `lr_3e-5` (`/runs/ft-20260930-prod2/candidates/lr_3e-5`). Seu score ponderado na validação local foi **0.6702**, contra **0.6495** do checkpoint original, ganho de **+0.0207** (+3.19%). O score N2 foi 0.6407, contra 0.6199 no baseline. Os valores `tau` não aumentaram em nenhum nível; a taxa `inventada → real` não aumentou. A rodada selecionada teve o maior score final entre os candidatos elegíveis; o desempate por N2 não foi necessário. Elegível para uso explícito: **true**. O pipeline padrão permanece no checkpoint original.

| Rodada | Score final | Score N2 | Precisão de spans | Elegível |
|---|---:|---:|---:|:---:|
| lr_2e-5 | 0.6674 | 0.6372 | 0.8533 | sim |
| lr_1e-5 | 0.6675 | 0.6387 | 0.8616 | sim |
| lr_3e-5 | 0.6702 | 0.6407 | 0.8556 | sim |

## Métricas da classificação por nível

F1 por classe (`real`, `inventada`, `incompleta`) e score local por nível:

| Conjunto | Nível | Score | Macro F1 | Real | Inventada | Incompleta |
|---|:---:|---:|---:|---:|---:|---:|
| baseline · validação local | N1 | 0.7088 | 0.6578 | 0.8451 | 0.5863 | 0.5421 |
| baseline · validação local | N2 | 0.6199 | 0.5811 | 0.6885 | 0.5208 | 0.5341 |
| selecionado · validação local | N1 | 0.7293 | 0.6772 | 0.8611 | 0.5617 | 0.6088 |
| selecionado · validação local | N2 | 0.6407 | 0.6008 | 0.7097 | 0.4800 | 0.6128 |
| selecionado · teste local | N1 | 0.7987 | 0.7399 | 0.8535 | 0.7237 | 0.6424 |
| selecionado · teste local | N2 | 0.6213 | 0.5789 | 0.5565 | 0.5619 | 0.6183 |

## Auditoria de spans

Precisão/recall usam IoU 0,5; a versão tipada exige também o tipo correto. `acertos/ouro` informa spans correspondidos sobre os spans anotados. `—` não se aplica aos campos de precisão tipada por tipo.

| Conjunto | Nível/tipo | Precisão | Recall | Precisão tipada | Recall tipada | Acertos/ouro |
|---|---|---:|---:|---:|---:|---:|
| validação local | geral | 0.8556 | 0.7493 | 0.8419 | 0.7373 | 1061/1416 |
| validação local | N1 | 0.8382 | 0.7286 | 0.8252 | 0.7173 | 518/711 |
| validação local | N2 | 0.8730 | 0.7702 | 0.8585 | 0.7574 | 543/705 |
| validação local | jurisprudencia | 0.6942 | 0.6206 | — | — | 386/622 |
| validação local | legislacao | 0.9620 | 0.8287 | — | — | 658/794 |
| teste local | geral | 0.8927 | 0.8492 | 0.8912 | 0.8477 | 1132/1333 |
| teste local | N1 | 0.8943 | 0.8488 | 0.8927 | 0.8473 | 567/668 |
| teste local | N2 | 0.8912 | 0.8496 | 0.8896 | 0.8481 | 565/665 |
| teste local | jurisprudencia | 0.7876 | 0.7986 | — | — | 456/571 |
| teste local | legislacao | 0.9782 | 0.8845 | — | — | 674/762 |

## Dados e recursos

- Dados locais: 412 documentos de treino, 62 de validação e 56 de teste. Os dez documentos oficiais autorizados entraram somente no treino; 16 ficaram reservados para score final quando solicitado.
- GPU: NVIDIA GeForce RTX 3060; pico alocado: 2.08 GiB; pico reservado: 2.24 GiB.
- Batch por dispositivo: 4; acumulação: 4; batch efetivo: 16; FP16: True; gradient checkpointing: True.
- Rodada selecionada: LR 0.0000300, 486 passos, época 3.00, loss treino 0.0308 e loss de validação 0.3901; runtime de treino 342.3 s.
- Tempo contabilizado no orçamento de preparação e treino: 1099.8 s de 36000 s.

| Etapa | Segundos |
|---|---:|
| `fine_tuning_lr_1e-5` | 348.3 |
| `fine_tuning_lr_2e-5` | 347.3 |
| `fine_tuning_lr_3e-5` | 342.4 |
| `prepare_and_teacher_inference` | 61.8 |

## Teste local e procedência

O score final local do checkpoint selecionado, avaliado uma vez após a seleção, foi **0.6804** (N1 0.7987; N2 0.6213). Este teste local não é score oficial do desafio. Os documentos oficiais reservados não foram avaliados nesta execução.

Hash dos pesos selecionados: `79ab819ef03afa691701b41e8bf1d73855c3f25101f50c4e3d086c3c6947f877`. Hiperparâmetros, defaults resolvidos, tokenizer, arquitetura, ambiente e configurações efetivas estão em `training_config.resolved.json`; hashes e métricas estruturadas estão em `model_report.json` e `provenance.json`.
