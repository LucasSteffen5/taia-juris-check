# Reprodução da rodada selecionada

Esta exportação é da execução `ft-20260930-v2-balanced-campaign`, rodada `s3_1_7e4cf94f_o00_w`. Os pesos completos estão em `model.safetensors` (SHA-256 `52b24416c23a645db9aa49e83cf59e14aa8b225ae396f01d48415d50ae67945c`). A rodada começou do candidato `s2_1_a4bafc99_a10_r3`, SHA-256 `7e4cf94f9621d6fd7548e14bb4dcbd9c0e698f7dec0312d9ec07c9b533eadb5b`; reproduzir apenas esta rodada exige antes reconstruir esse candidato. O checkpoint original e o v1 usados pela campanha constam de `campaign_config.resolved.json` e `provenance.json`.

## Parâmetros efetivos

- Seed: 42; learning rate: 3e-05; peso da loss auxiliar: 0.1; repetição dos dez documentos oficiais: 3.
- Aumento OCR: 0.0; sampler: `window`; batch por dispositivo: 16; acumulação: 1; batch efetivo: 16.
- FP16: True; gradient checkpointing: True; épocas máximas: 12; warmup efetivo: 192 passos; scheduler: `linear`.
- GPU: NVIDIA GeForce RTX 3060; driver: 616.92; Python: 3.11.11; PyTorch: 2.5.1+cu124; Transformers: 4.46.3; imagem Docker: `sha256:f8435e6a021b5bb9eb75f234fbb943fd27e4571d6db6aa416cce5437e4eb844a`.

`training_config.resolved.json` contém todos os argumentos, defaults da versão usada, arquitetura, tokenizer, mapa dos 13 rótulos, loss, parâmetros treináveis, grupos do otimizador, janelas, amostragem e ambiente. `campaign_config.resolved.json`, `dataset_manifest.json`, `experiment_queue.json` e `selection.json` fixam as fontes, divisões, transformações, experiências e escolha. `configuration_history.jsonl`, `history.jsonl`, `events.jsonl`, `campaign_progress.json` e `budget.json` preservam ajustes, passos, LR, métricas, interrupções, tempo e recursos. `checkpoint_archive.json` registra os hashes dos estados FP16, RNG, otimizador e scheduler antes de sua compactação.

## Reexecutar no repositório

Com os mesmos dados, checkpoints originais, código e Docker disponíveis, da raiz do repositório execute:

```powershell
.\fine-tuning\run.ps1 -Mode train -RunId <novo-run-id>
```

O comando reconstrói a campanha e só consulta os 16 documentos oficiais reservados após congelar a seleção local. Compare os hashes de entrada e a configuração resolvida do novo run com os arquivos deste pacote antes de comparar pesos ou métricas. A reprodução bit a bit não é garantida entre ambientes CUDA. Esta rodada concluída teve estados completos do Trainer durante a execução; estados completos retidos após a seleção: `False`. Se os estados foram compactados, uma nova execução começa com otimizador, scheduler e GradScaler novos e não constitui retomada exata do checkpoint arquivado.
