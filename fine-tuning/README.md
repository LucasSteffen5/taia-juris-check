# Fine-tuning isolado do extrator NER

Este módulo produz e avalia checkpoints derivados de `dominguesm/legal-bert-ner-base-cased-ptbr` para a extração de spans de jurisprudência e legislação no desafio descrito em `DESAFIO.md`. O treino usa imagem, projeto Compose, cache e diretório de execução próprios e não importa o pacote `src`. O pipeline principal carrega por padrão o v2 congelado em `fine-tuning/artifacts/v2-selected`; os avaliadores continuam recebendo explicitamente cada checkpoint que devem comparar.

O modelo mantém a cabeça NER de 13 rótulos. Ele identifica limites e tipos de entidades; a classificação final em `real`, `inventada` e `incompleta` continua sendo decidida pela resolução na base canônica do pipeline.

## Requisitos e início rápido

- Docker Desktop em modo Linux com suporte NVIDIA Container Toolkit, CUDA visível no Docker e `nvidia-smi` no host.
- RTX 3060 de 12 GB livre durante preparação e treinamento. O preflight exige pelo menos 8 GiB livres e mede o batch que cabe antes de começar.
- A exclusividade da GPU durante o treino é operacional: não inicie o pipeline da banca nem outro processo CUDA enquanto a campanha estiver rodando. A GPU não particiona VRAM entre esses contêineres; o preflight verifica memória livre no início, mas não bloqueia outros programas iniciados depois.
- Os datasets locais e o checkpoint original precisam estar no repositório nos caminhos documentados abaixo. O checkpoint original é validado pelos hashes fixados no projeto.
- A campanha não impõe limite de tempo de GPU. Tempo, throughput, memória e duração por etapa são registrados; batch e margem de VRAM são medidos no preflight.
- Para uma clonagem limpa, execute `git lfs pull`. O checkpoint v1 de referência está versionado em `fine-tuning/artifacts/v1-reference`; o v2 escolhido e todos os registros exportáveis estão em `fine-tuning/artifacts/v2-selected`. Veja [artifacts/README.md](artifacts/README.md).

No PowerShell, execute da raiz do repositório:

```powershell
.\fine-tuning\run.ps1 train
```

No Bash/WSL:

```bash
bash fine-tuning/run.sh train
```

Um identificador de execução pode ser passado depois de `train`. Sem ele, o wrapper cria um ID com data e hora. Os scripts verificam Docker, GPU, memória livre e os arquivos requeridos antes de iniciar; o preflight CUDA mede também o batch seguro antes da preparação. `train` conduz a campanha até a avaliação oficial final: primeiro congela o modelo por validação local e teste local, depois compara original, v1 e v2 nos 16 documentos reservados. Esse resultado não volta ao ajuste. Nenhum upload ao Hugging Face é automático. Use `campaign <run-id>` para continuar uma execução já iniciada.

Após a seleção local e a validação offline, `evaluate-official` repete apenas a avaliação final dos três modelos e atualiza o pacote sem treinar nem escolher outro checkpoint. Use `./fine-tuning/run.ps1 evaluate-official -RunId <run-id>` no PowerShell ou `bash fine-tuning/run.sh evaluate-official <run-id>` no Bash. O comando confirma o SHA-256 dos pesos congelados antes de iniciar e valida novamente o pacote ao terminar.

## Dados, proteção contra vazamento e preparação

O dataset local é lido sem alteração: 530 documentos, 10.296 anotações e 18 textos sem citação. Os splits LeNER originais permanecem intactos (100 treino, 20 validação e 20 teste). Os 65 contextos sintéticos são inicialmente ordenados por SHA-256 de `42:<contexto>` em 52/7/6; as seis variantes de cada contexto, inclusive N1/N2, ficam juntas. Como o sorteio inicial colocava todos os contextos negativos no treino, a preparação troca contextos completos, de forma determinística, para manter pelo menos um contexto negativo em cada divisão sem alterar as contagens 52/7/6. Assim treino, validação e teste têm seis documentos sem citações cada, sem vazamento de contexto. A preparação falha se documento, grupo ou texto idêntico atravessar divisões. O resultado esperado continua sendo 412 documentos locais de treino, 62 de validação e 56 de teste; o manifesto registra os grupos negativos e as contagens por divisão.

O estágio Docker lê o CSV oficial e copia para a execução apenas os dez documentos autorizados e suas linhas de anotação:

| Nível | IDs usados somente no treino |
|---|---|
| N1 | `gen_n1_001`, `gen_n1_005`, `gen_n1_006`, `gen_n1_008`, `gen_n1_013` |
| N2 | `gen_n2_001`, `gen_n2_002`, `gen_n2_006`, `gen_n2_011`, `gen_n2_013` |

A seleção é reproduzida pelos cinco menores SHA-256 de `42:<documento_id>` em cada nível. Os outros 16 arquivos e suas anotações não são montados no contêiner de treinamento nem participam de avaliação, seleção de checkpoint ou ajuste de hiperparâmetro. Depois de congelar a escolha local, a última fase compara original, v1 e v2 nesses 16 documentos; esse resultado encerra a campanha e não altera o modelo.

Os textos e CSVs originais nunca são regravados. Cópias de texto são normalizadas para LF antes de conferir offsets Unicode com `texto[inicio:fim]`; offsets divergentes interrompem a preparação. Identificadores administrativos confirmados como pertencentes ao processo no cabeçalho viram negativos supervisionados como `O`; sem marcador explícito, o filtro não presume que o começo do documento seja cabeçalho. Referências a outros processos no corpo permanecem disponíveis. Anotações sobrepostas ou ambíguas recebem peso zero e cada decisão fica no manifesto. A base SQLite é montada apenas no contêiner separado de avaliação para a resolução canônica.

O checkpoint original também gera pseudoanotações auxiliares nos textos de treino: confiança mínima de 0,90, somente rótulos auxiliares fora das citações supervisionadas, peso 0,25. O `train.jsonl` registra rótulo, peso e confiança aceita por token; seus hashes entram no manifesto e são verificados antes de treinar ou retomar. As anotações manuais de jurisprudência e legislação têm peso 1,0; predições do professor nunca viram positivos extras para esses dois tipos.

## Treinamento e retomada

Cada rodada começa do checkpoint original, v1 ou de um candidato intermediário registrado e preserva tokenizer, arquitetura e os 13 rótulos. Checkpoints antigos sem GradScaler servem como pesos iniciais, mas não permitem retomada fiel. O preflight tenta batch 16/8/4/2/1 e escolhe o maior que deixa a margem de memória definida; acumulação 1/2/4/8/16 mantém batch efetivo 16. Usa FP16, gradient checkpointing, AdamW, weight decay 0,01, warmup linear de 10%, clipping 1,0 e até 12 épocas. Avalia aproximadamente quatro vezes por época e começa a contar early stopping depois da primeira época, após oito avaliações sem melhora de `eval_loss` (diagnóstico do treino). Cada snapshot inclui tokenizer, configuração e procedência derivada CC BY 4.0 para que o loader congelado aceite o modelo; o avaliador externo mede o score completo local e escolhe o snapshot, com desempate por N2 e precisão de spans. `repair-snapshots` migra snapshots legados que ainda não contenham essa procedência. As janelas têm até 510 tokens incluindo especiais, sobreposição de 128 e nenhum truncamento.

O conjunto base contém oito rodadas: cada ponto de partida (`original` e `v1`) × learning rates `5e-6`, `1e-5`, `2e-5` e `3e-5`. A campanha acrescenta 18 ablações de peso auxiliar × exposição oficial; 12 variantes de ruído OCR × sampler; quatro réplicas de estabilidade para duas finalistas; e seis treinamentos de validação cruzada (três folds por finalista). Os documentos oficiais são repetidos conforme o fator de cada experimento; o sampler preserva essa proporção ao acrescentar OCR. O estimador mede a duração e não bloqueia rodadas por orçamento. Checkpoints completos mantêm pesos, otimizador, scheduler, scaler FP16, RNG e estado do Trainer e recebem um marcador com hashes. Snapshots menores guardam pesos e tokenizer para a avaliação completa. Uma retomada exige marcador e estados completos da mesma execução e do mesmo experimento:

```powershell
.\fine-tuning\run.ps1 resume -RunId ft-AAAAmmdd-HHmmss -Experiment original_lr_2e-05 -CheckpointPath .\fine-tuning\runs\ft-AAAAmmdd-HHmmss\experiments\original_lr_2e-05\checkpoint-100
```

```bash
bash fine-tuning/run.sh resume ft-AAAAmmdd-HHmmss original_lr_2e-05 fine-tuning/runs/ft-AAAAmmdd-HHmmss/experiments/original_lr_2e-05/checkpoint-100
```

A retomada preserva os argumentos do Trainer, grupos do otimizador, batch e experimento. Mudança incompatível de configuração é registrada e bloqueada para evitar continuar um estado do otimizador com parâmetros diferentes. Enquanto uma rodada está ativa, os estados completos ficam disponíveis. Depois de selecionar seus pesos pelo score integral, a execução grava hash e tamanho de cada estado em `checkpoint_archive.json` e remove os estados grandes do otimizador para manter o uso de disco sob controle; pesos selecionados, métricas, configuração e manifestos permanecem. Uma rodada já concluída reinicia como um novo experimento a partir desses pesos, não como retomada de Trainer. Os checkpoints intermediários da execução v1 anterior não incluem estado válido do GradScaler e não podem retomar fielmente FP16; continuam utilizáveis para inferência ou como inicialização de um treino novo. O teste CUDA interrompeu e retomou em processos distintos e comparou pesos, scaler, LR, otimizador, scheduler e RNG com um treino contínuo.

## Registro completo por execução

Tudo fica em `fine-tuning/runs/<run-id>/`, ignorado pelo Git. JSON registra configuração resolvida, ambiente, origem e hashes; JSONL recebe histórico incremental de métricas, LR por atualização, tempo, memória, interrupções, retomadas e eventos de orçamento; Markdown guarda o relatório humano e o model card. A configuração de cada experimento inclui defaults resolvidos das bibliotecas, tokenizer, modelo, rótulos, parâmetros treináveis, grupos efetivos do AdamW, perda e pesos, pseudoanotações, amostras/janelas, batching, passos, warmup e argumentos do `TrainingArguments`.

Os manifestos também guardam revisão/hash do modelo inicial, hashes das fontes e dados, divisões, transformações de anotação, versões de Python/PyTorch/Transformers/Accelerate/CUDA/cuDNN, GPU, imagem Docker, commit e hash do código copiado. `configuration_history.jsonl` registra a configuração antes de cada início ou retomada; cada rodada tem `budget_estimate.json` e `experiment_manifest.json`. As atualizações de LR, loss, validação, época, passos e pico de VRAM são acrescentadas em `history.jsonl`. O determinismo total não é garantido por CUDA; essa limitação e o ambiente real ficam documentados.

Arquivos centrais da execução:

- `prepared/dataset_manifest.json`: hashes, divisões, contagens, transformação de spans e pseudoanotações.
- `campaign_config.resolved.json`, `experiments/<rodada>/training_config.resolved.json` e `configuration_history.jsonl`: hiperparâmetros declarados e efetivos.
- `history.jsonl`, `events.jsonl`, `budget.json` e `campaign_progress.json`: treinamento, retomadas e consumo de recursos.
- `selection.json`, `metrics/` e `candidates/<rodada>/`: decisão e candidatos.
- `candidates/<rodada>/model_report.md` e `model_report.json`: relatório legível e estruturado com comparação, métricas por nível/tipo e consumo de GPU.
- `candidates/<rodada>/README.md`, `provenance.json` e pesos/tokenizer: pacote exportado pronto para revisão/publicação.

## Avaliação e escolha do modelo

O contêiner avaliador é uma imagem diferente, com volumes de dados, manifesto e diretório de saída próprios. Ele executa o pipeline existente com os baselines e cada candidato na validação local; as divisões reservam contextos negativos para medir falsos positivos fora do treino. O relatório separa NER bruto, regex, fusão e resolução, além de spans exatos, IoU, tipo, nível e falsos positivos em documentos sem citações. O maior score local elegível vence; um candidato só é recomendado se não aumentar `inventada → real` contra o original e o v1. Empates usam score N2 e precisão de spans. Se nenhum candidato cumprir o limite, a exportação recebe marca experimental e não é recomendada para troca. O teste local do selecionado é executado uma vez depois da escolha.

Depois de congelar a escolha local, o wrapper avalia o original, o v1 e o v2 nos 16 documentos oficiais restantes. Confere os IDs de treino pelo manifesto, prepara cópias UTF-8 com quebras LF para corresponder aos offsets oficiais, filtra as linhas do CSV oficial para esses 16 IDs e exige que cada trecho anotado coincida com `texto[inicio:fim]` antes de executar o pipeline. Documentos sem citações também entram na solução pontuada. As fontes não são alteradas; os hashes dos textos de origem e preparados, do CSV integral e filtrado e do código do filtro ficam registrados para auditoria. Exige uma linha de predição por documento e registra hashes do SQLite, da métrica, do avaliador e dos pesos. Audita correspondências exatas e similaridade alta com cada texto local e os dez oficiais usados no treino. Folds e variantes OCR derivam desses textos locais, sem adicionar novas fontes. A validação cruzada apresenta score médio e dispersão entre folds; os relatórios locais também incluem bootstrap agrupado por contexto, identificado como diagnóstico descritivo. O score oficial não volta à seleção ou ao ajuste de hiperparâmetros; essa avaliação encerra a campanha.

## Predição explícita com um checkpoint derivado

O comando principal e os comandos de predição deste módulo validam o pacote antes de executar. Para comparar outra exportação derivada de forma explícita:

```powershell
.\fine-tuning\run.ps1 predict -CheckpointPath .\fine-tuning\runs\<run-id>\candidates\<rodada> -InputDirectory .\caminho\para\txt -DatabasePath .\data\dataset\desafio1_bracis.db -OutputPath .\saida\submission.csv
```

```bash
bash fine-tuning/run.sh predict fine-tuning/runs/<run-id>/candidates/<rodada> caminho/para/txt data/dataset/desafio1_bracis.db saida/submission.csv
```

O launcher valida o pacote offline, monta o checkpoint selecionado como somente leitura e grava um manifesto com o hash efetivamente usado. O uso do v1, de outra exportação ou do modelo original fica disponível pela escolha explícita do checkpoint nos comandos de avaliação.

## Exportação e publicação no Hugging Face

O diretório exportado contém pesos de inferência, tokenizer, model card em português, procedência e hashes, configuração completa, manifestos, relatórios e `REPRODUCE.md`. Os estados grandes do otimizador, scheduler e GradScaler foram compactados após a seleção e não entram no pacote. O comando de upload valida de novo os hashes, a procedência, os 13 rótulos, o tokenizer e uma inferência local offline antes de chamar o Hub.

Execute `validate-export` para revisar um pacote e `upload` para publicá-lo. O [tutorial de publicação no Hugging Face](PUBLICAR_HUGGING_FACE.md) cobre autenticação, publicação privada ou pública, arquivos enviados e carregamento por revisão fixa. O serviço de publicação recebe rede; treino e avaliação permanecem em contêineres offline.

## Testes e falhas comuns

Execute os testes específicos dentro da imagem isolada:

```powershell
$env:REPO_ROOT = (Get-Location).Path
$env:RUNS_HOST_DIR = (Join-Path $env:REPO_ROOT 'fine-tuning/runs')
$env:MODEL_HOST_DIR = (Join-Path $env:REPO_ROOT 'models/ner_legal_large')
$env:OFFICIAL_TRAIN_HOST_DIR = (Join-Path $env:RUNS_HOST_DIR 'ft-tests/sources/official')
docker compose --project-name juris-check-finetuning -f fine-tuning/compose.yaml build trainer
New-Item -ItemType Directory -Force -Path $env:RUNS_HOST_DIR | Out-Null
docker compose --project-name juris-check-finetuning -f fine-tuning/compose.yaml run --rm stager --run-id ft-tests
docker compose --project-name juris-check-finetuning -f fine-tuning/compose.yaml run --rm --entrypoint python trainer -m pytest -q /workspace/fine-tuning/tests
```

Se o Docker disser que não encontra CUDA, troque o Docker Desktop para Linux containers, atualize/valide o NVIDIA Container Toolkit e confira `docker run --rm --gpus all ... nvidia-smi`. Se a GPU tiver menos de 8 GiB livres, encerre outras aplicações que usam CUDA e tente novamente; o preflight mede o maior microbatch que preserva 20% de margem. O orçamento de GPU desta campanha é ilimitado (`null`); quando uma execução definir um orçamento opcional, uma rodada cuja estimativa o exceda será adiada e o motivo ficará no histórico. Se falhar na validação de offset, investigue codificação/normalização do texto original; não corrija o CSV de origem para contornar a auditoria.
