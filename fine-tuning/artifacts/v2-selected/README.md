---
language:
  - pt
license: cc-by-4.0
library_name: transformers
pipeline_tag: token-classification
base_model: dominguesm/legal-bert-ner-base-cased-ptbr
base_model_relation: finetune
tags:
  - legal-ner
  - brazilian-portuguese
  - citation-extraction
---

# NER jurídico brasileiro para spans de citações

Este checkpoint continua o ajuste do modelo [dominguesm/legal-bert-ner-base-cased-ptbr](https://huggingface.co/dominguesm/legal-bert-ner-base-cased-ptbr/tree/44210927c925448df025985e0ed48081bb5ac57c), revisão `44210927c925448df025985e0ed48081bb5ac57c`, com objetivo de detectar spans de jurisprudência e legislação em documentos judiciais brasileiros. A cabeça original de treze rótulos foi mantida; a resolução canônica e as classes finais do desafio são produzidas pelo pipeline hospedeiro.

## Uso pretendido e limites

Use-o como extrator NER de jurisprudência, legislação e entidades auxiliares, carregando os arquivos locais com `AutoTokenizer` e `AutoModelForTokenClassification`. O modelo não determina sozinho se uma citação é real, inventada ou incompleta. Essa decisão depende da base canônica e das regras do pipeline. O checkpoint é experimental: `false`.

## Treinamento reproduzível

- Experimento selecionado: `s3_1_7e4cf94f_o00_w`. Veja `training_config.resolved.json` para todos os argumentos e valores efetivos, inclusive defaults resolvidos, parâmetros do otimizador, warmup, seeds, batch efetivo e versões do ambiente.
- `REPRODUCE.md` traz o comando de reexecução, a dependência do candidato anterior e as limitações de retomada após a compactação. `experiment_manifest.json`, `history.jsonl`, `configuration_history.jsonl` e `model_report.json` registram passos, hiperparâmetros, mudanças, interrupções, hashes e validação.
- Treino local: 412 documentos; validação: 62; teste local: 56. Os grupos de variantes do mesmo contexto permanecem juntos.
- A etapa final acrescentou apenas estes dez documentos autorizados do dataset oficial: gen_n1_001, gen_n1_005, gen_n1_006, gen_n1_008, gen_n1_013, gen_n2_001, gen_n2_002, gen_n2_006, gen_n2_011, gen_n2_013. Eles não foram usados para escolher o modelo.
- O dataset oficial restante não foi usado em treino, validação ou seleção.

## Resultados

Na divisão local, a seed 42 obteve 0.6754 na validação e 0.6695 no teste local. A escolha da configuração usou a média entre seeds; veja `model_report.md` para dispersão, níveis, tipos e regressões.

Após congelar o v2, os 16 documentos oficiais reservados foram avaliados uma vez para cada modelo, sem escolher outro checkpoint. A fórmula oficial inclui bônus de calibração e pode produzir score maior que 1.

| Modelo | Score oficial | N1 | N2 |
|---|---:|---:|---:|
| original | 1.082503 | 1.093750 | 1.076880 |
| v1 | 1.082703 | 1.093855 | 1.077127 |
| v2 | 1.082643 | 1.093820 | 1.077055 |

Maior score observado: **v1**. Diferença v2 − v1: **-0.000060**. Os três modelos tiveram o mesmo F1 por classe no conjunto reservado; a diferença de score veio do bônus de calibração. Troca automática do modelo da banca recomendada: **não**. `official_evaluation.json` e `official_comparison.json` trazem os IDs, métricas por nível/tipo, auditoria de offsets e hashes. A primeira tentativa com cópias CRLF foi invalidada antes desta comparação; `evaluation_correction.json` preserva o diagnóstico e os hashes dos relatórios descartados.

## Procedência e licença

Modelo de origem: [dominguesm/legal-bert-ner-base-cased-ptbr](https://huggingface.co/dominguesm/legal-bert-ner-base-cased-ptbr/tree/44210927c925448df025985e0ed48081bb5ac57c), licença CC BY 4.0. Este fine-tune mantém a atribuição exigida e usa os dados listados no relatório de preparação do projeto. `provenance.json` contém revisão, hashes dos pesos e hashes dos arquivos publicados.
