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

- Experimento selecionado: `lr_3e-5`. Veja `training_config.resolved.json` para todos os argumentos e valores efetivos, inclusive defaults resolvidos, parâmetros do otimizador, warmup, seeds, batch efetivo e versões do ambiente.
- `experiment_manifest.json` e `model_report.json` registram passos, interrupções, hashes e validação.
- Treino local: 412 documentos; validação: 62; teste local: 56. Os grupos de variantes do mesmo contexto permanecem juntos.
- A etapa final acrescentou apenas estes dez documentos autorizados do dataset oficial: gen_n1_001, gen_n1_005, gen_n1_006, gen_n1_008, gen_n1_013, gen_n2_001, gen_n2_002, gen_n2_006, gen_n2_011, gen_n2_013. Eles não foram usados para escolher o modelo.
- O dataset oficial restante não foi usado em treino, validação ou seleção.

## Resultados

Consulte `model_report.md` para o relatório legível, comparação ao checkpoint inicial, resultados por nível e tipo e recursos medidos; `model_report.json` traz os dados estruturados e hashes. Esses resultados são do conjunto local dividido para desenvolvimento; não representam um score do conjunto final da competição.

## Procedência e licença

Modelo de origem: [dominguesm/legal-bert-ner-base-cased-ptbr](https://huggingface.co/dominguesm/legal-bert-ner-base-cased-ptbr/tree/44210927c925448df025985e0ed48081bb5ac57c), licença CC BY 4.0. Este fine-tune mantém a atribuição exigida e usa os dados listados no relatório de preparação do projeto. `provenance.json` contém revisão, hashes dos pesos e hashes dos arquivos publicados.
