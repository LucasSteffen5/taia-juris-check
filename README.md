# Verificador de citações jurídicas

Pipeline offline para encontrar referências a jurisprudência e leis em documentos judiciais, resolver identificadores contra o SQLite congelado do desafio e gerar os arquivos de submissão.

## Executar

Requer Docker Desktop com motor Linux. A construção baixa versões fixadas das dependências e empacota o checkpoint local. Depois, a inferência roda sem rede. Os pesos do modelo são armazenados com Git LFS.

No PowerShell:

```powershell
.\run.ps1 build
.\run.ps1 test
.\run.ps1 predict
.\run.ps1 score
```

No Linux:

```bash
bash run.sh build
bash run.sh test
bash run.sh predict
bash run.sh score
```

`predict` verifica CUDA dentro de um contêiner; usa NVIDIA quando disponível e muda para CPU quando não estiver. Os documentos e o banco são montados para leitura. JSONs, CSV e manifesto aparecem em `output/simulation/`. `score` é uma ação separada e só então monta o gabarito.

O manifesto registra hashes do código, dependências, entrada, banco, checkpoint e CSV, além do dispositivo e tempo de execução. O índice SQLite fica no diretório de saída e é reconstruído quando muda o banco ou parser.

## Como decide

Regex identifica padrões de processo, súmula, lei e alusões vagas. O NER local procura trechos que escapam aos padrões. A fusão usa sobreposição dos spans e preserva os offsets do texto UTF-8 original, inclusive CRLF. O resolvedor consulta o índice dos identificadores principais dos acórdãos e os registros específicos de súmulas e leis; a busca FTS é confirmada contra o cabeçalho do processo.

Uma referência sem dados suficientes ou com múltiplos candidatos é `incompleta`; uma consulta suficiente sem resultado é `inventada`; um único registro confirmado é `real` e recebe o `id` numérico canônico. O campo de confiança é sempre `0.5`: estimativa neutra sem calibração, com Brier esperado de `0.25` em pares avaliados.

## Modelo, licenças e dados

O checkpoint é `dominguesm/legal-bert-ner-base-cased-ptbr`, revisão `44210927c925448df025985e0ed48081bb5ac57c`. O hash dos pesos corresponde ao arquivo publicado nessa revisão. A licença indicada pelo cartão oficial é **CC BY 4.0**; a atribuição, revisão e hashes locais ficam em `models/ner_legal_large/provenance.json`. O arquivo de configuração que acompanhava o exemplo dizia MIT; essa indicação diverge do cartão do modelo e não é usada como licença do checkpoint.

O código deste repositório está sob MIT. As dependências têm suas licenças próprias. Os dados oficiais e gabarito permanecem conforme os termos da competição; não são usados para treino ou calibração. A base de desenvolvimento local deve creditar Eduardo Osti e Thiago Meirelles Ventura por disponibilizarem os dados brutos.

Os 26 documentos presentes são a amostra de desenvolvimento do desafio. O resultado de `score` sobre eles é uma **simulação de teste cego em amostra conhecida**, não uma previsão do ranking final. Primeiro execute `predict`; examine o manifesto e só então `score`. Esta entrega não retreina nem ajusta o sistema com base nesse score.
