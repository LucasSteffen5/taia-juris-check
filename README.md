# Verificador de citações jurídicas

Pipeline offline para encontrar referências a jurisprudência e leis em documentos judiciais, resolver identificadores contra o SQLite congelado do desafio e gerar os arquivos de submissão.

## Executar

Requer Docker com motor Linux e Git LFS para obter os pesos após clonar (`git lfs pull`). Edite apenas [`.env`](.env) para apontar para outra base: `INPUT_TXT_DIR` é a pasta de TXT, `DATABASE_PATH` é o arquivo SQLite, `GOLDEN_PATH` é o gabarito opcional e `OUTPUT_DIR` é onde persistir as saídas. Os caminhos podem ser relativos à raiz do projeto ou absolutos; a base não precisa manter nomes de arquivo nem estrutura de pastas específicos. Os TXT e o gabarito não são versionados: forneça seus caminhos no `.env` quando estiverem fora dos locais padrão. Os dados são montados no contêiner para leitura e não entram na imagem.

Da raiz do projeto, um único comando constrói a imagem e executa a inferência offline:

```bash
docker compose up --build --exit-code-from predict
```

O padrão em `.env` usa a GPU NVIDIA. Para executar em CPU, altere apenas `COMPOSE_FILE=compose.yaml` no mesmo arquivo. Os scripts `run.ps1 predict` e `bash run.sh predict` continuam disponíveis e detectam a GPU automaticamente. O manifesto registra o dispositivo efetivo.

JSONs, `submission.csv`, manifesto e cache ficam no caminho configurado em `OUTPUT_DIR`; continuam disponíveis no computador depois que o contêiner termina. Para testar sem carregar dados oficiais, use `docker compose run --rm --build test`. A avaliação da submissão já gerada é separada: `docker compose run --rm --no-deps score`. Só esse serviço monta o gabarito, se `GOLDEN_PATH` existir.

O manifesto registra hashes do código, dependências, entrada, banco, checkpoint e CSV, além do dispositivo e tempo de execução. O índice SQLite fica no diretório de saída e é reconstruído quando muda o banco ou parser.

## Como decide

Regex identifica padrões de processo, súmula, lei e alusões vagas. O NER local procura trechos que escapam aos padrões. A fusão usa sobreposição dos spans e preserva os offsets do texto UTF-8 original, inclusive CRLF. O resolvedor consulta o índice dos identificadores principais dos acórdãos e os registros específicos de súmulas e leis; a busca FTS é confirmada contra o cabeçalho do processo.

Uma referência sem dados suficientes ou com múltiplos candidatos é `incompleta`; uma consulta suficiente sem resultado é `inventada`; um único registro confirmado é `real` e recebe o `id` numérico canônico. O campo de confiança é sempre `0.5`: estimativa neutra sem calibração, com Brier esperado de `0.25` em pares avaliados.

## Modelo, licenças e dados

O checkpoint é `dominguesm/legal-bert-ner-base-cased-ptbr`, revisão `44210927c925448df025985e0ed48081bb5ac57c`. O hash dos pesos corresponde ao arquivo publicado nessa revisão. A licença indicada pelo cartão oficial é **CC BY 4.0**; a atribuição, revisão e hashes locais ficam em `models/ner_legal_large/provenance.json`. O arquivo de configuração que acompanhava o exemplo dizia MIT; essa indicação diverge do cartão do modelo e não é usada como licença do checkpoint.

O código próprio deste repositório está sob MIT. As dependências e a métrica oficial têm seus próprios termos. Os dados oficiais e gabarito permanecem conforme os termos da competição; não são usados para treino ou calibração. A base de desenvolvimento local deve creditar Eduardo Osti e Thiago Meirelles Ventura por disponibilizarem os dados brutos.

A amostra local de 26 documentos é o conjunto de desenvolvimento do desafio. O resultado de `score` sobre ela é uma **simulação de teste cego em amostra conhecida**, não uma previsão do ranking final. Primeiro execute `predict`; examine o manifesto e só então `score`. Esta entrega não retreina nem ajusta o sistema com base nesse score.
