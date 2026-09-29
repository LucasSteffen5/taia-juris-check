# Reprodução das bases

Este repositório contém o código e as entradas necessárias para gerar as bases sintética, LeNER e mesclada. As bases geradas e as auditorias não são versionadas: elas são saídas reproduzíveis do pipeline.

## Requisitos

- Python 3.11 ou superior;
- nenhuma dependência externa: o pipeline usa apenas a biblioteca padrão;
- execução a partir da raiz do repositório.

O pacote mínimo de entrada contém:

- `data/data_oversampling/`: código, configurações, testes e recursos versionados;
- `data/LeNER-Br/leNER-Br/{train,dev,test,raw_text}/`: os documentos usados pelo conversor;
- `data/desafio-jusbrasil-bracis-2026/desafio1_bracis.db`: catálogo SQLite congelado para resolução das citações.

Os TXT, goldenset e utilitários oficiais do desafio são mantidos fora do upload e devem ser usados somente na avaliação cega. O SQLite é usado como catálogo de referências, não como conjunto de treinamento.

## Execução no PowerShell

Na raiz do repositório, escolha um Python 3.11+ disponível:

```powershell
$Python = 'python'
& $Python --version
```

Execute as verificações e a geração:

```powershell
& $Python data/data_oversampling/generate_synthetic.py --config data/data_oversampling/config_synthetic_v3.toml --check-only
& $Python data/data_oversampling/build_lener.py --check-only
& $Python -m unittest discover -s data/data_oversampling/tests

& $Python data/data_oversampling/generate_synthetic.py --config data/data_oversampling/config_synthetic_v3.toml
& $Python data/data_oversampling/validate_synthetic.py

& $Python data/data_oversampling/build_lener.py
& $Python data/data_oversampling/validate_lener.py

& $Python data/data_oversampling/merge_datasets.py --check-only
& $Python data/data_oversampling/merge_datasets.py
```

Os resultados aparecem em `data/data_oversampling/dataset_synthetic_v3/`, `dataset_lener_v1/` e `dataset_merged_v1/`. As auditorias ficam em `data/data_oversampling/audits/`.

Os geradores não sobrescrevem saídas existentes. Para uma nova execução, remova somente as pastas geradas ou use os parâmetros de destino alternativo documentados em `data/data_oversampling/README.md`.

Para reconstruir os 65 contextos sintéticos versionados a partir do código, antes da geração execute:

```powershell
& $Python data/data_oversampling/build_synthetic_contexts.py --replace
```

Não remova `data/data_oversampling/resources/canonical_overrides.csv`: esse arquivo é uma entrada necessária e não é recriado pelo construtor de contextos.
