# Verificador de Citações Jurídicas — Desafio Jusbrasil x BRACIS 2026

Solução para resolver e classificar citações jurídicas em documentos judiciais como **reais**, **inventadas** ou **incompletas**.

## Pré-requisitos

- Docker Desktop ou Docker Engine configurado para contêineres Linux.
- GPU NVIDIA e drivers configurados para acesso pelo Docker. No Linux, instale o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/container-toolkit/latest/install-guide.html); no Windows, use o Docker Desktop com suporte NVIDIA via WSL 2.

- Bash no Linux/WSL ou PowerShell no Windows.

## Executar

Passe o caminho do banco SQLite, a pasta com os `.txt` e o caminho completo para o CSV. O caminho de saída pode ser escolhido livremente; caminhos relativos partem do diretório atual.

```bash
bash run.sh ./data/desafio-jusbrasil-bracis-2026/desafio1_bracis.db ./data/desafio-jusbrasil-bracis-2026/txt ./output/submission.csv
```

No Windows PowerShell:

```powershell
.\run.ps1 .\data\desafio-jusbrasil-bracis-2026\desafio1_bracis.db .\data\desafio-jusbrasil-bracis-2026\txt .\output\submission.csv
```

O script constrói a imagem a partir do Dockerfile, valida CUDA dentro do contêiner e executa o pipeline com GPU NVIDIA. Na primeira execução, o build pode demorar enquanto baixa as dependências e os pesos. Execuções seguintes reaproveitam o cache do Docker.

O CSV é salvo exatamente no terceiro caminho informado. O contêiner é removido ao terminar; banco e documentos são montados somente para leitura.



## Arquitetura do Pipeline

```mermaid
flowchart LR
    TXT["Documentos .txt"] --> PIPE["Pipeline Docker (GPU)"]
    DB["Base SQLite Canônica"] --> PIPE
    PIPE --> EXTRACT["Extração híbrida<br/>(Regex + NER)"]
    EXTRACT --> RESOLVE["Resolução canônica<br/>(FTS5 + consulta exata)"]
    RESOLVE --> SUBMISSION["CSV de submissão"]
```
