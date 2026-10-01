# Publicar o fine-tuning v2 no Hugging Face

Este tutorial publica a exportação selecionada em `fine-tuning/artifacts/v2-selected`. O v2 foi escolhido pela validação local. O modelo extrai spans de jurisprudência, legislação e outras entidades; para classificar citações como reais, inventadas ou incompletas, use o pipeline e a base canônica deste repositório.

## Pré-requisitos

- Git LFS instalado e executado (`git lfs pull`) para obter o arquivo de pesos completo.
- Docker Desktop ou Docker Engine em modo Linux. O upload usa o contêiner isolado do fine-tuning e também pode ser feito sem GPU.
- Um token Hugging Face com permissão `write` para a conta ou organização e o repositório de modelo de destino. Tokens de escopo limitado devem ter acesso de gravação a esse repositório.

Confira a exportação antes de publicar:

```powershell
.\fine-tuning\run.ps1 validate-export -CheckpointPath .\fine-tuning\artifacts\v2-selected
```

```bash
bash fine-tuning/run.sh validate-export fine-tuning/artifacts/v2-selected
```

O validador confirma os hashes da exportação, a procedência e a revisão do modelo-base, os 13 rótulos, o tokenizer e uma inferência de teste, com a rede desligada. O SHA-256 dos pesos v2 esperado é `52b24416c23a645db9aa49e83cf59e14aa8b225ae396f01d48415d50ae67945c`.

## Autenticar e enviar

Crie previamente o repositório de modelo no Hub ou use o identificador `usuario/repositorio` da sua conta ou organização. Para um token pessoal, prefira um token `write` dedicado à publicação. Não salve o token em arquivo nem o inclua no histórico do terminal.

No PowerShell, leia o token sem exibi-lo e publique como privado:

```powershell
$secureToken = Read-Host 'Token Hugging Face com escrita' -AsSecureString
$env:HF_TOKEN = [System.Net.NetworkCredential]::new('', $secureToken).Password
.\fine-tuning\run.ps1 upload -CheckpointPath .\fine-tuning\artifacts\v2-selected -RepoId usuario/repositorio -Private
Remove-Item Env:HF_TOKEN
```

No Bash, leia o token sem eco e publique como privado:

```bash
read -rsp 'Token Hugging Face com escrita: ' HF_TOKEN
printf '\n'
export HF_TOKEN
bash fine-tuning/run.sh upload fine-tuning/artifacts/v2-selected usuario/repositorio --private
unset HF_TOKEN
```

Para publicar em um repositório público, retire `-Private` ou `--private`. O comando verifica o pacote localmente antes de fazer chamadas ao Hub, cria o repositório de modelo caso ainda não exista e envia os arquivos permitidos. Só o serviço `publisher` recebe rede e `HF_HUB_OFFLINE=0`; treino, validação e inferência permanecem offline. Se o envio falhar, confira o acesso `write`, a conectividade do host e se a revisão local do Git LFS foi baixada.

A lista autorizada contém os pesos, tokenizer, `README.md`, arquivos de configuração, procedência e relatórios de campanha. Textos de documentos, anotações CSV, SQLite, credenciais, cache e estado de otimizador não fazem parte da publicação.

## Conferir a publicação e carregar por revisão

Abra `https://huggingface.co/usuario/repositorio` e confirme a visibilidade, `README.md`, `config.json`, tokenizer, `model.safetensors`, `provenance.json` e os relatórios. No histórico do repositório, copie o SHA completo do commit publicado. Use esse SHA em vez de `main` ao baixar os pesos:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

repo_id = "usuario/repositorio"
revision = "<sha-completo-do-commit-publicado>"
tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
model = AutoModelForTokenClassification.from_pretrained(repo_id, revision=revision)
```

Para um repositório privado, autentique `huggingface_hub` ou forneça `token=True` aos dois carregadores. Compare o `model.safetensors` baixado com o SHA-256 em `provenance.json` antes de usar os pesos. O loader faz inferência NER; a classificação de citações e a resolução canônica ainda exigem os componentes e dados do pipeline.

## Atribuição e licença

Inclua e preserve o model card do pacote. O fine-tuning deriva de [Maicon Domingues, Legal-BERT NER em português jurídico, revisão `44210927c925448df025985e0ed48081bb5ac57c`](https://huggingface.co/dominguesm/legal-bert-ner-base-cased-ptbr/tree/44210927c925448df025985e0ed48081bb5ac57c), sob **CC BY 4.0**. Mantenha essa atribuição e a licença indicada no `README.md` e em `provenance.json` ao redistribuir o modelo.

Referências oficiais: [envio de arquivos e pastas ao Hub](https://huggingface.co/docs/huggingface_hub/v0.26.5/en/guides/upload) e [permissões de tokens de acesso](https://huggingface.co/docs/hub/security-tokens).
