param(
    [string]$DatabasePath,
    [string]$TextDirectory,
    [string]$OutputFile
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($DatabasePath) -or
    [string]::IsNullOrWhiteSpace($TextDirectory) -or
    [string]::IsNullOrWhiteSpace($OutputFile) -or
    $args.Count -gt 0) {
    [Console]::Error.WriteLine('Uso: .\run.ps1 <caminho_db> <pasta_txt> <arquivo_saida>')
    exit 2
}

$RepoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    [Console]::Error.WriteLine('Erro: Docker CLI não encontrado. Instale e inicie o Docker Desktop.')
    exit 1
}

try {
    $DatabasePath = (Resolve-Path -LiteralPath $DatabasePath).Path
    $TextDirectory = (Resolve-Path -LiteralPath $TextDirectory).Path
} catch {
    [Console]::Error.WriteLine("Erro: banco ou pasta de documentos não encontrado. $($_.Exception.Message)")
    exit 1
}
if (-not (Test-Path -LiteralPath $DatabasePath -PathType Leaf)) {
    [Console]::Error.WriteLine("Erro: banco SQLite não encontrado: $DatabasePath")
    exit 1
}
if (-not (Test-Path -LiteralPath $TextDirectory -PathType Container)) {
    [Console]::Error.WriteLine("Erro: pasta de documentos não encontrada: $TextDirectory")
    exit 1
}
if (-not (Get-ChildItem -LiteralPath $TextDirectory -Filter '*.txt' -File | Select-Object -First 1)) {
    [Console]::Error.WriteLine("Erro: a pasta não contém arquivos .txt: $TextDirectory")
    exit 1
}

if ([System.IO.Path]::IsPathRooted($OutputFile)) {
$OutputPath = [System.IO.Path]::GetFullPath($OutputFile)
} else {
    $OutputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $OutputFile))
}
$OutputDirectory = [System.IO.Path]::GetDirectoryName($OutputPath)
$OutputName = [System.IO.Path]::GetFileName($OutputPath)
if ([string]::IsNullOrWhiteSpace($OutputName) -or $OutputName -in @('.', '..')) {
    [Console]::Error.WriteLine('Erro: informe um caminho de arquivo CSV válido.')
    exit 1
}
if (Test-Path -LiteralPath $OutputPath -PathType Container) {
    [Console]::Error.WriteLine('Erro: o caminho de saída aponta para um diretório, não para um arquivo.')
    exit 1
}
if ([string]::Equals($OutputPath, $DatabasePath, [System.StringComparison]::OrdinalIgnoreCase)) {
    [Console]::Error.WriteLine('Erro: o arquivo de saída não pode substituir o banco de entrada.')
    exit 1
}
$TextDirectory = $TextDirectory.TrimEnd([char[]]@([char]92, [char]47))
$TextPrefix = $TextDirectory + [System.IO.Path]::DirectorySeparatorChar
if ([string]::Equals($OutputDirectory.TrimEnd([char[]]@([char]92, [char]47)), $TextDirectory, [System.StringComparison]::OrdinalIgnoreCase) -or
    $OutputDirectory.StartsWith($TextPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    [Console]::Error.WriteLine('Erro: o arquivo de saída não pode ficar dentro da pasta de documentos.')
    exit 1
}
try {
    [System.IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
} catch {
    [Console]::Error.WriteLine("Erro: não foi possível criar o diretório de saída. $($_.Exception.Message)")
    exit 1
}

$DockerOs = & docker info --format '{{.OSType}}' 2>$null
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine('Erro: Docker não está ativo ou não pode ser acessado.')
    exit 1
}
if (($DockerOs | Out-String).Trim() -ne 'linux') {
    [Console]::Error.WriteLine('Erro: este pipeline requer o Docker configurado para contêineres Linux.')
    exit 1
}

Write-Output 'Construindo a imagem do pipeline (o primeiro build pode baixar os pesos do modelo)...'
& docker build --progress plain --tag juris-check:local --file (Join-Path $RepoRoot 'Dockerfile') $RepoRoot
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine('Erro: não foi possível construir a imagem Docker do pipeline.')
    exit 1
}
$ImageIdOutput = & docker image inspect --format '{{.Id}}' juris-check:local
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine('Erro: não foi possível identificar a imagem recém-construída.')
    exit 1
}
$ImageId = ($ImageIdOutput | Out-String).Trim()
if ($ImageId -notmatch '^sha256:[0-9a-f]{64}$') {
    [Console]::Error.WriteLine('Erro: o build não retornou um ID de imagem Docker válido.')
    exit 1
}

Write-Output 'Validando GPU NVIDIA no contêiner e executando o pipeline...'
$DockerArguments = @(
    'run', '--rm', '--gpus', 'all', '--network', 'none',
    '--mount', "type=bind,source=$TextDirectory,target=/input/txt,readonly",
    '--mount', "type=bind,source=$DatabasePath,target=/input/reference.sqlite,readonly",
    '--mount', "type=bind,source=$OutputDirectory,target=/output",
    '--env', 'PROTECTED_DATA_DIR=/input',
    '--env', "OUTPUT_FILE=$OutputName",
    '--entrypoint', 'sh', $ImageId, '-ec',
    'python -m scripts.smoke_ner --require-cuda && exec python -m src.pipeline --input /input/txt --db /input/reference.sqlite --output /output --output-file "$OUTPUT_FILE"'
)
& docker @DockerArguments
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine("Erro: execução Docker falhou (código $LASTEXITCODE). Confira GPU, Docker e as mensagens acima.")
    exit $LASTEXITCODE
}
