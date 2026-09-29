param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('build', 'test', 'predict', 'score')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
    $Image = 'juris-check:local'
    $Official = Join-Path $Root 'data\desafio-jusbrasil-bracis-2026'
    $Output = Join-Path $Root 'output\simulation'

    if ($Action -eq 'build') {
        & docker build --tag $Image .
        if ($LASTEXITCODE -ne 0) { throw 'Falha ao construir a imagem Docker.' }
    }
    elseif ($Action -eq 'test') {
        & docker run --rm --network none --env CUDA_VISIBLE_DEVICES= --entrypoint python $Image -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'Os testes automatizados falharam.' }
        & docker run --rm --network none --env CUDA_VISIBLE_DEVICES= --entrypoint python $Image -m scripts.smoke_ner
        if ($LASTEXITCODE -ne 0) { throw 'O teste do NER em CPU falhou.' }
        & docker run --rm --network none --gpus all --entrypoint python $Image -m scripts.smoke_ner --require-cuda
        if ($LASTEXITCODE -ne 0) { Write-Host 'GPU indisponível para o teste; a execução em CPU foi validada.' }
    }
    elseif ($Action -eq 'predict') {
        New-Item -ItemType Directory -Force -Path $Output | Out-Null
        $InputMount = "type=bind,source=$(Join-Path $Official 'txt'),target=/input/txt,readonly"
        $DatabaseMount = "type=bind,source=$(Join-Path $Official 'desafio1_bracis.db'),target=/input/desafio1_bracis.db,readonly"
        $OutputMount = "type=bind,source=$Output,target=/output"
        $DockerArgs = @('--rm', '--network', 'none', '--mount', $InputMount, '--mount', $DatabaseMount, '--mount', $OutputMount)
        & docker run --rm --network none --gpus all --entrypoint python $Image -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
        if ($LASTEXITCODE -eq 0) {
            Write-Host 'GPU NVIDIA disponível; iniciando inferência em CUDA.'
            & docker run @DockerArgs --gpus all $Image --input /input/txt --db /input/desafio1_bracis.db --output /output
        }
        else {
            Write-Host 'GPU NVIDIA indisponível no Docker; iniciando inferência em CPU.'
            & docker run @DockerArgs $Image --input /input/txt --db /input/desafio1_bracis.db --output /output
        }
        if ($LASTEXITCODE -ne 0) { throw 'A inferência falhou.' }
    }
    elseif ($Action -eq 'score') {
        $GoldenMount = "type=bind,source=$(Join-Path $Official 'goldenset_offsets.csv'),target=/input/goldenset_offsets.csv,readonly"
        $OutputMount = "type=bind,source=$Output,target=/output"
        & docker run --rm --network none --mount $GoldenMount --mount $OutputMount --entrypoint python $Image -m scripts.evaluate --submission /output/submission.csv --golden /input/goldenset_offsets.csv --manifest /output/run_manifest.json --report /output/score.json
        if ($LASTEXITCODE -ne 0) { throw 'A avaliação local falhou.' }
    }
}
finally {
    Pop-Location
}
