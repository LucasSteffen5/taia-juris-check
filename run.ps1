param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('build', 'test', 'predict', 'score')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
try {
    switch ($Action) {
        'build' {
            & docker compose build predict
            if ($LASTEXITCODE -ne 0) { throw 'Falha ao construir a imagem Docker.' }
        }
        'test' {
            & docker compose run --rm --build test
            if ($LASTEXITCODE -ne 0) { throw 'Os testes automatizados falharam.' }
            & docker run --rm --network none --gpus all --entrypoint python juris-check:local -m scripts.smoke_ner --require-cuda
            if ($LASTEXITCODE -ne 0) { Write-Host 'GPU indisponível; teste em CPU concluído.' }
        }
        'predict' {
            & docker compose build predict
            if ($LASTEXITCODE -ne 0) { throw 'Falha ao construir a imagem Docker.' }
            & docker run --rm --network none --gpus all --entrypoint python juris-check:local -c 'import torch; assert torch.cuda.is_available()'
            if ($LASTEXITCODE -eq 0) {
                & docker compose up --no-build --exit-code-from predict
            }
            else {
                Write-Host 'GPU indisponível no Docker; executando em CPU.'
                & docker compose -f compose.yaml up --no-build --exit-code-from predict
            }
            if ($LASTEXITCODE -ne 0) { throw 'A inferência falhou.' }
        }
        'score' {
            & docker compose run --rm --no-deps score
            if ($LASTEXITCODE -ne 0) { throw 'A avaliação local falhou.' }
        }
    }
}
finally {
    Pop-Location
}
