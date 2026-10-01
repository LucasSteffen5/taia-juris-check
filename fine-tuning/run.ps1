param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('train', 'campaign', 'resume', 'evaluate-official', 'predict', 'validate-export', 'upload')]
    [string]$Mode,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$RunId,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$Experiment,
    [string]$CheckpointPath,
    [string]$InputDirectory,
    [string]$DatabasePath,
    [string]$OutputPath,
    [string]$RepoId,
    [switch]$Private
)

$ErrorActionPreference = 'Stop'
$FineTuneDir = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $FineTuneDir '..')).Path
$ComposeFile = Join-Path $FineTuneDir 'compose.yaml'
$RunsRoot = Join-Path $FineTuneDir 'runs'
$V1ModelDirectory = Join-Path $FineTuneDir 'artifacts/v1-reference'

function Invoke-Compose {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & docker compose --project-name juris-check-finetuning --file $ComposeFile @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose falhou com código $LASTEXITCODE."
    }
}

function Assert-DockerAndGpu {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker CLI não encontrado.'
    }
    $dockerOs = (& docker info --format '{{.OSType}}' 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $dockerOs -ne 'linux') {
        throw 'Inicie o Docker Desktop configurado para contêineres Linux.'
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw 'nvidia-smi não está disponível no Windows.'
    }
    $freeOutput = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw 'Não foi possível consultar a memória livre da GPU NVIDIA.'
    }
    $freeMiB = [int](($freeOutput | Select-Object -First 1).ToString().Trim())
    if ($freeMiB -lt 8192) {
        throw "A GPU tem $freeMiB MiB livres. Encerre as outras execuções e libere pelo menos 8192 MiB antes de iniciar."
    }
}

function Initialize-ComposeEnvironment {
    param([string]$CurrentRun)
    $env:REPO_ROOT = $RepoRoot
    $env:RUNS_HOST_DIR = $RunsRoot
    $env:MODEL_HOST_DIR = Join-Path $RepoRoot 'models/ner_legal_large'
    $env:FT_V1_MODEL = '/source/v1'
    $env:FT_V1_HOST_DIR = $V1ModelDirectory
    $gitCommit = (& git -C $RepoRoot rev-parse HEAD 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) { $env:FT_GIT_COMMIT = $gitCommit } else { $env:FT_GIT_COMMIT = 'unknown' }
    $env:FT_DOCKER_IMAGE_ID = 'unknown'
    $env:FT_GIT_WORKTREE_SHA256 = $null
    if ([string]::IsNullOrWhiteSpace($CurrentRun)) {
        $env:OFFICIAL_TRAIN_HOST_DIR = Join-Path $RunsRoot '_unused_official_subset'
    } else {
        $env:OFFICIAL_TRAIN_HOST_DIR = Join-Path $RunsRoot "$CurrentRun\sources\official"
    }
}

function Invoke-Score {
    param(
        [string]$CurrentRun,
        [string]$Name,
        [string]$Split,
        [string]$HostModelDirectory
    )
    $env:MODEL_HOST_DIR = (Resolve-Path -LiteralPath $HostModelDirectory).Path
    if ($Split -match '^fold_(\d{2})$') {
        $goldHost = "$RunsRoot\$CurrentRun\prepared\folds\fold_$($Matches[1])\gold.csv"
        $goldContainer = "/runs/$CurrentRun/prepared/folds/fold_$($Matches[1])/gold.csv"
        $textContainer = "/runs/$CurrentRun/prepared/folds/fold_$($Matches[1])/txt"
    } else {
        $goldHost = "$RunsRoot\$CurrentRun\prepared\local_$Split.csv"
        $goldContainer = "/runs/$CurrentRun/prepared/local_$Split.csv"
        $textContainer = "/runs/$CurrentRun/prepared/local/$Split/txt"
    }
    if (-not (Test-Path -LiteralPath $goldHost -PathType Leaf)) {
        throw "Gabarito local preparado ausente: $goldHost"
    }
    $metricHost = "$RunsRoot\$CurrentRun\metrics\${Name}_$Split.json"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $metricHost) | Out-Null
    Invoke-Compose -Arguments @('run', '--rm', '--entrypoint', 'python', 'evaluator',
        '/opt/fine-tuning/evaluate_model.py', '--mode', 'local', '--run-dir', "/runs/$CurrentRun",
        '--name', $Name, '--input-dir', $textContainer, '--golden', $goldContainer,
        '--report', "/runs/$CurrentRun/metrics/${Name}_$Split.json")
}

function Invoke-FinalTrainingWorkflow {
    param([string]$CurrentRun, [switch]$IsResume)
    Assert-DockerAndGpu
    New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
    Initialize-ComposeEnvironment -CurrentRun $CurrentRun
    Invoke-Compose -Arguments @('build', 'trainer', 'evaluator')
    $imageId = (& docker image inspect juris-check-finetuning-trainer:local --format '{{.Id}}' 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and $imageId) { $env:FT_DOCKER_IMAGE_ID = $imageId }

    $runDir = Join-Path $RunsRoot $CurrentRun
    if (-not $IsResume) {
        if (Test-Path -LiteralPath $runDir) {
            if (-not (Test-Path -LiteralPath (Join-Path $runDir 'prepared/dataset_manifest.json') -PathType Leaf)) {
                throw "A execução já existe sem preparação completa: $runDir. Escolha um novo run-id."
            }
            $savedCheckpoints = @(Get-ChildItem -LiteralPath (Join-Path $runDir 'experiments') -Directory -Filter 'checkpoint-*' -Recurse -ErrorAction SilentlyContinue)
            if ($savedCheckpoints.Count -gt 0) {
                throw "A execução já tem checkpoints retomáveis: $runDir. Use resume com um checkpoint completo."
            }
        } else {
            New-Item -ItemType Directory -Path $runDir | Out-Null
        }
    } elseif (-not (Test-Path -LiteralPath (Join-Path $runDir 'prepared/dataset_manifest.json'))) {
        throw "A execução $CurrentRun não tem dados preparados para retomada."
    }
    Invoke-Compose -Arguments @('run', '--rm', 'stager', '--run-id', $CurrentRun)
    if (-not $IsResume) {
        if (-not (Test-Path -LiteralPath (Join-Path $runDir 'campaign_config.resolved.json') -PathType Leaf)) {
            Invoke-Compose -Arguments @('run', '--rm', 'trainer', 'prepare', '--run-id', $CurrentRun)
        }
    } else {
        if ([string]::IsNullOrWhiteSpace($Experiment) -or [string]::IsNullOrWhiteSpace($CheckpointPath)) {
            throw 'resume exige -Experiment e -CheckpointPath para um checkpoint de Trainer completo.'
        }
        $checkpointFull = (Resolve-Path -LiteralPath $CheckpointPath).Path
        $experimentRoot = (Resolve-Path -LiteralPath (Join-Path $runDir "experiments\$Experiment")).Path
        if (-not $checkpointFull.StartsWith($experimentRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'O checkpoint de retomada precisa estar no diretório do experimento selecionado.'
        }
        foreach ($stateFile in @('optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth', 'scaler.pt')) {
            if (-not (Test-Path -LiteralPath (Join-Path $checkpointFull $stateFile) -PathType Leaf)) {
                throw "Checkpoint sem estado completo para retomada: falta $stateFile."
            }
        }
        $checkpointInContainer = "/runs/$CurrentRun/experiments/$Experiment/$((Split-Path -Leaf $checkpointFull))"
        Invoke-Compose -Arguments @('run', '--rm', 'trainer', 'resume', '--run-id', $CurrentRun,
            '--experiment', $Experiment, '--checkpoint', $checkpointInContainer)
    }

    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'metrics/baseline_validation.json') -PathType Leaf)) {
        Invoke-Score -CurrentRun $CurrentRun -Name 'baseline' -Split 'validation' -HostModelDirectory (Join-Path $RepoRoot 'models/ner_legal_large')
    }
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'metrics/baseline_v1_validation.json') -PathType Leaf)) {
        Invoke-Score -CurrentRun $CurrentRun -Name 'baseline_v1' -Split 'validation' -HostModelDirectory $V1ModelDirectory
    }

    if (-not $IsResume) {
        Invoke-Compose -Arguments @('run', '--rm', 'trainer', 'train', '--run-id', $CurrentRun)
    }

    $candidateNames = @('original_lr_5e-06', 'original_lr_1e-05', 'original_lr_2e-05', 'original_lr_3e-05',
        'v1_lr_5e-06', 'v1_lr_1e-05', 'v1_lr_2e-05', 'v1_lr_3e-05')
    $evaluatedCount = 0
    foreach ($candidate in $candidateNames) {
        $modelDirectory = Join-Path $runDir "candidates\$candidate"
        if (Test-Path -LiteralPath (Join-Path $modelDirectory 'model.safetensors')) {
            Invoke-Score -CurrentRun $CurrentRun -Name $candidate -Split 'validation' -HostModelDirectory $modelDirectory
            $evaluatedCount++
        }
    }
    if ($evaluatedCount -eq 0) { throw 'Nenhum checkpoint completo foi produzido dentro do orçamento de treino.' }

    Invoke-Compose -Arguments @('run', '--rm', '-e', 'CUDA_VISIBLE_DEVICES=', 'trainer', 'select', '--run-id', $CurrentRun)
    $selection = Get-Content -LiteralPath (Join-Path $runDir 'selection.json') -Raw | ConvertFrom-Json
    $selectedExperiment = [string]$selection.selected_experiment
    $selectedModel = Join-Path $runDir "candidates\$selectedExperiment"

    $selectedTestPath = Join-Path $runDir 'metrics/selected_test.json'
    if (-not (Test-Path -LiteralPath $selectedTestPath)) {
        Invoke-Score -CurrentRun $CurrentRun -Name 'selected_test' -Split 'test' -HostModelDirectory $selectedModel
        $renameFrom = Join-Path $runDir 'metrics/selected_test_test.json'
        if (Test-Path -LiteralPath $renameFrom) {
            Move-Item -LiteralPath $renameFrom -Destination $selectedTestPath -Force
        }
        $spanFrom = Join-Path $runDir 'metrics/selected_test_test_spans.json'
        $spanTo = Join-Path $runDir 'metrics/selected_test_spans.json'
        if (Test-Path -LiteralPath $spanFrom) { Move-Item -LiteralPath $spanFrom -Destination $spanTo -Force }
    }

    Invoke-Compose -Arguments @('run', '--rm', '-e', 'CUDA_VISIBLE_DEVICES=', 'trainer', 'finalize',
        '--run-id', $CurrentRun, '--experiment', $selectedExperiment)
    $env:MODEL_HOST_DIR = (Resolve-Path -LiteralPath $selectedModel).Path
    Invoke-Compose -Arguments @('run', '--rm', '--entrypoint', 'python', 'evaluator',
        '/opt/fine-tuning/validate_export.py', '/app/models/ner_legal_large',
        '--report', "/runs/$CurrentRun/offline_export_validation.json")
    Invoke-Compose -Arguments @('run', '--rm', '-e', 'CUDA_VISIBLE_DEVICES=', 'trainer',
        'record-validation', '--run-id', $CurrentRun,
        '--report', "/runs/$CurrentRun/offline_export_validation.json")

    $v1Model = $V1ModelDirectory
    $officialReports = @{}
    foreach ($modelName in @('original', 'v1', 'v2')) {
        switch ($modelName) {
            'original' { $officialModel = Join-Path $RepoRoot 'models/ner_legal_large' }
            'v1' { $officialModel = $v1Model }
            'v2' { $officialModel = $selectedModel }
        }
        $env:MODEL_HOST_DIR = (Resolve-Path -LiteralPath $officialModel).Path
        $reportPath = "/runs/$CurrentRun/metrics/official_$modelName.json"
        $officialReports[$modelName] = $reportPath
        if (-not (Test-Path -LiteralPath (Join-Path $runDir "metrics/official_$modelName.json") -PathType Leaf)) {
            Invoke-Compose -Arguments @('--profile', 'final-evaluation', 'run', '--rm', '--entrypoint', 'python', 'official-evaluator',
                '/opt/fine-tuning/evaluate_model.py', '--mode', 'official', '--run-dir', "/runs/$CurrentRun",
                '--name', "official_$modelName", '--report', $reportPath)
        }
    }
    Invoke-Compose -Arguments @('run', '--rm', '-e', 'CUDA_VISIBLE_DEVICES=', '--entrypoint', 'python', 'trainer',
        '-m', 'train.cli', 'compare-official', '--run-id', $CurrentRun,
        '--original', $officialReports.original, '--v1', $officialReports.v1, '--v2', $officialReports.v2)
    Invoke-Compose -Arguments @('run', '--rm', '-e', 'CUDA_VISIBLE_DEVICES=', '--entrypoint', 'python', 'trainer',
        '-m', 'train.cli', 'record-official-evaluation', '--run-id', $CurrentRun,
        '--checkpoint-dir', "/runs/$CurrentRun/candidates/$selectedExperiment",
        '--report', $officialReports.v2, '--comparison', "/runs/$CurrentRun/official_comparison.json")
    Write-Output "Pacote Hugging Face e seleção local: $selectedModel"
    Write-Output "Relatório final: $(Join-Path $selectedModel 'model_report.json')"
    Write-Output "Comparação oficial congelada: $(Join-Path $selectedModel 'official_comparison.json')"
}

function Invoke-TrainingCli {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [switch]$CpuOnly)
    $composeArguments = @('run', '--rm')
    if ($CpuOnly) { $composeArguments += @('-e', 'CUDA_VISIBLE_DEVICES=') }
    $composeArguments += @('--entrypoint', 'python', 'trainer', '-m', 'train.cli')
    $composeArguments += $Arguments
    Invoke-Compose -Arguments $composeArguments
}

function Get-ExperimentQueue {
    param([Parameter(Mandatory = $true)][string]$CurrentRun)
    $queuePath = Join-Path $RunsRoot "$CurrentRun/experiment_queue.json"
    if (-not (Test-Path -LiteralPath $queuePath -PathType Leaf)) {
        Invoke-TrainingCli -Arguments @('init-queue', '--run-id', $CurrentRun) -CpuOnly | Out-Null
    }
    return (Get-Content -LiteralPath $queuePath -Raw | ConvertFrom-Json).experiments
}

function Invoke-ExperimentStage {
    param([Parameter(Mandatory = $true)][string]$CurrentRun, [Parameter(Mandatory = $true)][string]$Stage)
    $runDir = Join-Path $RunsRoot $CurrentRun
    $queue = Get-ExperimentQueue -CurrentRun $CurrentRun
    foreach ($experiment in $queue) {
        if ([string]$experiment.stage -ne $Stage) { continue }
        $name = [string]$experiment.name
        $experimentDir = Join-Path $runDir "experiments\$name"
        $candidateDir = Join-Path $runDir "candidates\$name"
        if ($null -ne $experiment.fold) { $split = 'fold_{0:D2}' -f [int]$experiment.fold } else { $split = 'validation' }
        $selectedMarker = Join-Path $candidateDir 'checkpoint_selection.json'
        $selectedMetric = Join-Path $runDir "metrics\${name}_${split}.json"
        if ((Test-Path -LiteralPath $selectedMarker -PathType Leaf) -and (Test-Path -LiteralPath $selectedMetric -PathType Leaf)) {
            continue
        }
        $manifestPath = Join-Path $experimentDir 'experiment_manifest.json'
        $candidateWeights = Join-Path $candidateDir 'model.safetensors'
        if (-not (Test-Path -LiteralPath $candidateWeights -PathType Leaf)) {
            if ((Test-Path -LiteralPath $manifestPath -PathType Leaf) -and
                (Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json).status -eq 'interrupted_or_failed') {
                throw "O experimento $name foi interrompido. Retome-o com -Mode resume e depois rode -Mode campaign."
            }
            Invoke-TrainingCli -Arguments @('train', '--run-id', $CurrentRun, '--experiment', $name)
        }

        $snapshotRoot = Join-Path $experimentDir 'evaluation_checkpoints'
        $snapshots = @(Get-ChildItem -LiteralPath $snapshotRoot -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue)
        if ($snapshots.Count -eq 0) {
            Invoke-TrainingCli -Arguments @('import-checkpoints', '--run-id', $CurrentRun, '--experiment', $name) -CpuOnly
            $snapshots = @(Get-ChildItem -LiteralPath $snapshotRoot -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue)
        }
        if ($snapshots.Count -eq 0) { throw "O experimento $name não tem snapshots avaliáveis." }

        foreach ($snapshot in $snapshots) {
            $step = $snapshot.Name -replace '^checkpoint-', ''
            $snapshotName = "snapshot_${name}_step_${step}"
            $metricPath = Join-Path $runDir "metrics\${snapshotName}_${split}.json"
            if (-not (Test-Path -LiteralPath $metricPath -PathType Leaf)) {
                Invoke-Score -CurrentRun $CurrentRun -Name $snapshotName -Split $split -HostModelDirectory $snapshot.FullName
            }
        }
        Invoke-TrainingCli -Arguments @('select-experiment-checkpoint', '--run-id', $CurrentRun, '--experiment', $name) -CpuOnly
        Write-Output "Checkpoint selecionado pela métrica completa: $name"
    }
}

function Invoke-IndependentCampaign {
    param([Parameter(Mandatory = $true)][string]$CurrentRun)
    Assert-DockerAndGpu
    New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
    Initialize-ComposeEnvironment -CurrentRun $CurrentRun
    Invoke-Compose -Arguments @('build', 'trainer', 'evaluator')
    $imageId = (& docker image inspect juris-check-finetuning-trainer:local --format '{{.Id}}' 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and $imageId) { $env:FT_DOCKER_IMAGE_ID = $imageId }
    $runDir = Join-Path $RunsRoot $CurrentRun
    if (-not (Test-Path -LiteralPath $runDir)) { New-Item -ItemType Directory -Path $runDir | Out-Null }
    Invoke-Compose -Arguments @('run', '--rm', 'stager', '--run-id', $CurrentRun)
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'prepared/dataset_manifest.json') -PathType Leaf)) {
        Invoke-TrainingCli -Arguments @('prepare', '--run-id', $CurrentRun)
    }
    Invoke-TrainingCli -Arguments @('init-queue', '--run-id', $CurrentRun) -CpuOnly
    Invoke-TrainingCli -Arguments @('repair-snapshots', '--run-id', $CurrentRun) -CpuOnly

    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'metrics/baseline_validation.json') -PathType Leaf)) {
        Invoke-Score -CurrentRun $CurrentRun -Name 'baseline' -Split 'validation' -HostModelDirectory (Join-Path $RepoRoot 'models/ner_legal_large')
    }
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'metrics/baseline_v1_validation.json') -PathType Leaf)) {
        Invoke-Score -CurrentRun $CurrentRun -Name 'baseline_v1' -Split 'validation' -HostModelDirectory $V1ModelDirectory
    }

    Invoke-ExperimentStage -CurrentRun $CurrentRun -Stage 'base'
    Invoke-TrainingCli -Arguments @('extend-stage', '--run-id', $CurrentRun, '--stage', 'ablation') -CpuOnly
    Invoke-ExperimentStage -CurrentRun $CurrentRun -Stage 'ablation'
    Invoke-TrainingCli -Arguments @('prepare-augmentations', '--run-id', $CurrentRun)
    Invoke-TrainingCli -Arguments @('extend-stage', '--run-id', $CurrentRun, '--stage', 'robustness') -CpuOnly
    Invoke-ExperimentStage -CurrentRun $CurrentRun -Stage 'robustness'
    Invoke-TrainingCli -Arguments @('extend-stage', '--run-id', $CurrentRun, '--stage', 'stability') -CpuOnly
    Invoke-ExperimentStage -CurrentRun $CurrentRun -Stage 'stability'
    Invoke-TrainingCli -Arguments @('prepare-folds', '--run-id', $CurrentRun) -CpuOnly
    Invoke-TrainingCli -Arguments @('extend-stage', '--run-id', $CurrentRun, '--stage', 'cross_validation') -CpuOnly
    Invoke-ExperimentStage -CurrentRun $CurrentRun -Stage 'cross_validation'
    Invoke-TrainingCli -Arguments @('summarize-folds', '--run-id', $CurrentRun) -CpuOnly

    Invoke-TrainingCli -Arguments @('select', '--run-id', $CurrentRun) -CpuOnly
    $selection = Get-Content -LiteralPath (Join-Path $runDir 'selection.json') -Raw | ConvertFrom-Json
    $selectedExperiment = [string]$selection.selected_experiment
    $selectedModel = Join-Path $runDir "candidates\$selectedExperiment"
    $selectedTestPath = Join-Path $runDir 'metrics/selected_test.json'
    if (-not (Test-Path -LiteralPath $selectedTestPath)) {
        Invoke-Score -CurrentRun $CurrentRun -Name 'selected_test' -Split 'test' -HostModelDirectory $selectedModel
        $renameFrom = Join-Path $runDir 'metrics/selected_test_test.json'
        if (Test-Path -LiteralPath $renameFrom) { Move-Item -LiteralPath $renameFrom -Destination $selectedTestPath -Force }
        $spanFrom = Join-Path $runDir 'metrics/selected_test_test_spans.json'
        $spanTo = Join-Path $runDir 'metrics/selected_test_spans.json'
        if (Test-Path -LiteralPath $spanFrom) { Move-Item -LiteralPath $spanFrom -Destination $spanTo -Force }
    }
    Invoke-TrainingCli -Arguments @('finalize', '--run-id', $CurrentRun, '--experiment', $selectedExperiment) -CpuOnly
    $env:MODEL_HOST_DIR = (Resolve-Path -LiteralPath $selectedModel).Path
    Invoke-Compose -Arguments @('run', '--rm', '--entrypoint', 'python', 'evaluator',
        '/opt/fine-tuning/validate_export.py', '/app/models/ner_legal_large',
        '--report', "/runs/$CurrentRun/offline_export_validation.json")
    Invoke-TrainingCli -Arguments @('record-validation', '--run-id', $CurrentRun,
        '--report', "/runs/$CurrentRun/offline_export_validation.json") -CpuOnly

    Invoke-FinalOfficialEvaluation -CurrentRun $CurrentRun
    Write-Output "Campanha finalizada sem alterar o comando da banca: $selectedModel"
}

function Invoke-FinalOfficialEvaluation {
    param([Parameter(Mandatory = $true)][string]$CurrentRun, [switch]$BuildImages)
    Assert-DockerAndGpu
    Initialize-ComposeEnvironment -CurrentRun $CurrentRun
    if ($BuildImages) { Invoke-Compose -Arguments @('build', 'trainer', 'evaluator') }
    $runDir = Join-Path $RunsRoot $CurrentRun
    $selection = Get-Content -LiteralPath (Join-Path $runDir 'selection.json') -Raw | ConvertFrom-Json
    $selectedExperiment = [string]$selection.selected_experiment
    $selectedModel = (Resolve-Path -LiteralPath (Join-Path $runDir "candidates/$selectedExperiment")).Path
    $finalExport = Get-Content -LiteralPath (Join-Path $runDir 'final_export.json') -Raw | ConvertFrom-Json
    $modelSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $selectedModel 'model.safetensors')).Hash.ToLowerInvariant()
    if (-not $finalExport.validated_offline -or $modelSha -ne [string]$selection.selected_checkpoint_sha256 -or
        $modelSha -ne [string]$finalExport.model_weights_sha256 -or
        [string]$finalExport.checkpoint_dir -ne "/runs/$CurrentRun/candidates/$selectedExperiment") {
        throw 'O checkpoint local congelado não corresponde à seleção e à exportação validada.'
    }

    $officialReports = @{}
    foreach ($modelName in @('original', 'v1', 'v2')) {
        switch ($modelName) {
            'original' { $officialModel = Join-Path $RepoRoot 'models/ner_legal_large' }
            'v1' { $officialModel = $V1ModelDirectory }
            'v2' { $officialModel = $selectedModel }
        }
        $env:MODEL_HOST_DIR = (Resolve-Path -LiteralPath $officialModel).Path
        $reportPath = "/runs/$CurrentRun/metrics/official_$modelName.json"
        $officialReports[$modelName] = $reportPath
        Invoke-Compose -Arguments @('--profile', 'final-evaluation', 'run', '--rm', '--entrypoint', 'python', 'official-evaluator',
            '/opt/fine-tuning/evaluate_model.py', '--mode', 'official', '--run-dir', "/runs/$CurrentRun",
            '--name', "official_$modelName", '--report', $reportPath)
    }
    Invoke-TrainingCli -Arguments @('compare-official', '--run-id', $CurrentRun,
        '--original', $officialReports.original, '--v1', $officialReports.v1, '--v2', $officialReports.v2) -CpuOnly
    Invoke-TrainingCli -Arguments @('record-official-evaluation', '--run-id', $CurrentRun,
        '--checkpoint-dir', "/runs/$CurrentRun/candidates/$selectedExperiment",
        '--report', $officialReports.v2, '--comparison', "/runs/$CurrentRun/official_comparison.json") -CpuOnly
    $env:MODEL_HOST_DIR = $selectedModel
    Invoke-Compose -Arguments @('run', '--rm', '--entrypoint', 'python', 'evaluator',
        '/opt/fine-tuning/validate_export.py', '/app/models/ner_legal_large',
        '--report', "/runs/$CurrentRun/offline_export_validation_final.json")
    Invoke-TrainingCli -Arguments @('record-validation', '--run-id', $CurrentRun,
        '--report', "/runs/$CurrentRun/offline_export_validation_final.json") -CpuOnly
    Write-Output "Comparação oficial finalizada e pacote validado: $selectedModel"
}

try {
    switch ($Mode) {
        'train' {
            if ([string]::IsNullOrWhiteSpace($RunId)) { $RunId = 'ft-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
            Invoke-IndependentCampaign -CurrentRun $RunId
        }
        'campaign' {
            if ([string]::IsNullOrWhiteSpace($RunId)) { throw 'campaign exige -RunId.' }
            Invoke-IndependentCampaign -CurrentRun $RunId
        }
        'evaluate-official' {
            if ([string]::IsNullOrWhiteSpace($RunId)) { throw 'evaluate-official exige -RunId.' }
            Invoke-FinalOfficialEvaluation -CurrentRun $RunId -BuildImages
        }
        'resume' {
            if ([string]::IsNullOrWhiteSpace($RunId)) { throw 'resume exige -RunId.' }
            if ([string]::IsNullOrWhiteSpace($Experiment) -or [string]::IsNullOrWhiteSpace($CheckpointPath)) {
                throw 'resume exige -Experiment e -CheckpointPath.'
            }
            Assert-DockerAndGpu
            Initialize-ComposeEnvironment -CurrentRun $RunId
            Invoke-Compose -Arguments @('build', 'trainer', 'evaluator')
            $runDir = Join-Path $RunsRoot $RunId
            $checkpointFull = (Resolve-Path -LiteralPath $CheckpointPath).Path
            $experimentRoot = (Resolve-Path -LiteralPath (Join-Path $runDir "experiments\$Experiment")).Path
            if (-not $checkpointFull.StartsWith($experimentRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'O checkpoint precisa pertencer ao experimento indicado.'
            }
            $checkpointRelative = [System.IO.Path]::GetRelativePath($runDir, $checkpointFull).Replace('\', '/')
            Invoke-TrainingCli -Arguments @('resume', '--run-id', $RunId, '--experiment', $Experiment,
                '--checkpoint', "/runs/$RunId/$checkpointRelative")
            Invoke-IndependentCampaign -CurrentRun $RunId
        }
        'predict' {
            Assert-DockerAndGpu
            if ([string]::IsNullOrWhiteSpace($CheckpointPath) -or [string]::IsNullOrWhiteSpace($InputDirectory) -or
                [string]::IsNullOrWhiteSpace($DatabasePath) -or [string]::IsNullOrWhiteSpace($OutputPath)) {
                throw 'predict exige -CheckpointPath, -InputDirectory, -DatabasePath e -OutputPath.'
            }
            $checkpointFull = (Resolve-Path -LiteralPath $CheckpointPath).Path
            $inputFull = (Resolve-Path -LiteralPath $InputDirectory).Path
            $databaseFull = (Resolve-Path -LiteralPath $DatabasePath).Path
            if (-not (Test-Path -LiteralPath $databaseFull -PathType Leaf)) { throw "Banco SQLite inválido: $databaseFull" }
            if (-not (Test-Path -LiteralPath $inputFull -PathType Container)) { throw "Diretório TXT inválido: $inputFull" }
            $outputFull = [System.IO.Path]::GetFullPath($OutputPath)
            if ([string]::Equals($outputFull, $databaseFull, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'A saída não pode sobrescrever o banco.' }
            $outputDirectory = [System.IO.Path]::GetDirectoryName($outputFull)
            $outputName = [System.IO.Path]::GetFileName($outputFull)
            if ([string]::IsNullOrWhiteSpace($outputName)) { throw 'OutputPath precisa identificar um arquivo CSV.' }
            if ([string]::Equals($outputDirectory.TrimEnd('\'), $inputFull.TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase) -or
                $outputDirectory.StartsWith($inputFull.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'A saída não pode ficar dentro do diretório de entrada.'
            }
            New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
            Initialize-ComposeEnvironment
            Invoke-Compose -Arguments @('build', 'evaluator')
            $image = 'juris-check-finetuning-evaluator:local'
            $arguments = @(
                'run', '--rm', '--gpus', 'all', '--network', 'none',
                '--mount', "type=bind,source=$inputFull,target=/input/txt,readonly",
                '--mount', "type=bind,source=$databaseFull,target=/input/reference.sqlite,readonly",
                '--mount', "type=bind,source=$outputDirectory,target=/output",
                '--mount', "type=bind,source=$checkpointFull,target=/app/models/ner_legal_large,readonly",
                '--env', 'PROTECTED_DATA_DIR=/input', '--env', "OUTPUT_FILE=$outputName",
                '--entrypoint', 'sh', $image, '-ec',
                'python /opt/fine-tuning/validate_export.py /app/models/ner_legal_large && exec python -m src.pipeline --input /input/txt --db /input/reference.sqlite --output /output --output-file "$OUTPUT_FILE"'
            )
            & docker @arguments
            if ($LASTEXITCODE -ne 0) { throw "Predição com checkpoint selecionado falhou ($LASTEXITCODE)." }
            Write-Output "Submissão gerada com o checkpoint selecionado: $outputFull"
        }
        'validate-export' {
            if ([string]::IsNullOrWhiteSpace($CheckpointPath)) { throw 'validate-export exige -CheckpointPath.' }
            $checkpointFull = (Resolve-Path -LiteralPath $CheckpointPath).Path
            if (-not (Test-Path -LiteralPath $checkpointFull -PathType Container)) { throw "Checkpoint inválido: $checkpointFull" }
            $env:CHECKPOINT_HOST_DIR = $checkpointFull
            Initialize-ComposeEnvironment
            Invoke-Compose -Arguments @('build', 'validator')
            Invoke-Compose -Arguments @('--profile', 'validation', 'run', '--rm', 'validator')
        }
        'upload' {
            if ([string]::IsNullOrWhiteSpace($CheckpointPath) -or [string]::IsNullOrWhiteSpace($RepoId)) {
                throw 'upload exige -CheckpointPath e -RepoId. Defina HF_TOKEN no ambiente.'
            }
            $env:CHECKPOINT_HOST_DIR = (Resolve-Path -LiteralPath $CheckpointPath).Path
            Initialize-ComposeEnvironment
            Invoke-Compose -Arguments @('build', 'publisher')
            $uploadArguments = @('--profile', 'publish', 'run', '--rm', 'publisher', '--repo-id', $RepoId)
            if ($Private) { $uploadArguments += '--private' }
            Invoke-Compose -Arguments $uploadArguments
        }
    }
} catch {
    [Console]::Error.WriteLine("Erro: $($_.Exception.Message)")
    exit 1
}
