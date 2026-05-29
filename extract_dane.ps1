$rawDir  = "C:\Users\ma_va\Documents\Salary\RAW DATA"
$dataDir = "C:\Users\ma_va\Documents\Salary\DATA"

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

$meses = @{
    "enero" = 1; "febrero" = 2; "marzo" = 3; "abril" = 4; "mayo" = 5; "junio" = 6;
    "julio" = 7; "agosto" = 8; "septiembre" = 9; "octubre" = 10; "noviembre" = 11; "diciembre" = 12
}

$zipFiles = Get-ChildItem -Path $rawDir -Filter *.zip

foreach ($file in $zipFiles) {
    $mesNum = $null
    
    foreach ($mes in $meses.Keys) {
        if ($file.Name.ToLower() -match $mes) {
            $mesNum = $meses[$mes]
            break
        }
    }

    if ($null -eq $mesNum) {
        if ($file.Name -match '(\d{1,2})') {
            $mesNum = [int]$Matches[1]
        }
    }

    if ($null -eq $mesNum) {
        Write-Warning "Saltando archivo sin mes: $($file.Name)"
        continue
    }

    Write-Host "🛠️ Procesando mes $mesNum -> $($file.Name)..." -ForegroundColor Cyan

    # Seguros para extraer solo UNA VEZ por archivo por mes
    $extrCarac = $false
    $extrOcup  = $false
    $extrIngr  = $false

    try {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($file.FullName)

        foreach ($entry in $archive.Entries) {
            $nameLower = $entry.FullName.ToLower()

            # Bloqueador de archivos basura
            if ($nameLower -match "__macosx") { continue }

            # 1. Características Generales
            if (-not $extrCarac -and $nameLower -like "*csv/*aracter*sticas generales*.csv") {
                $targetPath = Join-Path $dataDir "Características generales_$mesNum.csv"
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetPath, $true)
                Write-Host "   -> Extraído: Características generales_$mesNum.csv" -ForegroundColor Green
                $extrCarac = $true
            }
            
            # 2. Ocupados
            elseif (-not $extrOcup -and $nameLower -like "*csv/*ocupados*.csv" -and $nameLower -notlike "*no ocupados*") {
                $targetPath = Join-Path $dataDir "Ocupados_$mesNum.csv"
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetPath, $true)
                Write-Host "   -> Extraído: Ocupados_$mesNum.csv" -ForegroundColor Yellow
                $extrOcup = $true
            }

            # 3. Otros ingresos
            elseif (-not $extrIngr -and $nameLower -like "*csv/*otros ingresos*.csv") {
                $targetPath = Join-Path $dataDir "Otros ingresos e impuestos_$mesNum.csv"
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetPath, $true)
                Write-Host "   -> Extraído: Otros ingresos e impuestos_$mesNum.csv" -ForegroundColor Green
                $extrIngr = $true
            }
        }
    } catch {
        Write-Host "Error en $($file.Name): $_" -ForegroundColor Red
    } finally {
        if ($archive) { $archive.Dispose() }
    }
}

Write-Host "`n¡Extracción finalizada! Revisa tu carpeta DATA." -ForegroundColor Green