$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pageUrl = 'http://127.0.0.1:4173/'

function Test-PageReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $pageUrl -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

try {
    if (-not (Test-PageReady)) {
        $powerShellExe = Join-Path $PSHOME 'powershell.exe'
        $serverScript = Join-Path $projectDir 'local-server.ps1'
        Start-Process -FilePath $powerShellExe -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $serverScript + '"')) -WorkingDirectory $projectDir -WindowStyle Hidden

        $ready = $false
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            Start-Sleep -Milliseconds 250
            if (Test-PageReady) {
                $ready = $true
                break
            }
        }
        if (-not $ready) {
            throw 'The local page could not start. Check whether port 4173 is in use.'
        }
    }

    Start-Process $pageUrl
} catch {
    Write-Host "Unable to open Wuqinxi: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
