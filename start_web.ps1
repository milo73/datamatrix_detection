<#
.SYNOPSIS
    PDF DataMatrix Scanner - locked-down Windows launcher.

.DESCRIPTION
    Bootstraps a per-user Python (no admin, no installer) and starts the
    Streamlit web UI. Designed for Windows machines where:
      - The user can run PowerShell but does NOT have admin rights.
      - Python is not installed and cannot be installed via MSI.
      - The execution policy may be Restricted/AllSigned.

    On first run this downloads the Python 3.11 embeddable distribution
    (~10 MB) from python.org and pip into a project-local "python-embed\"
    folder, then installs the project's requirements there. Subsequent
    runs reuse that folder and start instantly.

    For air-gapped machines, drop these two files into ".\offline\" and
    re-run; no network access will be attempted:
      - python-3.11.9-embed-amd64.zip
      - get-pip.py

.NOTES
    Double-click "start_web.cmd" to launch with the right execution-policy
    flags. Or run directly:
        powershell -NoProfile -ExecutionPolicy Bypass -File .\start_web.ps1
#>

#Requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$Reinstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$PythonVersion   = '3.11.9'
$PythonZipName   = "python-$PythonVersion-embed-amd64.zip"
$PythonZipUrl    = "https://www.python.org/ftp/python/$PythonVersion/$PythonZipName"
$GetPipUrl       = 'https://bootstrap.pypa.io/get-pip.py'
$EmbedDirName    = 'python-embed'
$OfflineDirName  = 'offline'
$AppEntryPoint   = 'app_web.py'
$RequirementsTxt = 'requirements.txt'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Header {
    param([string]$Text)
    Write-Host ''
    Write-Host '============================================' -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host '============================================' -ForegroundColor Cyan
}

function Write-Ok   { param([string]$M) Write-Host "[OK]   $M" -ForegroundColor Green }
function Write-Info { param([string]$M) Write-Host "[INFO] $M" -ForegroundColor Cyan }
function Write-Warn { param([string]$M) Write-Host "[WARN] $M" -ForegroundColor Yellow }
function Write-Fail { param([string]$M) Write-Host "[FAIL] $M" -ForegroundColor Red }

# Pip and other native tools write benign warnings to stderr (e.g. "script
# X is installed in '...' which is not on PATH"). With $ErrorActionPreference
# set to 'Stop', Windows PowerShell 5.1 promotes those into terminating
# NativeCommandError records even when the exit code is 0. Localize the
# preference around native calls and judge success purely by $LASTEXITCODE.
function Invoke-WithNativeErrorTolerance {
    param([Parameter(Mandatory)][scriptblock]$ScriptBlock)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $ScriptBlock } finally { $ErrorActionPreference = $prev }
}

function Stop-WithError {
    param([string]$Message, [int]$Code = 1)
    Write-Fail $Message
    Write-Host ''
    if ($Host.Name -eq 'ConsoleHost') {
        Write-Host 'Press any key to close this window...' -ForegroundColor Yellow
        try { [void][System.Console]::ReadKey($true) } catch { Read-Host | Out-Null }
    }
    exit $Code
}

function Test-Python {
    param([string]$Exe)
    if (-not (Test-Path -LiteralPath $Exe)) { return $false }
    try {
        $output = & $Exe --version 2>&1
        return ($LASTEXITCODE -eq 0 -and $output -match 'Python 3\.')
    } catch {
        return $false
    }
}

function Find-Python {
    param([string]$EmbedRoot)

    $embedExe = Join-Path $EmbedRoot 'python.exe'
    if (Test-Python $embedExe) { return $embedExe }

    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($cmd -and (Test-Python $cmd.Source)) { return $cmd.Source }

    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $resolved = & $py.Source -3 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Python $resolved)) {
                return $resolved.Trim()
            }
        } catch { }
    }

    $userInstalls = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    if (Test-Path -LiteralPath $userInstalls) {
        $hit = Get-ChildItem -LiteralPath $userInstalls -Directory -Filter 'Python3*' -ErrorAction SilentlyContinue |
               ForEach-Object { Join-Path $_.FullName 'python.exe' } |
               Where-Object { Test-Python $_ } |
               Select-Object -First 1
        if ($hit) { return $hit }
    }

    $storeExe = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\python.exe'
    if (Test-Python $storeExe) { return $storeExe }

    return $null
}

function Get-OfflineFile {
    param(
        [string]$OfflineDir,
        [string]$FileName
    )
    if (-not (Test-Path -LiteralPath $OfflineDir)) { return $null }
    $candidate = Join-Path $OfflineDir $FileName
    if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    return $null
}

function Get-RemoteFile {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Label
    )
    Write-Info "Downloading $Label from $Url"
    $oldPref = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing -TimeoutSec 120
    } catch {
        throw "Download failed for $Label ($Url): $($_.Exception.Message)"
    } finally {
        $ProgressPreference = $oldPref
    }
    if (-not (Test-Path -LiteralPath $Destination)) {
        throw "Download completed but file not found at $Destination"
    }
    Write-Ok "$Label downloaded ($([math]::Round((Get-Item $Destination).Length / 1MB, 2)) MB)"
}

function Initialize-EmbeddedPython {
    param(
        [string]$EmbedRoot,
        [string]$OfflineDir
    )

    if ((Test-Path -LiteralPath $EmbedRoot) -and -not $Reinstall) {
        $exe = Join-Path $EmbedRoot 'python.exe'
        if (Test-Python $exe) {
            Write-Ok "Embedded Python already present at $EmbedRoot"
            return $exe
        }
        Write-Warn "Embedded Python folder exists but is incomplete - rebuilding"
        Remove-Item -LiteralPath $EmbedRoot -Recurse -Force
    } elseif ((Test-Path -LiteralPath $EmbedRoot) -and $Reinstall) {
        Write-Info 'Reinstall flag set - removing existing python-embed\'
        Remove-Item -LiteralPath $EmbedRoot -Recurse -Force
    }

    Write-Header "Bootstrapping Python $PythonVersion (no admin required)"

    $offlineZip   = Get-OfflineFile -OfflineDir $OfflineDir -FileName $PythonZipName
    $offlineGetPip = Get-OfflineFile -OfflineDir $OfflineDir -FileName 'get-pip.py'

    $zipPath = Join-Path $PSScriptRoot $PythonZipName
    if ($offlineZip) {
        Write-Ok "Using offline Python archive: $offlineZip"
        Copy-Item -LiteralPath $offlineZip -Destination $zipPath -Force
    } else {
        try {
            Get-RemoteFile -Url $PythonZipUrl -Destination $zipPath -Label 'Python embeddable'
        } catch {
            Stop-WithError @"
Could not download Python from python.org.

Reason: $($_.Exception.Message)

If this machine has no internet access, drop these two files into:
  $OfflineDir\
    - $PythonZipName        (from $PythonZipUrl)
    - get-pip.py            (from $GetPipUrl)
Then re-run start_web.cmd.
"@
        }
    }

    Write-Info "Extracting Python to $EmbedRoot"
    Expand-Archive -LiteralPath $zipPath -DestinationPath $EmbedRoot -Force
    Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue

    # Embeddable Python disables `site` by default which prevents pip from
    # finding installed packages. Uncomment `import site` in pythonXY._pth.
    $pthFile = Get-ChildItem -LiteralPath $EmbedRoot -Filter 'python*._pth' -File |
               Select-Object -First 1
    if (-not $pthFile) {
        Stop-WithError "Could not find python*._pth in $EmbedRoot - extraction may be corrupt"
    }
    $pthContent = Get-Content -LiteralPath $pthFile.FullName -Raw
    $pthContent = $pthContent -replace '(?m)^\s*#\s*import\s+site\s*$', 'import site'
    if ($pthContent -notmatch '(?m)^\s*import\s+site\s*$') {
        $pthContent = $pthContent.TrimEnd() + "`r`nimport site`r`n"
    }
    Set-Content -LiteralPath $pthFile.FullName -Value $pthContent -Encoding ASCII
    Write-Ok "Enabled site-packages in $($pthFile.Name)"

    $pythonExe = Join-Path $EmbedRoot 'python.exe'
    if (-not (Test-Python $pythonExe)) {
        Stop-WithError "Embedded Python at $pythonExe failed to run"
    }

    $getPipPath = Join-Path $EmbedRoot 'get-pip.py'
    if ($offlineGetPip) {
        Write-Ok "Using offline get-pip.py: $offlineGetPip"
        Copy-Item -LiteralPath $offlineGetPip -Destination $getPipPath -Force
    } else {
        try {
            Get-RemoteFile -Url $GetPipUrl -Destination $getPipPath -Label 'get-pip.py'
        } catch {
            Stop-WithError @"
Could not download get-pip.py from $GetPipUrl

Reason: $($_.Exception.Message)

For offline machines: place get-pip.py in $OfflineDir\ and re-run.
"@
        }
    }

    Write-Info 'Installing pip into embedded Python'
    Invoke-WithNativeErrorTolerance {
        & $pythonExe $getPipPath --no-warn-script-location
    }
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError 'pip bootstrap failed - see output above'
    }
    Remove-Item -LiteralPath $getPipPath -Force -ErrorAction SilentlyContinue
    Write-Ok 'pip installed'

    return $pythonExe
}

function Install-Requirements {
    param([string]$Python)

    Write-Header 'Installing project dependencies'

    $logPath = Join-Path $PSScriptRoot 'pip_install.log'
    if (Test-Path -LiteralPath $logPath) { Remove-Item -LiteralPath $logPath -Force }

    Invoke-WithNativeErrorTolerance {
        & $Python -m pip install --upgrade pip setuptools --disable-pip-version-check 2>&1 |
            Tee-Object -FilePath $logPath -Append
    }
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "Failed to upgrade pip/setuptools - see $logPath"
    }

    $reqPath = Join-Path $PSScriptRoot $RequirementsTxt
    Invoke-WithNativeErrorTolerance {
        & $Python -m pip install -r $reqPath --disable-pip-version-check 2>&1 |
            Tee-Object -FilePath $logPath -Append
    }
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError @"
pip install failed for requirements.txt - see $logPath

If pylibdmtx or pyzbar fails to build, the most common fix is the
Visual C++ 2015-2022 Redistributable. On a locked-down machine your
IT team usually deploys it; download links:
  https://aka.ms/vs/17/release/vc_redist.x64.exe
  https://aka.ms/highdpimfc2013x64enu (older zbar DLL)
"@
    }
    Write-Ok 'All dependencies installed'
}

function Test-CriticalImports {
    param([string]$Python)

    Write-Header 'Verifying critical packages'

    $packages = @(
        @{ Module = 'setuptools';  Hint = 'pip install setuptools' },
        @{ Module = 'pylibdmtx';   Hint = 'Needs Visual C++ 2015-2022 Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe)' },
        @{ Module = 'pyzbar';      Hint = 'Needs Visual C++ Redistributable - try https://aka.ms/highdpimfc2013x64enu for libzbar-64.dll' },
        @{ Module = 'fitz';        Hint = 'pip install PyMuPDF' },
        @{ Module = 'cv2';         Hint = 'pip install opencv-python' },
        @{ Module = 'streamlit';   Hint = 'pip install streamlit' }
    )

    $failed = @()
    foreach ($pkg in $packages) {
        & $Python -c "import $($pkg.Module)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok $pkg.Module
        } else {
            Write-Fail "$($pkg.Module) - $($pkg.Hint)"
            $failed += $pkg.Module
        }
    }

    if ($failed.Count -gt 0) {
        Stop-WithError "One or more critical imports failed: $($failed -join ', ')"
    }
}

# ---------------------------------------------------------------------------
# Self-relaunch with bypassed execution policy if needed (process scope only,
# no admin required and machine policy is untouched).
# ---------------------------------------------------------------------------
$currentPolicy = Get-ExecutionPolicy -Scope Process
if ($currentPolicy -in @('Restricted', 'AllSigned', 'Undefined') -and
    -not $env:DM_PS_RELAUNCHED) {
    $env:DM_PS_RELAUNCHED = '1'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath @PSBoundParameters
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Push-Location $PSScriptRoot
try {
    Write-Header 'PDF DataMatrix & QR Code Scanner'
    Write-Host '  Locked-down Windows launcher (no admin required)'
    Write-Host ''

    $embedRoot   = Join-Path $PSScriptRoot $EmbedDirName
    $offlineDir  = Join-Path $PSScriptRoot $OfflineDirName

    $python = if ($Reinstall) {
        Initialize-EmbeddedPython -EmbedRoot $embedRoot -OfflineDir $offlineDir
    } else {
        $found = Find-Python -EmbedRoot $embedRoot
        if ($found) {
            Write-Ok "Using Python: $found"
            $found
        } else {
            Write-Info 'No Python installation found - will bootstrap embedded Python'
            Initialize-EmbeddedPython -EmbedRoot $embedRoot -OfflineDir $offlineDir
        }
    }

    Invoke-WithNativeErrorTolerance {
        $version = & $python --version 2>&1
        Write-Ok "$version"
    }

    Install-Requirements -Python $python
    Test-CriticalImports -Python $python

    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $AppEntryPoint))) {
        Stop-WithError "$AppEntryPoint not found in $PSScriptRoot - run this script from the project folder"
    }

    Write-Header 'All checks passed - starting web app'
    Write-Host '  The browser will open automatically at http://localhost:8501'
    Write-Host '  Press Ctrl+C in this window to stop the server.'
    Write-Host ''

    # Streamlit writes startup banner and ongoing logs to stderr. Without
    # this wrapper the first stderr line would terminate the launcher.
    $appPath = Join-Path $PSScriptRoot $AppEntryPoint
    Invoke-WithNativeErrorTolerance {
        & $python -m streamlit run $appPath
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
