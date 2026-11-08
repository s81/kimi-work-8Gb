# setup-windows.ps1 - one command from a fresh clone to a built, tested k3.exe on Windows.
#
# Usage, from the repository root (or double-click setup.bat):
#   setup.bat                 build the toolchain if needed, build k3, run the tests
#   setup.bat -SkipTests      build only
#
# WHAT IT DOES
#   1. Downloads the official Tiny C Compiler 0.9.27 for 64-bit Windows and checks its
#      SHA-256. This is only the bootstrap compiler.
#   2. Downloads TinyCC's source at the pinned commit (the 0.9.28rc this project is
#      tested with) as a zip, no git needed, and checks a SHA-256 over every file it
#      contains, so a changed or truncated download is refused.
#   3. Builds TCC 0.9.28rc with TCC 0.9.27, using TinyCC's own build-tcc.bat.
#   4. Builds bin\k3.exe with `tcc -run build.c`, then runs `tcc -run build.c test`.
#
# Everything goes into .tcc\ inside this repository (about 30 MB); nothing is installed
# system-wide and PATH is not changed. Re-running skips the toolchain once it exists.
# Needs Windows 10 or later (PowerShell 5.1 and TLS 1.2), and internet access to
# download.savannah.gnu.org and repo.or.cz the first time.
#
# Why not the official 0.9.27 compiler alone: it miscompiles negated floating-point
# comparisons with NaN, which breaks the engine's NaN guards (docs/BUILD-WINDOWS.md).
# CI does the same steps in .github/actions/setup-tcc, with git instead of the zip.

param([switch]$SkipTests)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # Invoke-WebRequest is very slow with it on
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$TinyccCommit = 'e23dabed154a8495f3ac1ff3cc9d04fa1c207893'   # 2026-10-16, 0.9.28rc
$BootUrl      = 'https://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27-win64-bin.zip'
$BootSha256   = '34a721949a2583fdff725312da092fa0f5f1f284b702e6f811c6954714faabb2'
$SrcUrl       = "https://repo.or.cz/tinycc.git/snapshot/$TinyccCommit.zip"
# SHA-256 of the lines "<sha256 of file>  <path>\n" for every file in the commit, sorted
# by path. Computed from a git checkout of the commit; independent of how the zip is
# packed, so a re-generated archive with the same content still verifies.
$SrcTreeSha256 = 'fddc5b8f03367f9d510d8c7f5c7058f8a49f2409259a4d43980ac3d626043210'

$Root  = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root '.tcc'
$Tcc   = Join-Path $Tools 'tinycc\win32\tcc.exe'

function Step($msg) { Write-Host ""; Write-Host "== $msg" }

# curl.exe ships with Windows 10 (1803) and later. It is preferred because repo.or.cz
# puts a "are you a bot?" web page in front of browser-like clients, and PowerShell's
# Invoke-WebRequest identifies as one; curl's user agent gets the file. The fallback
# sends the same kind of user agent.
function Get-Download($url, $out) {
    Write-Host "   downloading $url"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -fsSL --retry 3 -o $out $url
        if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE): $url" }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -UserAgent 'curl/8.4.0'
    }
}

# A zip starts with "PK". Anything else is usually a web page served in its place.
function Assert-Zip($path, $url) {
    $fs = [System.IO.File]::OpenRead($path)
    try { $b0 = $fs.ReadByte(); $b1 = $fs.ReadByte() } finally { $fs.Close() }
    if ($b0 -ne 0x50 -or $b1 -ne 0x4B) {
        throw "$url did not return a zip archive (the server probably sent a web page, such as a bot check). Try again later, or build TCC by hand as in docs/BUILD-WINDOWS.md."
    }
}

function Get-TreeSha256($dir) {
    $base  = (Resolve-Path $dir).Path.TrimEnd('\') + '\'
    $files = Get-ChildItem -LiteralPath $dir -Recurse -File -Force
    $rows  = @{}
    foreach ($f in $files) {
        $rel = $f.FullName.Substring($base.Length).Replace('\', '/')
        $rows[$rel] = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLower()
    }
    $keys = [string[]]$rows.Keys
    [Array]::Sort($keys, [StringComparer]::Ordinal)
    $sb = New-Object System.Text.StringBuilder
    foreach ($k in $keys) { [void]$sb.Append($rows[$k]).Append('  ').Append($k).Append("`n") }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $bytes = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($sb.ToString()))
    return ([BitConverter]::ToString($bytes) -replace '-', '').ToLower()
}

if (Test-Path $Tcc) {
    Step "TCC 0.9.28rc already built in .tcc\ (delete .tcc\ to rebuild it)"
} else {
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null

    Step "1/4  bootstrap compiler: official TCC 0.9.27"
    $bootZip = Join-Path $Tools 'tcc-0.9.27-win64-bin.zip'
    $boot    = Join-Path $Tools 'boot'
    Get-Download $BootUrl $bootZip
    Assert-Zip $bootZip $BootUrl
    $sha = (Get-FileHash -LiteralPath $bootZip -Algorithm SHA256).Hash.ToLower()
    if ($sha -ne $BootSha256) { throw "TCC 0.9.27 download has SHA-256 $sha, expected $BootSha256. Not using it." }
    Write-Host "   SHA-256 ok"
    if (Test-Path $boot) { Remove-Item -Recurse -Force $boot }
    Expand-Archive -LiteralPath $bootZip -DestinationPath $boot

    Step "2/4  TinyCC source at $($TinyccCommit.Substring(0, 8))"
    $srcZip = Join-Path $Tools 'tinycc-src.zip'
    $unpack = Join-Path $Tools 'src-unpack'
    $src    = Join-Path $Tools 'tinycc'
    Get-Download $SrcUrl $srcZip
    Assert-Zip $srcZip $SrcUrl
    if (Test-Path $unpack) { Remove-Item -Recurse -Force $unpack }
    Expand-Archive -LiteralPath $srcZip -DestinationPath $unpack
    $top = @(Get-ChildItem -LiteralPath $unpack -Directory)
    if ($top.Count -ne 1) { throw "unexpected layout in $srcZip" }
    $tree = Get-TreeSha256 $top[0].FullName
    if ($tree -ne $SrcTreeSha256) { throw "TinyCC source content hash $tree does not match $SrcTreeSha256. Not building it." }
    Write-Host "   content SHA-256 ok (every file of the pinned commit)"
    if (Test-Path $src) { Remove-Item -Recurse -Force $src }
    Move-Item -LiteralPath $top[0].FullName -Destination $src
    Remove-Item -Recurse -Force $unpack

    Step "3/4  building TCC 0.9.28rc with TCC 0.9.27"
    # 0.9.27's kernel32 import list predates calls the new TCC uses
    # (AddVectoredExceptionHandler among them); give the bootstrap the new lists.
    Copy-Item (Join-Path $src 'win32\lib\*.def') (Join-Path $boot 'tcc\lib') -Force
    Push-Location (Join-Path $src 'win32')
    try {
        cmd /c "build-tcc.bat -c `"$boot\tcc\tcc.exe`" -t x86_64"
        if ($LASTEXITCODE -ne 0) { throw "build-tcc.bat failed with exit code $LASTEXITCODE" }
    } finally { Pop-Location }
    if (-not (Test-Path $Tcc)) { throw "build-tcc.bat finished but $Tcc is missing" }
}

& $Tcc -v
if ($LASTEXITCODE -ne 0) { throw "the built tcc.exe does not run" }

Push-Location $Root
try {
    Step "4/4  building k3"
    & $Tcc -run build.c
    if ($LASTEXITCODE -ne 0) { throw "tcc -run build.c failed with exit code $LASTEXITCODE" }
    if (-not $SkipTests) {
        Step "testing (about a minute; no model weights needed)"
        & $Tcc -run build.c test
        if ($LASTEXITCODE -ne 0) { throw "tcc -run build.c test failed with exit code $LASTEXITCODE" }
    }
} finally { Pop-Location }

Write-Host ""
Write-Host "Done. The engine is $Root\bin\k3.exe"
Write-Host "  bin\k3.exe --help"
Write-Host "To use this TCC directly in this PowerShell window:"
Write-Host "  `$env:Path = `"$(Split-Path -Parent $Tcc);`$env:Path`""
