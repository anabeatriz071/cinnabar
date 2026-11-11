# Compiles the synthetic DXR hardware fixture with the existing hash-pinned DXC archive.
param([string]$Root = "")
$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = Join-Path $PSScriptRoot "../.." }
$Root = (Resolve-Path $Root).Path
$out = Join-Path $Root ".local/dxr"
$compiler = Join-Path $out "compiler"
& (Join-Path $PSScriptRoot "fetch-dxc.ps1") -Root $Root -Out $compiler
$pin = Get-Content -Raw (Join-Path $PSScriptRoot "dxc.json") | ConvertFrom-Json
$archive = Join-Path $Root (".local/downloads/" + [IO.Path]::GetFileName(([Uri]$pin.archive_url).AbsolutePath))
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pin.archive_sha256) {
    throw "DXC fixture compiler archive hash mismatch"
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    foreach ($file in @("dxc.exe", "dxil.dll")) {
        $member = "$($pin.archive_directory)/$file"
        $entry = $zip.Entries | Where-Object { $_.FullName.Replace('\', '/') -eq $member } | Select-Object -First 1
        if (-not $entry) { throw "DXC archive is missing $member" }
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $compiler $file), $true)
    }
} finally { $zip.Dispose() }
& (Join-Path $compiler "dxc.exe") -T lib_6_3 -WX -Fo (Join-Path $out "native-smoke.dxil") (Join-Path $Root "crates/dxr-pipeline/tests/smoke.hlsl")
if ($LASTEXITCODE -ne 0) { throw "Native DXR fixture compilation failed ($LASTEXITCODE)" }

& (Join-Path $compiler "dxc.exe") -T lib_6_3 -WX -Fo (Join-Path $out "native-trace.dxil") (Join-Path $Root "crates/dxr-pipeline/tests/trace.hlsl")
if ($LASTEXITCODE -ne 0) { throw "Native TraceRay fixture compilation failed ($LASTEXITCODE)" }

# Keep identical DXIL mtimes stable so restoring the Cargo cache does not force a rebuild.
$terrain = Join-Path $Root "target/native-dxr/terrain.dxil"
New-Item -ItemType Directory -Force (Split-Path $terrain) | Out-Null
$temporary = "$terrain.tmp"
& (Join-Path $compiler "dxc.exe") -T lib_6_3 -WX -Fo $temporary (Join-Path $Root "crates/render/src/chunk/dxr/native/terrain.hlsl")
if ($LASTEXITCODE -ne 0) { throw "Native terrain library compilation failed ($LASTEXITCODE)" }
if (-not (Test-Path $terrain) -or (Get-FileHash $temporary).Hash -ne (Get-FileHash $terrain).Hash) {
    Move-Item -Force $temporary $terrain
} else { Remove-Item $temporary }
if ($env:GITHUB_ENV) { "CINNABAR_NATIVE_DXR_DXIL=$terrain" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append }
$env:CINNABAR_NATIVE_DXR_DXIL = $terrain
