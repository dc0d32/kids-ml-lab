# Kids ML Lab launcher for Windows PowerShell.
#   .\run.ps1 app     -> the interactive playground (start here)
#   .\run.ps1 lab     -> JupyterLab with the chapter notebooks
#   .\run.ps1 test    -> the tests
#   .\run.ps1 build   -> regenerate notebooks from notebooks/_src/
param(
    [Parameter(Position = 0)] [string] $Command = "app",
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)] [string[]] $Rest
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Python reads and writes UTF-8 here; the chapters are full of emoji and the Windows
# console default (cp1252) cannot handle them.
$env:PYTHONUTF8 = "1"

switch ($Command) {
    "app"   { uv run streamlit run app/Home.py }
    "lab"   { uv run jupyter lab notebooks }
    "test"  { uv run pytest tests -q }
    "build" { uv run python tools/build_notebooks.py @Rest }
    default { Write-Host "usage: .\run.ps1 [app|lab|test|build]"; exit 1 }
}
