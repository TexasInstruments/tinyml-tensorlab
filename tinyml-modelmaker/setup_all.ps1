# PowerShell equivalent of setup_all.sh

# Set directory variables
$CUR_DIR = Get-Location
$PARENT_DIR = (Get-Item $CUR_DIR).Parent.FullName
$HOME_DIR = $env:USERPROFILE
$HOME_DIR = (Get-Item $HOME_DIR).FullName

# Repository source location
$USE_INTERNAL_REPO = 0

if ($USE_INTERNAL_REPO -eq 1) {
    $SOURCE_LOCATION = "ssh://git@bitbucket.itg.ti.com/tinyml-algo/"
}
else {
    $SOURCE_LOCATION = "https://github.com/TexasInstruments/tinyml-tensorlab/"
}

# Print source location
Write-Output "SOURCE_LOCATION=$SOURCE_LOCATION"

# Clone repositories
Write-Output "Cloning/updating git repositories. This may take some time..."
Write-Output "If there is any issue, please remove these folders and try again $PARENT_DIR/tinyml-tinyverse"

# Clone tinyml-tinyverse
if (-not (Test-Path "$PARENT_DIR/tinyml-tinyverse")) {
    git clone --depth 1 --branch main "${SOURCE_LOCATION}tinyml-tinyverse.git" "$PARENT_DIR/tinyml-tinyverse"
}
else {
    Get-ChildItem "$PARENT_DIR/tinyml-tinyverse"
}

# Clone tinyml-modeloptimization
if (-not (Test-Path "$PARENT_DIR/tinyml-modeloptimization")) {
    git clone --depth 1 --branch main "${SOURCE_LOCATION}tinyml-modeloptimization.git" "$PARENT_DIR/tinyml-modeloptimization"
}
else {
    Get-ChildItem "$PARENT_DIR/tinyml-modeloptimization"
}

Set-Location $PARENT_DIR/tinyml-modelmaker
Write-Output "Cloning/updating done."

# Upgrade pip
python -m ensurepip --upgrade
python -m pip install --no-input --upgrade pip setuptools
python -m pip install --no-input --upgrade wheel

# For -m pip install --editable . --use-pep517 mode to work in PowerShell
git config --global --add safe.directory (Get-Location)

Write-Output "Installing repositories..."

Write-Output "Installing: tinyml-modeloptimization"
Set-Location $PARENT_DIR/tinyml-modeloptimization/torchmodelopt
python -m pip install --no-input --editable .

Write-Output "Installing: tinyml-tinyverse"
Set-Location $PARENT_DIR/tinyml-tinyverse
python -m pip install --no-input --editable .

Write-Output "Installing tinyml-modelmaker"
Set-Location $PARENT_DIR/tinyml-modelmaker
python -m pip install --no-input --editable .

Get-ChildItem $PARENT_DIR/tinyml-*

Write-Output "Installation done."
