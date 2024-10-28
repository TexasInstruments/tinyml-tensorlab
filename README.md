# tinyml-tensorlab

<hr>

## 1. Set up TI tinyml-tensorlab

### Prequisite:
#### Step 1.1: Make sure that you are using bash shell. If it is not bash shell, change it to bash. Verify it by typing:
```
echo ${SHELL}
```

#### Step 1.2: Install system dependencies
```
sudo apt update
sudo apt install build-essential curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev xz-utils wget curl
sudo apt install -y libffi-dev libjpeg-dev zlib1g-dev graphviz graphviz-dev protobuf-compiler
```
#### Step 1.3: Install pyenv using the following commands
```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc

exec ${SHELL}
```

Further details on pyenv installation are given here https://github.com/pyenv/pyenv and https://github.com/pyenv/pyenv-installer


#### Step 1.4: Install Python 3.10 in pyenv and create an environment
```
pyenv install 3.10
pyenv virtualenv 3.10 py310
pyenv rehash
pyenv activate py310
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools
```

Note: This activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate py310
```

#### Step 1.5: Set up the repositories

0. **NOTE: Please download and install [C2000Ware](https://www.ti.com/tool/C2000WARE)**
   * Please set the installed path in your terminal: `export C2000WARE_PATH="/path/to/C2000Ware_5_03_00_00"`
1. Clone this repository
2. `cd tinyml-tensorlab/tinyml-modelmaker`
3. Execute (Requires sudo permissions): ``` ./setup_all.sh ```

#### Step 1.6: Keeping up to date

Since these repositories are undergoing a massive feature addition stage, it is recommended to keep your codes up to date by running the following command:
```commandline
git_pull_all.sh
```


<hr>

## tinyml-docs
[tinyml-docs](tinyml-docs)
- Overview/landing page

<hr>

## tinyml-modelzoo
[tinyml-modelzoo](tinyml-modelzoo)
- Browse our collection of models, understand their performance trade-offs.

<hr>

## tinyml-modelmaker
[tinyml-modelmaker](tinyml-modelmaker)
- Bring your own data and train/compile models

<hr>
