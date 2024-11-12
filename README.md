# tinyml-tensorlab

<hr>
The Tiny ML Tensorlab repository is meant to be as a starting point to install and explore TI's AI offering for MCUs.
It helps to install all the required repositories to get started.
Currently, it can handle Time series Classification tasks. A lot more READMEs are present under the Tiny ML Modelmaker Repo

There are a few productized applications:

| Select sector (Industrial, automotive, personal electronics) | Technology  | Application (Title)           | Application Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Features / advantages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Call to action                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------ | ----------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Industrial                                                   | Time series | Arc fault detection           | An arc fault is an electrical discharge that occurs when an electrical current flows through an unintended path, often due to damaged, frayed, or improperly installed wiring. This can produce heat and sparks, which can ignite surrounding materials, leading to fires.<br><br>Due to the ability of AI to analyze complex patterns, continuously learn and improve from new data, address a wide range of faults, it is advantageous to use AI. Using AI at the edge empowers the customer with reduced latency, enhanced privacy and scalability while saving bandwidth.<br><br>TI provides UL-1699B tested AI models which have impeccable accuracy and ultra low latency.                                                    | By utilising benefits of AI such as its ability to analyze patterns in signals and ability to handle larger volumes of data, TI's solution allows for immediate detection response to arc faults.<br><br>Coupled with an NPU that provides enhanced AI performance, TI's  brings additional benefits in terms of speed, reliability, and scalability, making it a powerful approach for enhancing electrical safety.<br><br>With TI's complete solution, AFD will never be a showstopper for you.                              | To empower your solution with TI’s AI, you can use the Model Composer GUI to quickly train an AI model or use the Tiny ML Modelmaker for an advanced set of capabilities. To customers who rely on their own AI training framework, TI’s Neural Network Compiler can help you get your AI model compatible with MCUs (P55x,P66x or any other F28 device). For a full-fledged reference solution, find the comprehensive project here. |
| Industrial                                                   | Time series | Motor Bearing Fault Detection | Motor bearing faults are often seen in HVAC systems with rotating parts. Itoccurs due to the wear and tear of moving parts, lack of lubrication, and due to overloading of equipment. It adversely affects the motor lifespan and increases energy consumption, potentially even can cause a failure of the system.<br><br>By using AI, these faults can be detected early by monitoring signs such as subtle changes in vibration patterns. Processing such data locally at the HVAC system can provide real time fault detection and immediate response, which is crucial for preventing damage and ensuring continuous operation.<br><br>TI provides handcrafted AI models which have impeccable accuracy and ultra low latency. | TI's AI solution addresses these by montoring the vibration and temperature of the motor through sensors and provides a reliable solution by combining the strengths of advanced analytics and real-time processing, leading to more reliable, efficient, and cost-effective maintenance and operation.<br><br>Put together with an NPU for advanced AI performance capabilities, this prevents unexpected failures as the algorithms can detect early signs of faults that might not be noticeable through manual inspections | To empower your solution with TI’s AI, you can use the Model Composer GUI to quickly train an AI model or use the Tiny ML Modelmaker for an advanced set of capabilities. To customers who rely on their own AI training framework, TI’s Neural Network Compiler can help you get your AI model compatible with MCUs (P55x,P66x or any other F28 device). For a full-fledged reference solution, find the comprehensive project here. |


* To empower your solution with TI’s AI, you can use the **Tiny ML Modelmaker** for an advanced set of capabilities.
  * Supports any Time series Classification tasks (including Arc Fault and Motor Bearing Fault Classification)
* You can also use the [Edge AI Studio Model Composer GUI](https://dev.ti.com/modelcomposer/) to quickly train an AI model (No Code Platform)
  * This supports only Arc Fault and Motor Bearing Fault Classification applications currently.
* To customers who rely on their own AI training framework, TI’s Neural Network Compiler can help you get your AI model compatible with MCUs (P55x,P66x or any other F28 device).
* For a full-fledged reference solution on Arc Fault and Motor Bearing Fault, find the comprehensive project in [Digital Power SDK](https://www.ti.com/tool/C2000WARE-DIGITALPOWER-SDK) and [Motor Control SDK](https://www.ti.com/tool/C2000WARE-MOTORCONTROL-SDK).


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
