# ESL-3DClouds

## Installation with pixi

1. Install pixi from terminal: 

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
Close terminal then check if pixi command is recognized.

2. create pixi environment.

  a. copy pixi_env folder from pixi_env branch of this repo to your home directory
  ```bash
    sudo cp -r pixi_env /home/myusername/
  ```
  b. find PIXI_PATH where pixi is whith  
  ```bash
    which pixi
  ```
  c. install pixi from /home/myusername/pixi_env/extreme/
  ```bash
  sudo PIXI_PATH install
  ```
3. activate pixi environment.
go to /home/myusername/pixi_env/extreme/ (ie where .toml file is) and run
```bash
  pixi shell
  ```

4. set evironment in VS code: 
  a. install python + jupyter extensions
  b. ctr+shift+P: Python select interpreter > find python path: /home/myusername/pixi_env/extreme/.pixi/envs/default/bin/python


## Installation conda

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the fdl-sar-env environment:

```bash
conda env create -f environment.yml

conda activate esl3d-env
```


### Optional, but highly recommended

Install [pre-commit](https://pre-commit.com/) by running the following command to automatically run code formatting and linting before each commit:

```bash
pre-commit install
```

If using pre-commit, each time you commit, your code will be formatted, linted, checked for imports, merge conflicts, and more. If any of these checks fail, the commit will be aborted.

## Adding a new package

To add a new package to the environment, open the environment.yml file and add it under dependencies. By default packages are installed using conda. If pip needs to be used, add it under `- pip`.
