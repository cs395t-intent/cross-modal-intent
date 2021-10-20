# Cross Modal Transformer for Intentonomy

This repository contains the Cross Modal Transformer for Intentonomy project for the [CS 395T Deep Learning Seminar](https://www.philkr.net/cs395t/) course with [Philipp Krähenbühl](http://www.philkr.net/).

The project is built off of [Intentonomy: a Dataset and Study towards Human Intent Understanding](https://github.com/KMnP/intentonomy). You can find its README in [intent.md](intent.md).

## Environment Setup
1. Login to `eldar-11.cs.utexas.edu`.
2. Download the latest miniconda into your scratch space on Condor:
```
cd /scratch/cluster/${USER}/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
3. Follow through the installation process:
    * `yes` to the license agreement.
    * Install miniconda3 to `/scratch/cluster/${USER}/miniconda3`.
    * `yes` to conda init.
    * `conda config --set auto_activate_base false` to not run `conda` all the time.
    * Do anything else it tells you to do for `.bashrc` if needed.
    * Add `conda`'s path to your `PATH` environment variable. One way of doing this is to add `PATH=${PATH}:/scratch/cluster/${USER}/miniconda3/bin/` to your `~/.profile`.
3. Clone this repository to anywhere in your Condor scratch space.
4. `cd` the `env/` folder of this repository and setup the environment by running `./install.sh`.
5. Run `conda activate intent`, and test that your environment works by running the following in `python`:
```python3
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```

## Download Data
1. Create a data folder in this repository by running `mkdir -p ./data/2020intent/annotations`.
2. Download the annotations to `./data/2020intent/annotations/` as indicated in [DATA.md](DATA.md), which are the [intentonomy_train2020.json](https://cornell.box.com/s/rff4fuq20t7tc4edx2wl0golh64zf9qh), [intentonomy_val2020.json](https://cornell.box.com/s/rff4fuq20t7tc4edx2wl0golh64zf9qh), [intentonomy_test2020.json](https://cornell.box.com/s/3ep2w96qf91w9qvqop2ri95g0e3fx4zj) files.
3. Open up a `tmux`, `cd` to this repository, and run `python3 download_data.py`.
4. You can use `ctrl+b` then `d` to temporarily minimize the tmux, then run `tmux attach` to reconnect to it. This allows you to download the dataset in the background, even when you log off.
