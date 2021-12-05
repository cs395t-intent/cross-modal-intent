# Cross Modal Transformer for Intentonomy

This repository contains the Cross Modal Transformer for Intentonomy project for the [CS 395T Deep Learning Seminar](https://www.philkr.net/cs395t/) course with [Philipp Krähenbühl](http://www.philkr.net/).

The project is built off of [Intentonomy: a Dataset and Study towards Human Intent Understanding](https://github.com/KMnP/intentonomy). You can find its README in [intent.md](intent.md).

## Download Data
1. Create a data folder in this repository by running `mkdir -p ./data/2020intent/annotations`.
2. Download the annotations to `./data/2020intent/annotations/` as indicated in [DATA.md](DATA.md), but instead of downloading the original data files (intentonomy_[train/val/test]2020.json), download the modified files (intentonomy_[train/val/test]2020_ht_bert.json) from [this Google Drive directory](https://drive.google.com/drive/folders/1XkhK98sM7-M-Cy81yTEzImZVIHZq9Whw?usp=sharing).
3. Open up a `tmux`, `cd` to this repository, and run `python3 download_data.py`.
4. You can use `ctrl+b` then `d` to temporarily minimize the tmux, then run `tmux attach` to reconnect to it. This allows you to download the dataset in the background, even when you log off.
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
5. Run `conda activate intent` and if you get the error `conda: command not found` run `source /scratch/cluster/${USER}/miniconda3/etc/profile.d/conda.sh`
6. Test that your environment works by running the following in `python`:
```python3
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```

## Train Command
Here's an example command:
```bash
python3 -m train \
    --name baseline \
    --model_type vis_baseline \
    --bs 50 \
    --lr 1e-3 \
    --linear_warmup \
    --warmup_epochs 5 \
    --epochs 50 \
    --use_loc_loss \
    --loc_loss_alpha 1.0
```
The model will be trained and saved at `models/baseline` and logged at `logs/baseline`. You can use `tensorboard --logdir logs` to view it. This command correspond to training with batch size 50, learning rate of 0.001, linearly increase learning rate from 0 to 0.001 for the first 5 epochs, training for 50 epochs, and use the localization loss.

For the Visual VirTex model, run the following:
```bash
python3 -m train \
    --name virtex \
    --model_type vis_virtex \
    --bs 50 \
    --lr 1e-3 \
    --linear_warmup \
    --warmup_epochs 5 \
    --epochs 50
```

For the Visual Swin Transformer (Small) model, run:
```bash
python3 -m train \
    --name swin_small \
    --model_type vis_swin_small \
    --bs 50 \
    --lr 1e-5 \
    --wd 1e-8 \
    --opt_type adamw \
    --epochs 50
```
You can specify the optimizer type with the `--opt_type` parameter. `--wd` corresponds to the weight decay of the optimizer. Swin transformers are fine-tuned with the parameters given here on ImageNet, which we tried to use for Intentonomy as well.

For the Visual + Hashtag models, run:
1. Baseline:
```
python3 -m train \
    --name baseline \
    --model_type baseline \
    --bs 50 \
    --lr 1e-3 \
    --linear_warmup \
    --warmup_epochs 5 \
    --epochs 50 \
    --use_loc_loss \
    --loc_loss_alpha 1.0 \
    --use_hashtags
```
2. VirTex:
```
python3 -m train \
    --name virtex \
    --model_type virtex \
    --bs 50 \
    --lr 1e-3 \
    --linear_warmup \
    --warmup_epochs 5 \
    --epochs 50 \
    --use_hashtags
```
3. Swin Transformer (Tiny):
```
python3 -m train \
    --name swin_tiny \
    --model_type swin_tiny \
    --bs 50 \
    --lr 1e-5 \
    --wd 1e-8 \
    --opt_type adamw \
    --epochs 50 \
    --use_hashtags
```
