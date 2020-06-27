#!/usr/bin/env bash

rm -rf ./env
rm -rf .ipynb_checkpoints

conda create -p .env python=3.6.2
conda activate ./.env
conda install ipython
conda install jupyter

pip install -Ur requirements.txt
pre-commit install
