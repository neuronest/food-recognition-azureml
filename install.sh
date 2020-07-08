#!/usr/bin/env bash

rm -rf .ipynb_checkpoints

conda remove --name food_prediction --all
conda create --name food_prediction python=3.6.2
conda activate food_prediction
conda install ipython
conda install jupyter

pip install -Ur requirements.txt
pre-commit install
