#!/bin/sh

# run once to setup python venv so you can install python packages

cd ~
python3 -m venv --system-site-packages jigsaw
source jigsaw/bin/activate
pip install --upgrade pip
pip install --ignore-installed --upgrade torch torchvision torchaudio
pip install --ignore-installed --upgrade notebook
pip install --ignore-installed --upgrade matplotlib
pip install --ignore-installed --upgrade flask
pip install --ignore-installed --upgrade opencv-python
pip install --ignore-installed --upgrade pandas
deactivate

