#!/bin/bash
# Run tensorboard in the background - tensorboard is installed in base env but not in pytorch env .. also it takes ages to install
tensorboard --logdir=runs &
# Activate pytorch environment within the ec2 instance
eval "$(conda shell.bash hook)"
source activate pytorch_p36
# Run model training script
python3 main.py
# Kill tensorboard process that has been sent to the background
# kill $!