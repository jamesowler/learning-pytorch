#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate
conda env list
echo Executing script within pytorch environment ...
python3 main.py