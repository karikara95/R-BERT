#/!bin/bash
#create env
echo "Installing r-bert env..."
conda create --name r-bert python==3.7 ipython
conda activate r-bert
#install packages
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers[sentencepiece]==3.3.1