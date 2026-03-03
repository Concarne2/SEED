conda create --name seed python=3.8 -y
conda activate seed

# Installing Pytorch
conda install pytorch torchvision -c pytorch

# Note: you may want to install
# refer to this doc for previous versions: https://pytorch.org/get-started/previous-versions/

# Installing mmdetection
# information for installing mmdetection can also be found here: https://mmdetection.readthedocs.io/en/latest/get_started.html
pip install openmim
mim install mmengine
mim install mmcv

# Installing SEED packages as a editable local install
pip install -e .

# Optional: Install from GitHub instead
# pip install git+https://github.com/Concarne2/SEED.git