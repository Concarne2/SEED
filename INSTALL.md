# Installation Guide

## Quick Install from GitHub

```bash
# Create conda environment
conda create --name seed python=3.8 -y
conda activate seed

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision -c pytorch

# Install mmengine and mmcv (using prebuilt wheels - much faster!)
pip install openmim
mim install mmengine
mim install mmcv

# Install this package directly
pip install git+https://github.com/Concarne2/SEED.git
```

That's it! The modified mmdetection code is included in the package.

## Development Install

For local development:

```bash
# Clone the repository
git clone https://github.com/Concarne2/SEED.git
cd seed

# Create conda environment
conda create --name seed python=3.8 -y
conda activate seed

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install mmcv first (using prebuilt wheels)
pip install openmim
mim install mmcv

# Install in editable mode
pip install -e .
```

## Using the Modified MMDetection

The modified mmdetection code is included as `mmdet`. Import normally:

```python
from mmdet.apis import init_detector, inference_detector
```

## Verification

After installation, verify by running:

```bash
python -c "import torch; import mmdet; import transformers; print('Installation successful!')"
```

## Notes

- Python 3.8 is required for compatibility
- PyTorch should be installed via conda for better CUDA support
- The package includes a modified version of mmdetection with custom fixes and features
