# SEED
Official implementation of **SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding (ICLR 2026)**.

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2503.06437-brown?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2503.06437)
&nbsp;
[![Project Page](https://img.shields.io/badge/Project%20Page-0A7B83?style=flat-square)](https://concarne2.github.io/seed_project_page/)

</div>

## 1. What Is SEED?
SEED is a semantic evaluation metric for visual brain decoding that compares reconstructed images against ground-truth images using complementary semantic signals.
It combines object-level agreement, image-level feature similarity, and caption-level semantic similarity into one score.

The final SEED score is:

```text
SEED = (Object F1 + Cap-Sim + EffNet) / 3
```

Where:
- `Object F1` measures object category overlap between detection results.
- `Cap-Sim` is the cosine similarity between generated caption embeddings.
- `EffNet` is image feature similarity.

Implementation note for this repo: `seed/metrics.py` computes EffNet as correlation distance and converts it back to similarity during final aggregation (`1 - effnet_distance`), which matches the paper-level formulation above.

## 2. Quick Start

### 2.1 Environment Setup
```bash
# Create and activate environment
conda create --name seed python=3.8 -y
conda activate seed

# Install PyTorch (choose the command matching your CUDA setup)
conda install pytorch torchvision -c pytorch

# Install MMDetection dependencies
pip install openmim
mim install mmengine
mim install mmcv

# Install SEED package in editable mode
pip install -e .

```

If you prefer, you can run the provided setup script instead (you may need to change the pytorch installation line):

```bash
bash installation.sh
```

### 2.2 Download Pre-trained Grounding Model
Download the Grounding DINO checkpoint (required for detection).
This links to the MM-Grounding-DINO-L model. For more information, see https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md

```
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth
```

### 2.3 Run Full Evaluation 
```bash
bash seed_evaluation.sh
```

Before running, update image directory variables in `seed_evaluation.sh` to match your data paths:
- `RECON_IMAGE_PATH`
- `GT_IMAGE_PATH`

This runs object detection for reconstruction and GT images, then computes Object F1, Cap-Sim, EffNet, and the final SEED score.

## 3. Expected Input and Output Structure

### 3.1 Input Structure
Prepare paired reconstruction and GT image folders with matching filenames:

```text
human_eval_data/
  images_test/
    gt/
      a.png
      b.png
      ...
    recon/
      a.png
      b.png
      ...
```

Requirements:
- `gt/` and `recon/` must contain corresponding image files with the same names for corresponding GT and reconstruction pairs.
- Supported image files should be readable by `PIL` (e.g., `.png`, `.jpg`).

### 3.2 Output Structure
Running detection + evaluation creates outputs under:

```text
evaluations/<model_name>/
  recon_detection_results/
    preds/
      *.json
    vis/
      *.png
  gt_detection_results/
    preds/
      *.json
    vis/
      *.png
  intermediate_results/
    obj_f1.npy
    effnet.npy
    cap_sim.npy
    recon_captions.npy
    gt_captions.npy
```

Notes:
- `preds/*.json` stores per-image detection outputs used by Object F1.
- `vis/*.png` stores visualized detections.
- `intermediate_results/*.npy` stores per-image metric values and generated captions.

## 4. Metrics Explained

### 4.1 Object F1
Object F1 measures object-category agreement between reconstruction and GT images.

- Objects are detected for both image sets using `image_detection.py` (Grounding DINO config + weights).
- Per-image categories are collected from `preds/*.json` with score threshold sweeps `seed/metrics.py`.
- Precision/recall are computed from category overlap and converted to per-image F1.

Higher is better.

### 4.2 Cap-Sim
Cap-Sim measures semantic similarity between generated captions for reconstruction and GT images.

- Captions are generated with GIT.
- Captions are embedded with Sentence Transformer.
- Cosine similarity is computed between paired caption embeddings.

Higher is better.

### 4.3 EffNet
EffNet measures image-level feature similarity using EfficientNet-B1 features.

In this repo:
- Features are extracted from EfficientNet-B1 (`avgpool`).
- The metric usually used in relevant literature is the correlation distance per image (`scipy.spatial.distance.correlation`).
- For our purposes, the correlation distance is converted to cosine similarity as `1 - effnet_distance`.

Higher is better after conversion to similarity.

### 4.4 Final SEED Score
The final score is the average of the three components:

```text
SEED = (Object F1 + Cap-Sim + EffNet) / 3
```

## 5. (Optional) Download Human Survey Results
We provide our collected human survey results for researchers interested in developing new evaluation metrics and plan to use the survey results to meta-evaluate different evaluation metrics.

Download and unzip the data:
```
wget https://github.com/Concarne2/SEED/releases/download/v1.0.0/human_eval_data.tar.gz
tar -xzf human_eval_data.tar.gz
```

This release contains the human survey results and related data, including the image files used for the survey, our evaluation results for those images, and the suggested usage of the survey results.

### Included files

- `250131_final.csv`  raw survey responses
- `images/`  paired image sets
  - `images/gt/` (1000 PNGs)
  - `images/recon/` (1000 PNGs, filename-matched to `gt`)
- `eval_metrics.npz`  precomputed metric arrays for 1000 items
  - Keys: `pixcorr`, `ssim`, `alexnet2`, `alexnet5`, `inception`, `clip`, `effnet`, `swav`, `obj_f1`, `git_st`
- `survey_analysis.ipynb`: suggested survey analysis notebook
- `dataset.py`, `tau_optimization.py`: metric/correlation utilities used by the notebook. These utilities are adopted from the t2v_metrics repro: https://github.com/linzhiqiu/t2v_metrics

## 6. License
This project is released under the Apache 2.0 License.
See `LICENSE` for details.
