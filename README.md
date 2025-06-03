# Talk2Dino Segmentation Evaluation

This repository is set up for evaluating segmentation models (`OpenSeg` and `DINO`).

## 🚀 Setup Instructions

### ✅ System Requirements

- **CUDA**: 11.8  
  Confirm with:
  ```bash
  nvcc --version
  ```
  Output should include:
  ```
  Cuda compilation tools, release 11.8, V11.8.89
  ```

- **GCC**: 11.4.0  
  Confirm with:
  ```bash
  gcc --version
  ```
  Output should include:
  ```
  gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
  ```

### 🐍 Python Environment (via Conda)

Create and activate a conda environment with Python 3.9:

```bash
conda create --name talk2dino python=3.9 -y
conda activate talk2dino
```

### 🔧 Install Dependencies

#### PyTorch (CUDA 11.7 build)

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

#### Project Requirements

```bash
pip install -r requirements.txt
```

#### MMEngine and Segmentation Packages

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.6.2"
mim install "mmsegmentation==0.27.0"
```

---

## 🧪 Run Evaluation

Run the segmentation evaluation with either model:

### OpenSeg

```bash
python eval_segmentation.py --model openseg --output_dir results_openseg_test
```

### DINO

```bash
python eval_segmentation.py --model dino --output_dir results_dino_test
```
