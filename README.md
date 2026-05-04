# 📄 Paper Detection — Image Classification with Faster RCNN

A computer vision project that trains a model to detect whether **paper** (white paper, newspaper, cardboard) is present in an image — and if so, locates it with a bounding box.

---

## 🎯 Objective

Train an object detection model using **Faster RCNN** (ResNet50 or MobileNet V3 backbone) to classify and localize paper objects in images. The model outputs bounding boxes around detected paper regions.

---

## 📁 File Structure

| File | Description |
|---|---|
| `main.py` | Entry point — runs the full training pipeline |
| `model.py` | Builds Faster RCNN with a chosen backbone and custom predictor head |
| `trainer.py` | Training loop with loss tracking per epoch |
| `dataset.py` | Custom PyTorch Dataset — loads images and bounding box annotations |
| `data_preparation.py` | Parses and preprocesses CSV annotation files |
| `augmentations.py` | Data augmentation transforms for training |
| `evaluation.py` | Evaluates model performance |
| `args.py` | CLI argument parser for hyperparameters |
| `utils.py` | Shared utility functions |
| `gpu_test.py` | Checks CUDA availability |
| `data/CSVs/` | Bounding box annotations in CSV format |

---

## ⚙️ How to Run

**1. Install dependencies**
```bash
pip install torch
```

**2. Train the model**
```bash
python main.py 
```

**3. Run paper-detecting-images model**
```bash
python evaluation.py
```

---

## 🖼️ Detection Results

> Sample images showing model predictions (bounding boxes around detected paper):

| Input Image | Detected |
|---|---|
| <img src="https://raw.githubusercontent.com/luu2310/detecting-paper-project/main/test_results/result_IMG_0098.jpg" width="400"/> | ✅ Paper detected |
| <img src="https://raw.githubusercontent.com/luu2310/detecting-paper-project/main/test_results/result_IMG_100.jpg" width="400"/> | ❌ No paper found |
