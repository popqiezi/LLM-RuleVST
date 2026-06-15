

# 🚢 LLM-RuleVST (YOLO Module Only)

## 📌 Overview

This repository contains a **partial implementation** of the paper:

**LLM-RuleVST: A Rule-Guided Visual-Sequence Transformer Framework for UAV-Based Vessel Trajectory and Risk Prediction in Inland Waterways**

⚠️ **Current release only includes the YOLO-based UAV visual perception module.**
The LLM reasoning module and Transformer-based trajectory & risk prediction modules are not included yet.

---

## 🎯 Released Features

### ✅ UAV Visual Perception (YOLO-based)

This module focuses on vessel detection in UAV aerial imagery, including:

* UAV-based vessel detection in inland waterways
* Small object detection enhancement
* Sub-pixel center refinement for improved localization accuracy
* Output generation for downstream trajectory modeling (not included in this repo)

---

## 🧠 Method Summary

This implementation is built on a YOLO-based detector with UAV-specific improvements:

### 1. Sub-pixel Center Refinement

Improves localization accuracy for small vessels using:

* 1×1 convolution-based feature expansion
* Pixel Shuffle for high-resolution feature reconstruction
* Offset prediction branch for center correction

### 2. UAV Scene Optimization

Designed for inland waterway environments with:

* Dense vessel traffic
* Small-scale objects
* Complex water backgrounds

---

## 🚀 Usage

### Training

```bash
python train.py --config config.yaml
```

### Inference

```bash
python infer.py --weights best.pt --source test_images/
```

---

## 📊 Dataset

The model is designed for UAV-based inland waterway scenarios, including data collected from the Yangtze River (Jiangsu section).

Typical scenarios include:

* Single vessel navigation
* Encounter situations (head-on / crossing / overtaking)
* Dense multi-vessel traffic

⚠️ The dataset is not publicly available.

---

## 📌 Roadmap

* UAV-TSD full trajectory extraction module
* LLM-based navigation rule reasoning module
* Transformer-based trajectory prediction module
* Collision risk prediction (CRI) module
* Full end-to-end LLM-RuleVST system

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@article{LLM-RuleVST,
  title={LLM-RuleVST: A Rule-Guided Visual-Sequence Transformer Framework for UAV-Based Vessel Trajectory and Risk Prediction in Inland Waterways},
  author={Ang Li},
  year={2026}
}
```

---

## ⚠️ Notes

* This repository only provides the **visual perception (YOLO) module**
* Full system (LLM + Transformer) will be released in future updates
* The codebase is still being organized and improved

---

## 📬 Contact

Email: [popqiezi@gmail.com](mailto:popqiezi@gmail.com)

---

