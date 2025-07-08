# TBPI: Tendency-Based Parameter Initialization for Few-Spike Neurons

This repository contains the official implementation of **TBPI (Tendency-Based Parameter Initialization)**, a novel training strategy designed to enhance the accuracy of non-linear activation function approximation in **Few-Spike (FS) neurons**. The method is particularly beneficial when integrating FS neurons into deep architectures such as **diffusion models**.

> 📌 **Accepted at**: IJCAI 2024 Workshop on Human Brain and Artificial Intelligence (HBAI)  
> 📄 **arXiv Preprint**: [arxiv.org/abs/2409.00044](https://arxiv.org/abs/2409.00044)

---

## 🧠 Overview

Spiking Neural Networks (SNNs) offer high energy efficiency but struggle to approximate complex non-linear activation functions (e.g., Swish, GELU). Few-Spike (FS) neurons were introduced to address this, but their performance has been limited due to ineffective training strategies.

**TBPI** solves this by introducing a tendency-aware initialization approach that considers the **temporal dependency** of neuron parameters across time steps (`h(t)`, `d(t)`, `T(t)`).

---

## 🛠️ Method Pipeline

TBPI consists of three main stages:

1. **Pre-training**  
   Train FS neurons from random initialization to obtain parameter tendencies across time.

2. **Function Fitting**  
   Fit the observed parameter trajectories (`log`, `poly`) to capture temporal dynamics.

3. **Re-initialization and Training**  
   Extract fitted initial values and re-train the neurons with improved convergence and generalization.

---

## 📊 Results Summary

### FS neuron (Swish approximation)
| Method        | K=4     | K=8     | K=12    | K=16    |
|---------------|---------|---------|---------|---------|
| FS-Swish      | 3.8797  | 0.9776  | 0.1521  | 0.1425  |
| + Gaussian    | 3.8796  | 0.8968  | 0.1517  | 0.1249  |
| **TBPI (Ours)**| 3.8795  | **0.6295**  | **0.1413**  | **0.1246**  |

### DDIM Network-Level Results (Oxford Flowers 102)
- TBPI-integrated FS neurons achieved lower image loss and KID scores across all time steps compared to baselines.
- Full results and training configuration available in [arXiv paper](https://arxiv.org/abs/2409.00044).

---

## 📁 Repository Structure

```bash
TBPI/
├── FS_conversion_swish.ipynb           # Swish function approximation using FS neurons
├── FS_conversion_new_Swish.ipynb       # Extended FS-Swish training and logging
├── fsneuron_graph.ipynb                # Visualizing parameter trends (h(t), d(t), T(t)) and function fitting
├── spiking_ddim_fs_swish.ipynb         # FS neuron integration into DDIM and network-level training
├── README.md
````

### File Descriptions

* **`FS_conversion_swish.ipynb`**
  Initial experiment comparing FS-Swish vs. ANN-Swish approximations.

* **`FS_conversion_new_Swish.ipynb`**
  TBPI-integrated Swish approximation using fitted parameter initialization.

* **`fsneuron_graph.ipynb`**
  Visualization and fitting of temporal parameter curves. Core part of TBPI pipeline.

* **`spiking_ddim_fs_swish.ipynb`**
  Application of trained FS-Swish neurons into **DDIM**, validating performance at the network level.

---

## 🔍 Key Contributions

* Proposed **TBPI**, the first principled parameter initialization method for FS neurons.
* Demonstrated accurate approximation of Swish using FS neurons trained with TBPI.
* Improved network-level performance in **denoising diffusion models (DDIM)**.
* Verified applicability across multiple time steps (K=4,8,12,16) and activation types.

---

## 📦 Dependencies

* Python ≥ 3.8
* PyTorch ≥ 1.12
* NumPy, SciPy, Matplotlib
* For DDIM evaluation: `diffusers`, `transformers`, `PIL`

Install via:

```bash
pip install -r requirements.txt
```

---

## 📌 Citation

If you use this work, please cite our paper:

```bibtex
@article{jeong2024tbpi,
  title   = {A More Accurate Approximation of Activation Function with Few-Spike Neurons},
  author  = {Dayena Jeong and Jaewoo Park and Jeonghee Jo and Jongkil Park and Jaewook Kim and Hyun Jae Jang and Suyoun Lee and Seongsik Park},
  journal = {arXiv preprint arXiv:2409.00044},
  year    = {2024}
}
```

---

## ✉️ Contact

For questions or collaborations:
📧 [enju0817@gmail.com](mailto:enju0817@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/dayena-jeong)

---


