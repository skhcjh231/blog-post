---
type: docs
bookToc: True
weight: 1
---

# **ViTAR: Vision Transformer with Any Resolution**
*Authors: Qihang Fan, Quanzeng You, Xiaotian Han, Yongfei Liu, Yunzhe Tao, Huaibo Huang, Ran He, Hongxia Yang*

## Vision Transformers (ViTs)

<p align="center">
  <img src="./ViT.png" alt="." width="500" height="300" > 
</p>

## Challenge: Multi-Resolution ViT Modeling



## Method: ViTAR


### 1. Adaptive Token Merger (ATM Module)

<p align="center">
  <img src="./ATM.png" alt="." width="500" height="300" > 
</p>

### 2. Fuzzy Positional Encoding (FPE)
 Vision Transformer Models generally use learnable positional encoding or sin-cos positional encoding. However, these methods are highly sensitive to changes in input resolution, and they fail to provide effective resolution adaptability. To improve this, ResFormer proposed adding depth-wise convolution to the existing positional encoding method when performing global-local positional embedding, enabling it to work well even with unseen resolutions. (Chu et al., 2023; Tian et al., 2023).

 However, convolution-based positional embedding requires adjacent patches to fully extract and utilize spatial features. Therefore, it is not suitable for self-supervised learning methods like masked auto-encoding (MAE), which require masking parts of image patches. This limitation poses challenges for large-scale training.

 Fuzzy Positional Encoding(FPE) differs from the previously mentioned methods. It enhances the model's resolution robustness without introducing specific spatial structures like convolutions. Therefore, it can be applied to self-supervised learning frameworks. This property enables ViTAR to be applied to large-scale, unlabeled training sets for training, aiming to obtain a more powerful vision foundation model.

<p align="center">
  <img src="./FPE.png" alt="." width="500" height="300" > 
</p>


## Experiments


<p align="center">
  <img src="./result1.png" alt="." width="500" height="300" > 
</p>
