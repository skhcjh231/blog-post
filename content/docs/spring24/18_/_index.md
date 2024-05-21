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
 ViT Models generally use learnable positional encoding or sin-cos positional encoding. However, these methods are highly sensitive to changes in input resolution, and they fail to provide effective resolution adaptability. 
 
 The most common method used to address this problem is to apply interpolation to positional encoding before feeding it into the ViT. This approach allows for some compensation of positional information even when the input resolution changes. However, this method has shown significant performance degradation in image classification tasks.
 
 Recently, ResFormer proposed adding depth-wise convolution to the existing positional encoding method when performing global-local positional embedding, enabling it to work well even with unseen resolutions. (Chu et al., 2023; Tian et al., 2023).

<p align="center">
  <img src="./ResFormer_pe.png" alt="." width="400" height="200" > 
</p>

 However, ResFormer has three drawbacks.
 - Shows high performance only in a relatively small range of resolutions (Degradation significantly when resolution is greater than 892)
 - It cannot be used with self-supervised learning methods like masked auto-encoding (MAE).
 - Computation cost increases as input resolution increases, which has a negative impact on the training and inference process.


## ViTAR: Vision Transformer with Any Resolution
In this section, we introduces two key innovations to address this issue. Firstly, we propose a novel module for dynamic resolution adjustment, designed with a single Transformer block, specifically to achieve highly efficient incremental token integration. Secondly, we introduce fuzzy positional encoding in the Vision Transformer to provide consistent positional awareness across multiple resolutions, thereby preventing overfitting to any single training resolution.

<p align="center">
  <img src="./ResFormer_pe.png" alt="." width="500" height="300" > 
</p>


### 1. Adaptive Token Merger (ATM Module)

<p align="center">
  <img src="./ATM.png" alt="." width="500" height="300" > 
</p>

### 2. Fuzzy Positional Encoding (FPE)
 Fuzzy Positional Encoding(FPE) differs from the previously mentioned methods. It enhances the model's resolution robustness without introducing specific spatial structures like convolutions. Therefore, it can be applied to self-supervised learning frameworks. This property enables ViTAR to be applied to large-scale, unlabeled training sets for training, aiming to obtain a more powerful vision foundation model.

<p align="center">
  <img src="./FPE.png" alt="." width="500" height="300" > 
</p>

 Initially, the learnable positional embedding is randomly initialized and used as the model's positional embedding. At this time, FPE provides only fuzzy positional information and experiences changes within a certain range. Specifically, assuming that the exact coordinates of the target token are (i, j), the fuzzy positional information is (i + s1, j + s2). s1 and s2 satisfy -0.5 ≤ s1, s2 ≤ 0.5 and follows uniform distribution.

 During training, randomly generated coordinate offsets are added to the reference coordinates during the training process, and grid samples for learnable location embeddings are performed based on the newly generated coordinates to generate fuzzy location encoding.

 In case of inference, precise positional encoding is used instead of FPE. When there is a change in input resolution, interpolation is performed on learnable positional embedding. This has strong positional resilience because it was somehow seen and used in the FPE used in the training phase.
 

## Experiments


<p align="center">
  <img src="./result1.png" alt="." width="500" height="300" > 
</p>
