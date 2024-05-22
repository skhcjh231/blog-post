---
type: docs
bookToc: True
weight: 1
---

# Scaling (Down) CLIP: A Comprehensive
*Posted by: Harit Keawmuang, Junkyeong Park*

*Authors: Zichao Li (University of California, Santa Cruz), Cihang Xie (University of California, Santa Cruz), Ekin Dogus Cubuk (Google Deepmind)*

introduction

## Data
### Data Quantity


# What is CLIP?
[CLIP](https://arxiv.org/abs/2103.00020) effectively merges the capabilities of natural language processing (NLP) and computer vision. By learning from images and their textual descriptions, CLIP unifies text and image understanding, allowing it to perform various tasks without task-specific training.

## Related Works
In the recent years, NLP has advanced significantly in pre-training language models on large text datasets. Simultaneously, computer vision has improved by pre-training convolutional neural networks (CNNs) on extensive image datasets. The CLIP model combines these approaches by jointly pre-training on images and text using a contrastive loss function, creating a shared embedding space for both. 

Recent efforts have focused on improving scalability and efficiency of CLIP model. For example, [FLIP]([ref](https://arxiv.org/abs/2212.00794)) was introduced to minimize the computation by masking  image patches, enabling larger batch sizes without sacrificing performance. Most research has focused on large-scale training with significant computational resources, utilizing [ViT](https://arxiv.org/abs/2010.11929) large models on extensive datasets. However, less attention has been given to optimizing CLIP for smaller training budgets.

## Method

### Training Pipeline, Dataset, and Hyperparameters
In this paper, they adopted the identical training approach as [CLIP](https://arxiv.org/abs/2103.00020), which employs a contrastive loss to simultaneously train the vision and text encoders from scratch. This loss function encourages the encoders to map related image-text pairs to similar feature representations in a shared embedding space by minimizing the distance between positive pairs and maximizing the distance between negative pairs. Key aspects of the training include:

- Minimal data augmentation: Images resized to 224x224 and pixel values normalized to the range of -1 to 1.
- Optimizer: AdafactorShazeer & Stern with β1 = 0.9 and β2 = 0.999.
- Batch size: 16k
- Learning rate: Initial rate of 0.001 with a cosine learning scheduler and weight decay of 0.0001.

### Evaluation Matrices
- [**Zero-shot transfer evaluation**](https://en.wikipedia.org/wiki/Zero-shot_learning): Assesses the model's ability to generalize to new tasks without fine-tuning.
- [**Linear probe evaluations**](https://en.wikipedia.org/wiki/Linear_probing): Freezes the vision encoder and optimizes the fully connected layer's learning rate.
- [**Retrieval performance on MSCOCO captions**](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)): Ranks text captions based on cosine similarity with image embeddings, reporting Recall@1 for image-to-text retrieval and average results for text-to-image retrieval.

## Comparison of Network Architectures
To effectively choose the best network architectures, they performed a comparison among the various architectures. Previous studies have explored various vision encoders for CLIP, such as ResNet, MLP-Mixer, and ViT, but some architectures like Swin-Transformer and ConvNext haven't been investigated. Here, they compared CNN and vision transformer architectures with similar computational costs, including ViT-B/32, ResNet-50, ConvNext-T, Swin-T, and Mixer-B/32. In Zero-shot, when considering limited data samples, ResNet-50 performs better initially, but ViT-B/32 achieves superior performance with more samples due to its stronger ability to capture global information (see Figure 1(a)). In linear probing, MLP-Mixer outperforms others with fewer samples, but ViT-B/32 excels with larger datasets. ViT and MLP-Mixer show better robustness, likely due to their lower inductive bias, leading to improved generalization (Figure 1(b)). For retrieval tasks, ResNet-50 is better with smaller sample sizes, but ViT-B/32 surpasses it as sample sizes increase. Mixer-B/32 performs poorly in retrieval tasks, making ViT the preferred choice for CLIP's vision encoder across various tasks. 


Figure
