---
type: docs
bookToc: True
weight: 1
---

# Scaling (Down) CLIP: A Comprehensive Analysis of Data, Architecture, and Training Strategies
*Posted by: Harit Keawmuang, Junkyeong Park*

*Authors: Zichao Li (University of California, Santa Cruz), Cihang Xie (University of California, Santa Cruz), Ekin Dogus Cubuk (Google Deepmind)*

In recent years, there has been a growing interest in image-and-language representation learning, which aims to capture the complex interactions between visual and textual information. The Contrastive Language-Image Pre-Training (CLIP) framework has emerged as a leading approach in this field, utilizing large-scale text and image data to create a unified representation space. CLIP has achieved remarkable performance across various tasks and has demonstrated robust generalization to out-of-distribution data. While prior studies on scaling CLIP have focused on scenarios with substantial computational resources, this paper investigates the performance of CLIP under resource constraints, specifically examining the effects of data size, architecture, and training strategies.

The study explores the impact of different training data sizes, showing that smaller, high-quality datasets can outperform larger, lower-quality ones. This is critical for practical applications where data quality and computational limits are significant considerations. The research also compares various architectures, highlighting that larger vision transformers (ViTs) do not always guarantee better performance and that CNNs may be more effective when data is limited. Additionally, the paper evaluates different training strategies, including SLIP, FLIP, CLIP, and CLIP+Data Augmentation, revealing that data augmentation can enhance performance without significant computational costs. These findings provide valuable insights for efficiently training and deploying CLIP models, making advanced image-and-language learning more accessible and affordable.

## What is CLIP?
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

## Data
### Data Quantity
To evaluate the effect of data quantity on CLIP's performance, they conducted experiments with datasets of different sizes: 10M, 25M, 100M, 200M, and 400M. Using ViT-B/32 as the vision encoder, models were trained for 2 to 32 epochs.

<p align="center">
    <img src="./Figure 2.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 1.</strong> Zero-shot performance with the various training strategies
</div>
</p>

Results showed that for smaller datasets (e.g., 25M), increasing epochs did not significantly improve ImageNet performance. In contrast, larger datasets (e.g., 400M) benefited from more epochs. Additionally, zero-shot performance on ImageNet variants followed a similar pattern: larger datasets and longer training improved performance. However, the correlation between performance on ImageNet and its variants was inconsistent, with some datasets showing improved results in specific variants but not others.

<p align="center">
    <img src="./Figure 3.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 2.</strong> Data Quantity: Few-Shot Performances on ImageNet
</div>
</p>

They also observed that the few-shot performance also showed a similar trend to the zero-shot performance.

<p align="center">
    <img src="./Figure 4.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 3.</strong> Retrieval Performances on MSCOCO
</div>
</p>

In Retrieval Performances, a slightly different trend emerged. Specifically, they found that there was little to no improvement in both image retrieval and text retrieval performance when the number of epochs exceeded eight.

### Data Quality
They also examined the impact of data quality by creating subsets of the 3.4B dataset based on image-text similarity, selecting the top 20%, 40%, 60%, and 80% highest-quality data.

<p align="center">
    <img src="./Figure 5.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 4.</strong> Data Quality: Zero-Shot Performances on ImageNet. (a) trained for one epoch. (b) trained for the same number of sampled data.
</div>
</p>

Models trained on these subsets for a single epoch demonstrated that higher quality data subsets yielded superior zero-shot performance on ImageNet. Specifically, the Top40% subset outperformed the entire dataset despite fewer iterations. When comparing datasets with an equal number of samples, the Top40% dataset achieved the best performance, highlighting the importance of data quality in training CLIP models.

<p align="center">
    <img src="./Figure 6.png" width="1100"> 
</p>
<div align="center">
  <strong>Figure 5.</strong> Data Quality: Few-Shot Performances on ImageNet. (a) one epoch. (b) the same number of sampled data.
</div>
</p>

Additionally, when the number of sample data points is the same, higher quality datasets have superior 5-shot and 10-shot performance.

<p align="center">
    <img src="./Figure 7.png" width="1100"> 
</p>
<div align="center">
  <strong>Figure 6.</strong> Data Quality: Retrieval Performances on MSCOCO. (a) one epoch. (b) the same number of sampled data.
</div>
</p>

When it comes to search performance, the top 80% datasets in particular show the most impressive retrieval performance.

## Variants of Vision Transformers
This study examines how the performance of various CLIP models, differentiated by the size of their vision encoders, is influenced by dataset size and the number of sampled data points. They used different vision encoders (ViT-Ti/16, S/16, B/32, B/16, L/16) while keeping text transformers fixed at vit-base. They sampled ten subsets from the full dataset, ranging from 10M to 3.4B samples, maintaining consistent data distribution and quality. Models were trained for one epoch to assess the effect of data quantity, ensuring fair comparison by training all subsets for the same number of iterations.

<p align="center">
    <img src="./Figure 8.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 7.</strong> Various ViTs: Zero-Shot performances with various numbers of sample data
</div>
</p>

Zero-shot performance on ImageNet revealed that larger vision encoders (e.g., ViT-L/16) did not consistently outperform smaller ones when the sample size was under 100M. As data size increased, larger encoders showed better performance.

<p align="center">
    <img src="./Figure 9.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 8.</strong> Various ViTs: Zero-Shot performances with the same number of sampled data: 3.4B
</div>
</p>

As the dataset size grows, the performance difference between larger ViTs and their smaller counterparts becomes more pronounced. Additiallay, accuracy trends across various datasets (ImageNet-R, ImageNet-Sketch, ImageNet-V2, ObjectNet) were nearly linear, except for ImageNet-A, which had a non-linear improvement, highlighting its challenging nature. [(appendix)](https://arxiv.org/abs/2404.08197)

<p align="center">
    <img src="./Figure 10.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 9.</strong> Various ViTs: Linear probing performances with various sizes of vision encoders with the same number of sampled data: 3.4B
</div>
</p>

Linear probing results indicated that for smaller datasets, ViT-L/16 underperformed compared to smaller models, but excelled with more data. Larger ViTs demonstrated better robustness on out-of-distribution datasets.

<p align="center">
    <img src="./Figure 10.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 10.</strong> Various ViTs: Retrieval Performances on MSCOCO
</div>
</p>

Retrieval tasks showed ViT-L/16 performed poorly with less than 100M samples but improved with more data, aligning with zero-shot trends and benefiting more from larger datasets compared to smaller models.

## Comparison of Network Architectures
To effectively choose the best network architectures, they performed a comparison among the various architectures. Previous studies have explored various vision encoders for CLIP, such as ResNet, MLP-Mixer, and ViT, but some architectures like Swin-Transformer and ConvNext haven't been investigated. Here, they compared CNN and vision transformer architectures with similar computational costs, including ViT-B/32, ResNet-50, ConvNext-T, Swin-T, and Mixer-B/32. In Zero-shot, when considering limited data samples, ResNet-50 performs better initially, but ViT-B/32 achieves superior performance with more samples due to its stronger ability to capture global information (see Figure 11(a)). In linear probing, MLP-Mixer outperforms others with fewer samples, but ViT-B/32 excels with larger datasets. ViT and MLP-Mixer show better robustness, likely due to their lower inductive bias, leading to improved generalization (Figure 11(b)). For retrieval tasks, ResNet-50 is better with smaller sample sizes, but ViT-B/32 surpasses it as sample sizes increase. Mixer-B/32 performs poorly in retrieval tasks, making ViT the preferred choice for CLIP's vision encoder across various tasks. 

<p align="center">
    <img src="./Figure 12.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 11.</strong> Performances of the various network architectures
</div>
</p>

## Training Strategies
In this section, the various training strategies for CLIP are explored, including SLIP, FLIP, and a proposed method from this paper called CLIP+Data Augmentation. SLIP enhances the vision encoder through self-supervised learning but is computationally expensive compared to the original CLIP. FLIP masks patches in training images to reduce computation. 
However, CLIP+Data Augmentation aimed to enhance CLIP's vision encoder while mitigating the computational demands associated with previous self-supervised learning approaches. By applying data augmentation directly to input images, they offered a cost-effective alternative, validated across four subsets with 30 epochs of training using techniques like crop&flip, RandAugment, and Stacked RandAugment. The results in Figure 12 demonstrated consistent performance improvements of all three methods over raw CLIP, with no additional computational burden incurred, even enabling comparable performance to larger datasets, exemplified by the Stacked RA model trained on a dataset half the size achieving similar results.

<p align="center">
    <img src="./Figure 13.png" width="400"> 
</p>
<div align="center">
  <strong>Figure 12.</strong> Comparison between various data augmentation for CLIP
</div>
</p>

## Performance Evaluation

### Zero-shot
Their experiments on the ImageNet dataset show that SLIP outperforms CLIP and FLIP when training samples are under one billion, indicating the benefit of self-supervised learning for limited data. However, as sample size increases, CLIP and FLIP surpass SLIP, suggesting that enhancing vision encoders isn't necessary for large datasets. Additionally, SLIP is twice as computationally expensive as CLIP and performs worst in zero-shot tasks when costs are equal. Data augmentation, particularly CLIP + Data Aug, improves performance and generalization on ImageNet and its variants without extra computational costs, especially for larger datasets and multiple epochs of training as presented in Figure 13.

<p align="center">
    <img src="./Figure 14.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 13.</strong> Zero-shot performance with the various training strategies
</div>
</p>

### Linear Probing
In the linear probing evaluation, vision encoders trained with CLIP + Data Aug consistently outperformed the other strategies, particularly on OOD datasets. CLIP and CLIP + Data Aug also showed better robustness than SLIP with similar ImageNet accuracy. Combining CLIP with data augmentation offers a more effective feature extractor, balancing performance, and computation cost. The training results on linear probing performance are shown in Figure 14.

<p align="center">
    <img src="./Figure 15.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 14.</strong> Linear probing performance with the various training strategies
</div>
</p>

### Retrieval
In retrieval tasks, SLIP consistently outperformed CLIP, CLIP + Data Aug, and FLIP on both image and text retrieval across all dataset sizes. Unlike its zero-shot performance, SLIP showed the best results for retrieval tasks as presented in Figure 15, suggesting it is a superior strategy for these tasks despite being less effective for classification.

<p align="center">
    <img src="./Figure 16.png" width="800"> 
</p>
<div align="center">
  <strong>Figure 15.</strong> Retrieval performances with the various training strategies
</div>
</p>

## Conclusion
This study examines how data size, network architecture, and training methods affect CLIP's performance. Their experiments highlight the critical roles of data quantity and quality. They also demonstrate that data augmentation can improve CLIP's performance with minimal additional computational cost. Furthermore, they investigate various network architectures and training strategies, finding that some outperform others depending on the computational budget, emphasizing the need for careful selection.
From my perspective, the balance between computational efficiency and model accuracy is crucial, and exploring adaptive methods could yield significant benefits. Future research could focus on integrating transfer learning with CLIP to enhance domain-specific performance and investigating AutoML techniques for optimal architecture and strategy selection.
