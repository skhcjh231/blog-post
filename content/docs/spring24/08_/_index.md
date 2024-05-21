---
type: docs
bookToc: True
weight: 1
---

# MIXTURE OF LORA EXPERTS
<!--
제목으로 바꾸기
처음에 어그로 끌 수 있는 내용 먼저
-->
## Background

### What is LoRA?
_LoRA is a methodology for effective fine-tuning large-scale pretrained models._

Models such as OPT, LLaMA, and CLIP demonstrate remarkable performance when fine-tuned for various downstream tasks. However, full fine-tuning of these massive models requires substantial computational resources. LoRA enables parameter-efficient fine-tuning by keeping the pretrained model's weights frozen and adding trainable low-rank decomposition matrices.

<p align="center">
    <img src=./LoRA.png> 
</p>

In the above figure, only the matrices A and B are trained, with dimensions (d x r) and (r x d) respectively. By setting r << d, the number of parameters to be trained can be reduced. These trained matrices are then added to the existing pretrained weights, allowing tuning without affecting the inference speed of the original model.

### LoRAs Composistion

The common solution to further improve the performance of LoRA is to compose multiple trained LoRAs. Research on LoRA composition can be broadly categorized into the following two methodologies.

**Linear arithmetic composition.** It is a method of directly adding multiple LoRAs. This approach is simple and has been effective in the NLP and Vision-Language domain, but it can result in the loss of pre-trained model's generative capabilities or the individual characteristics of each LoRA.

**Reference tuning-based composition** tackles the above limitations of linear arithmetic method by introducing gradient fusion and controllable sampling, but is requires retaining when incorporating different LoRAs or creating new masks, which results non-trivial computational costs.


<p align="center">
    <img src=./lora_comp.png> 
</p>






### Mixture-of-Experts

## Mixture of LoRA experts

### Motivations
1. Direct linear arithmetic composition reduced the generative power of the model, while normalized linear arithmetic composition retained the generative power of the model but lost its LORA character.
<p align="center">
    <img src=./motiv1_1.png align="center" width="48%">
    <img src=./motiv1_2.png align="center" width="48%">
    <figcaption align="center">
<p align="center">
    <img src=./motiv1_3.png>
</p>
2. Each layer of the trained LoRA represented a unique characteristic, which cumulatively defined the overall properties of the LoRA.
<p align="center">
    <img src=./motiv2_1.png align="center" width="48%">
    <img src=./motiv2_2.png align="center" width="48%">
    <figcaption align="center">
</p>
Right: Observed that different layers of LoRA encode distinct features, such as dog coat color and facial features.

left: When evaluated on a subset of datasets, there were significant differences in performance across the different layers of LoRA. 

**So, The conjecture is that adjusting the characteristics by varying the layer-specific weights according to the desired domain objective will result in a more effective composition of trained LORAs.**
### Method
| !(https://github.com/effml-postech/blog-post/tree/main/content/docs/spring24/08_/Method1.png) | This is a sample text next to the image. The image and the text are aligned side by side. |
    
### Training

## Results

## Analyisis and Limitations
