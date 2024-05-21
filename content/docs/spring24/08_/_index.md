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

**Linear arithmetic composition.**

<p align="center">
    <img src=./linear_com.png> 
</p>



**Reference tuning-based composition.**


### Mixture-of-Experts

## Mixture of LoRA experts

### Motivations

### Method

### Training

## Results

## Analyisis and Limitations
