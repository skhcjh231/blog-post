---
type: docs
bookToc: True
weight: 1
---

# Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting
* Author: Liu F, Tang Y, Liu Z, Ni Y, Han K, Wang Y

Written by Nayoung Kwon and Jiwoong Im

## Introduction

The growing demand for rapid and efficient inference in large language models (LLMs) faces a significant bottleneck
* decoding K tokens requires K sequential runs of the model. 

⇒ LLM inference is slow

To address this issue, speculative decoding has been introduced as a promising approach to accelerate LLM inference without altering the output quality. This method leverages two key observations about LLM inference: 
1) many tokens can be predicted with minimal computational overhead. 
2) LLM inference is predominantly constrained by memory bandwidth rather than arithmetic computations. 

However, existing speculative decoding still face limitations, such as high inference latency and suboptimal token acceptance rates. 
This papaer proposed Kangaroo to address this challenge.


## Backgrounds

<p align="center">
    <img src='./speculative decoding.png' width="700">
</p>

## Layer Early Exiting

The author has proposed a novel self-speculative decoding framework, named Kangaroo. Kangaroo utilizes double early exiting mechanisms, layer early exiting and draft early exiting. Layer early exiting suggests the equivalent self-draft small model exiting early from the fixed shallow layers of the large LLM and connecting to an adapter network to generate draft tokens. While this strategy is common for self-speculative decoding frameworks, Kangaroo has further investigated suitable architectures of the adapter module and offered a low-cost approach to train a lightweight model. Draft early exiting uses early exiting at suitable points during the drafting phase to avoid unnecessary computational overhead on more challenging tokens.

### Evaluation Metrics

Speculative decoding is often evaluated using two primary metrics: walltime speedup ratio and compression rate. Given a speculative decoding algorithm, we assume that {{< katex >}}N{{< \katex >}} tokens should be generated via the drafting model. As the drafting model predicts multiple tokens in each decoding step and multiple tokens can be accepted by the large model in a step, we record the number of accepted tokens per step as a list {{< katex >}} S = \[s_1,\, s_2,\, \dots,\, s_{|S|}\] {{< \katex >}}, where {{< katex >}} \sum_k s_k = N {{< \katex >}} and {{< katex >}} |S| {{< \katex >}} denotes the number of steps. Then, the compression rate (CR) is defined as:
{{< katex display = true >}}
\text{CR} = \frac{1}{|S|} \sum_k s_k.
{{< \katex >}}
However, once a draft token is rejected during the verification, all subsequent tokens sampled from the drafting model will be discarded. Therefore, CR does not accurately reflect the acceptance levels for tokens at varying distances, and the author has proposed a new evaluation metric named _consistent token acceptance rate_.

The consistent token acceptance rate {{< katex >}} \text{CTAR}(w) {{< \katex >}} is calculated as:
{{< katex display = true >}}
\text{CTAR}(w) = \frac{1}{|S|} \sum_k \
{{< \katex >}}



## Draft Early Exiting

## Experiments

## Discussion and Conclusion
