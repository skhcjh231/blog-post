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
* Decoding K tokens requires K sequential runs of the model. 

⇒ LLM inference is slow

To address this issue, speculative decoding has been introduced as a promising approach to accelerate LLM inference without altering the output quality. This method leverages two key observations about LLM inference: 
* Many tokens can be predicted with minimal computational overhead. 
* LLM inference is predominantly constrained by memory bandwidth rather than arithmetic computations. 

However, existing speculative decoding still face limitations, such as high inference latency and suboptimal token acceptance rates. 
This papaer proposed Kangaroo to address this challenge.


## Backgrounds
### What is speculative decoding?

Speculative decoding is an apporach to accelerate LLM inference. 
*Draft model: Additional model to accelerate inference (also known drafter) 
*Verifier or target model: Original large LLM
<p align="center">
    <img src='./speculative decoding.png' width="700">
</p>

**Left model**: The target LLM generates K tokens in K forward steps, which is a "serial" process.

**Right model**: The drafter generates tokens in parallel. Each generated token is then verified with a verification step.

`Speculative decoding` can be implemented through methods such as independent drafting and self-drafting.

**Independent Drafting**: This approach uses a small language model (LM) from the same series as the target LLM.

- Requires additional training and increases computational complexity by integrating separate target and drafting models.

**Self-Speculative Decoding**: This method utilizes the target LLM itself.

- Employs techniques such as Blockwise Decoding, Medusa, and early exiting to reduce computational burden.
- Computational efficiency can also be achieved through layer skipping.

## Layer Early Exiting

The author has proposed a novel self-speculative decoding framework, named Kangaroo. Kangaroo utilizes double early exiting mechanisms, layer early exiting and draft early exiting. Layer early exiting suggests the equivalent self-draft small model exiting early from the fixed shallow layers of the large LLM and connecting to an adapter network to generate draft tokens. While this strategy is commonly used for self-speculative decoding frameworks, Kangaroo has further investigated suitable architectures of the adapter module and offered a low-cost approach to train a lightweight model. Draft early exiting uses early exiting at suitable points during the drafting phase to avoid unnecessary computational overhead on more challenging tokens.

### Evaluation Metrics

Speculative decoding is often evaluated using two primary metrics: walltime speedup ratio and compression rate. Given a speculative decoding algorithm, we assume that {{< katex >}}N{{< \katex >}} tokens should be generated via the drafting model. As the drafting model predicts multiple tokens in each decoding step and multiple tokens can be accepted by the large model in a step, we record the number of accepted tokens per step as a list {{< katex >}} S = \[s_1, s_2, \dots, s_{|S|}\] {{< \katex >}}, where {{< katex >}} \sum_k s_k = N {{< \katex >}} and {{< katex >}} |S| {{< \katex >}} denotes the number of steps. Then, the compression rate (CR) is defined as:
{{< katex display = true >}}
\text{CR} = \frac{1}{|S|} \sum_k s_k.
{{< \katex >}}
However, once a draft token is rejected during the verification, all subsequent tokens sampled from the drafting model will be discarded. Therefore, CR does not accurately reflect the acceptance levels for tokens at varying distances, and the author has proposed a new evaluation metric named _consistent token acceptance rate_.

The consistent token acceptance rate {{< katex >}} \text{CTAR}(w) {{< \katex >}} is calculated as:
{{< katex display = true >}}
\text{CTAR}(w) = \frac{1}{|S|} \sum_k \mathbb{I} (s_k - w > 0),
{{< \katex >}}
where {{< katex >}}\mathbb{I}(\cdot){{< \katex >}} denotes an indicator function and {{< katex >}} w {{< \katex >}} denotes a window size. CTAR can be interpreted as a rate of the number of steps to accept over {{< katex >}} w {{< \katex >}} tokens.

{{< figure src="./CTAR.png" alt="." width="600" height="600" >}}

Figure 1 represents the empirical CTARs for {{< katex >}}w = 1,2,\dots,6 {{< \katex >}} of self-drafting speculative decoding frameworks including Kangaroo on the mathematical reasoning subtask of Spec-Bench [1].

[1] Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. _arXiv preprint arXiv:2401.07851_, 2024.

### Adapter Network as Self-Drafting Model

We assume the target LLM has {{<katex>}} L {{<\katex>}} layers and the self-draft model {{<katex>}} \mathcal{M}^s {{<\katex>}} consists of shallow sub-network {{<katex>}} \mathcal{M}^b[:l] {{<\katex>}}, which is first {{<katex>}} l {{<\katex>}} layers of the target LLM {{<katex>}}\mathcal{M}^b{{<\katex>}}, and a adapter network {{<katex>}} \mathcal{A} {{<\katex>}}


## Draft Early Exiting

## Experiments

## Discussion and Conclusion
