---
type: docs
bookToc: True
weight: 1
---

# Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length
Authors: Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, Chunting Zhou

Reviewer: Hyunho Kook

![Megalodon](./Megalodon.jpg)

## 1. Introduction
Recently, Large Language Models (LLMs) have been gaining popularity. The impressive performance and versatility demonstrated by large models above a certain level have started to be utilized in various fields. However, as the size of the models grows, the size of the data that the models are expected to process is also increasing. Examples of this include processing currently open issues by inputting a GitHub repository or translating a large volume of books without losing context. In addition, the ability to maintain context and carry on a conversation for an extended period within a single chat is also sometimes required. The transformer, which is the foundation model of modern LLMs, exhibits vulnerabilities in this regard. Firstly, since it uses KV cache, memory usage increases rapidly as the sequence length grows, and it has a computational complexity proportional to the square of the sequence length.

To address this problem, the authors propose a method that inherits and advances MEGA (exponential moving average with gated attention), the predecessor of this paper. The overall contributions are as follows:

1. **CEMA (Extending Multi-dimensional Damped EMA to Complex Domain)**, an extension of Exponential Moving Average (EMA) to the complex domain, is proposed.
2. **Timestep Normalization, an extension of Group Norm to the timestep domain**, is proposed as an alternative to Layer Norm.
3. **Normalized Attention**, which performs normalization during attention computation, is proposed.
4. **2-hop Residual Connection**, which composes residual connections in 2-hop units, is proposed.

By employing these methods, the authors have created a transformer architecture that is linear with respect to context length. They have also addressed the issues encountered in the previous research, MEGA, which were (i) low performance and (ii) the need for different architecture structures for each data type or task.

## 2. MEGA (exponential Moving avErage with Gated Attention)

## 3. Methods
### CEMA (Extending Multi-dimensional Damped EMA to Complex Domain)
### Timestep Normalization
### Normalized Attention
### 2-hop Residual Connection

## 4. Experiments

## 5. Comparison with related works

Here is the cleaned-up English translation:

There have been several similar related studies:

1. Efficient Attention: FlashAttention optimized the GPU computation of Attention, showing advantages in speed without changing the existing mechanism. Additionally, there have been attempts to increase the Context Length by converting the Attention Mechanism to a linear one or compressing the KV cache.

2. Structured State Space Model: A notable study in this area is Mamba, which added mechanisms like Selective Scan to a State Space model with linear time complexity, enabling it to process Long Context.

However, these studies have limitations. Even if we accept that Flash Attention did not change the Attention Mechanism itself, Linear Attention, KV cache compression, and State Space Models have shown significantly lower performance on general benchmarks, although they may perform better than standard Transformers in long contexts.

## 6. Discussion
In my opinion, there are a few potential limitations that are not extensively discussed in the paper:

1. **Reliance on CEMA for Out-of-Chunk Context**: The self-attention mechanism in MEGALODON is applied within each chunk. For data that falls completely outside the chunk boundaries, the model relies solely on CEMA for processing. This limitation could potentially hinder the model's ability to handle long-range dependencies that span across multiple chunks.

2. **Complexity of the Architecture**: Compared to the traditional Transformer layer, the MEGALODON architecture is considerably more complex. It requires the computation of EMA, including the complex domain, for each token. Additionally, several normalization and attention components have been introduced, such as Timestep Normalization, which further increases the complexity of the model compared to the previous works.

4. **Limited Exploration of Downstream Tasks**: While the paper demonstrates the effectiveness of MEGALODON on long-context question answering tasks from the Scrolls dataset, the range of downstream tasks explored is relatively narrow. Evaluating the model's performance on a broader set of tasks, such as summarization, dialogue generation, and composition, would provide a more comprehensive assessment of its capabilities and potential limitations.

Despite these limitations, MEGALODON presents a promising direction for efficient long-context modeling. In my opinion, this kind of efficent and linear processing of **memory** can be a breakthrough for long-context LLMs.
