---
type: docs
bookToc: True
weight: 1
---
# **Accelerating Transformers via Conditional Computation: As Aspect of Mixture of Depths**
*Posted by: Inkwan Hwang, Minjae Park*

---

<p align="center">
    <img src=./Pondering_Transformer.jpg> 
</p>
This image was produced using DALL·E 3.

## **Introduction**
“Choice and concentration” is an effective strategies for achieving success in problems. Sometimes, it is not necessary to consume same amount of effort and time into all problems. Expending energy on trivial issues may fail to concentrate on what truly matters. Similarly, in language models, there is a technique that does not focus equally on all tokens but allocates less budget to non-essential tokens. This technique is called conditional computation.

In this post, We will explain conditional computation strategies for Transformers, focusing on a technology announced this year called **Mixture-of-Texture.**

paper:  [<U>Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</U>](https://arxiv.org/abs/2404.02258){:target="_blank"}

paper: <a href="https://arxiv.org/abs/2404.02258" target="_blank"> Mixture-of-Depths: Dynamically allocating compute in transformer-based language models </a>


Let's dive in!

## **Understanding the problem: Uniform computation in Transformers**
These days, most language models are based on Transformers, and we stack these blocks to make big models. When given an input sequence, tokens pass through these blocks to predict the next token. The problem is that the models spread computations uniformly across input sequences. Transformers use the same amount of computation for essential tokens as for non-essential ones. For instance, predicting a token within a sentence is cheaper than predicting the first token of the next sentence. Researchers want to address this issue by making Transformers focus on important tokens by allocating less computing resources.

## **Conditional Computation for transformers**
- Early exiting
  early exiting
  Early Exit method is a method when the model decides to end computation on a given token, allowing it skips the remaining layers. Difference between MoD is, MoD can choose whether skip middle layer or not, but Early Exit method can't.
  
- CoLT5

- Mixture of Experts (MoE)

  MoE is an model which consists of parallel expert models which is fitted to certain domains. Like MoD, token-level routing decisions are made across the network depth. Difference between MoD is, MoD chooses path to transformer or to residual connection, MoE chooses path to transformer(Expert) or to transformer(Expert) or both.
  
## **Overview to Mixture-of-Depths (MoD)**

Self-attention + MLP, Residual Connection 중 고른다는 내용. MoE는 넓이를 줄였지만, MoD는 깊이에 해당한다는 내용.

<p align="center">
    <img src=./Mixture-of-Depths.png> 
</p>

MoE is an model which consists of parallel expert models which is fitted to certain domains.
Like MoD, token-level routing decisions are made across the network depth.
Difference between MoD is, MoD chooses path to transformer or to residual connection, MoE chooses path to transformer(Expert) or to transformer(Expert) or both.

## **Routing schemes**
Routing implementation is the most crucial part of MoD. The authors compare three routing strategies, demonstrating that MoD is an efficient approach.

<p align="center">
    <img src=./Routing_Schemes.png> 
</p>

### Token-choice routing

Token-choice routing is a method where each tokens select the path it will follow. The router produces probability distributions for each token across the computational paths. Based on this distribution, each token chooses its preferred path at each layer.
  
In token-choice routing, tokens have the flexibility to select their path, allowing for dynamic processing. However, this can lead to path balancing issues as all tokens might preger on the same path. It causes potential overloads on specific paths. To mitigate it, auxility loss is used to ensure that most tokens do not prefer on a single path.
  
### Expert-choice routing

Expert-choice routing is the reverse of token-choice routing. Similar to token-choice routing, the router produces a probability distribution for each token. In expert-choice routing, instead of tokens selecting their paths, each path selects the top-k tokwns based on the tokens' preferences.

Using this method ensures that each paths receives k tokens, maintauing balance among the paths. However, some tokens may not be selected beacuse there might be common tokens that multiple paths prefer.

### Expert-choice MoD

This method applies expert-choice routing but uses only a single expert. Since only a single path is utilized, if $k$ is less than the sequence length, not all tokens need to undergo self-attention and MLP computation.

이러이러한 이유로 expert-choice routing을  사용한다~~

## **Implementation detail**

capacity에 관한 설명.

### 1. Calculate routing weight
{{< katex display=true >}}
x^{l+1}_i=\begin{cases}r^{l}_i f_i(\tilde{X}^l)+x^{l}_i, &    \text{if } r^{l}_i >  P_\beta(R^l)\\x^{l}_i, & \text{if }r^{l}_i <  P_\beta(R^l)\end{cases}
{{< /katex >}}
### 2. Select top-k tokens

논문 요약

## **Open source MoD**

https://github.com/astramind-ai/Mixture-of-depths

## **Conclusion and discussion**

논문 요약 + 내 생각

## **Some resources**

참고문헌 정리
