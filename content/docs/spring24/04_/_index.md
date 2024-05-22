---
type: docs
bookToc: True
weight: 1
---

# Better & Faster Large Language Models via Multi-token Prediction
*Posted by Jinoh Cho and Seonghyeon Park*
- Authors: Gloeckle et al. 
- Institution : FAIR at Meta, CERMICS Ecole des Ponts ParisTech and LISN Universite Paris-Saclay
  
  
# Preliminaries

### Language Modeling and Next-Token Prediction Task

Standard language modeling involves learning from a large text corpus $x_1, \ldots, x_T$ by implementing a next-token prediction task. The goal is to minimize the cross-entropy loss, defined as:

$$ 
L_1 = - \sum_{t} \log P_{\theta}(x_{t+1} \mid x_{t:1}), 
$$

where $P_{\theta}$ represents our large language model under training. The objective is to maximize the probability of the next token $x_{t+1}$, given the history of previous tokens $x_{t:1} = x_1, \ldots, x_t$.

# Core Idea

### Multi-Token Prediction Task 

In this work, authors propose to learn language modeling from multi-token prediction rather than next-token prediction. At each position of the training corpus, the model is instructed to predict $n$ future tokens at once. Thus, training objective is changed as follow:

$$
L_n = - \sum_{t} \log P_{\theta}(x_{t+n:t+1} \mid x_{t:1}) = - \sum_{t}\sum_{i=1}^{n} \log P_{\theta}(x_{t+i} \mid x_{t:1}). 
$$

This formulation allows the model to learn to predict multiple future tokens simultaneously, enhancing its predictive capabilities and efficiency.

### Memory-Efficient Implementation
Naive 
