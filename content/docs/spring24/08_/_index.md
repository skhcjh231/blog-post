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

<p align="center">
    <img src=./Method1.png align="center" width="400">
</p>

<details>
    <summary>See related formulas</summary>
        <b>Symbols</b> <br/>
        input $x \in \mathbb{R} ^ {L \times d}$ <br/>
        L: sequence length <br/>
        d: dim of $x$ <br/>
        Multi attention layer : $$\mathcal{f}_{Attn} (\centerdot)$$ <br/>
        Feed forward neural network layer: $$\mathcal{f}_{FFN} (\centerdot)$$   <br/>
        LN: layer normalization <br/>
        Trained LORAs $$\Omega = \left\{ \Delta \Theta \right\}^N_{i=0}$$ <br/>
        learnable gating function $$\mathcal{G} (\centerdot)$$ <br/>
        The weight of the $i^{th}$ trained LorA $$\mathcal{G}_i (\centerdot)$$ <br/>
        Concatenation operation: $$\oplus$$ <br/>
        Learnable parameter $e \in \mathbb{R} ^ {N^2 \times L \times d}$ <br/>
        Learnable temperature scalar $\tau$ <br/>
        <br/>
        <b>Freezing part</b>
        $$x^\prime_{\theta} = x + \mathcal{f}_{Attn} (LN(x)|\theta)$$ <br/>
        $$\mathbf{F}_\theta (x) = x^\prime_{\theta} + \mathcal{f}_{Attn} (LN(x^\prime_{\theta})|\theta)$$ <br/>
        <br/>
        <b>LoRA part</b>
        $$x^\prime_{\Delta \Theta_i} = x + \mathcal{f}_{Attn} (LN(x)|\Delta \Theta_i)$$ <br/>
        The output of each LoRA $$\mathbf{E} _{\Delta \Theta_i} (x) = x^\prime_{\Delta \Theta_i} + \mathcal{f}_{FFN} (LN(x^\prime_{\Delta \Theta_i})|\Delta \Theta_i)$$ <br/>
        The output of all LoRA $$\mathbf{E}_\Omega (x) = Normalization(\mathbf{E}_{\Delta \Theta_0} (x) \oplus \ldots \oplus \mathbf{E}_{\Delta \Theta_{N-1}} (x)) \in \mathbb{R} ^ {N \times L \times d}$$ <br/>
        Flatten and dot product operation $$\epsilon = Flatten(\mathbf{E}_\Omega (x))^T \centerdot e,  \epsilon \in \mathbb{R} ^ N$$ <br/>
        Gate value for each LoRA $$\mathcal{G} (\epsilon_i) = \frac {exp(^{\epsilon_i} /_ \tau)} {\displaystyle\sum_{j=1}^N {exp(^{\epsilon_j} /_ \tau)}} $$ <br/>
        Final output of the gating function $${\tilde{\mathbf{E}}_\Omega (x)} = \displaystyle\sum_{i=0}^N {\mathcal{G} (\epsilon_i) \centerdot \mathbf{E} _{\Delta \Theta_i} (x)} , {\tilde{\mathbf{E}}_\Omega (x)} \in \mathbb{R} ^ {L \times d} $$ <br/>
        <b>Final output of Transformer block</b>
        $$\mathcal{O}(x) = {\mathbf{F}_\theta (x)} + {\tilde{\mathbf{E}}_\Omega(x)} $$ 
</details> 

### Training
The final loss function used in MoLE is as follows:
<p align="left">
    <img src=./training5.png width="200">
</p>
Alpha is a coefficient for weight balancing. 
<br/>

**Gating Balacing Loss**
<p align="center">
    <img src=./training1.png width="400">
</p>
As shown in Figure 5 (a), the average entropy of the distribution probabilities from the gating functions gradually decreases as training progresses. In Figure 5 (b), we can see a gating probability of 64% for LoRA β among the three LoRAs, indicating that the gating function tends to converge to a state where it assigns large weights to well-performing LoRAs in the early stages. This can result in a significantly larger impact from a few specific LoRAs compared to others, potentially leading to biased outcomes. <br/>
<br/>
To avoid this, the author created a gating balancing loss.<br/>
The gating balancing loss helps prevent bias by ensuring that the loss value decreases as the model becomes less biased. <br/>
<br/>
<p align="left">
    <img src=./training2.png width="200">
</p>
<details>
    <summary>See related Symbols</summary>
    M: the nu of blocks where gating functions are placed <br/>
    N: num of LoRAs
</details>     
<br/>

**Domain-specific Loss**
<br/>
In V&L, Using a loss in CLIP(Radford et al,20221b) <br/>
<p align="left">
    <img src=./training3.png width="300">
</p>

In NLP, Using a loss in FLAN-T5(Chung et al,2022)
<p align="left">
    <img src=./training4.png width="200">
</p>

## Results

## Analyisis and Limitations












