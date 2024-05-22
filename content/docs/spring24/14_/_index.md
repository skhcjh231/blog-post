---
type: docs
bookToc: True
weight: 1
---

# BinaryDM: Towards Accurate Binarization of Diffusion Model

*Authors: Xingyu Zheng, Haotaong Qin, Xudong Ma, Mingyuan Zhang, Haojie Hao, Jiakai Wang, Zixiang Zhao, Jinyang Guo, Xianglong Liu*  
*Posted by Junhyuk So, Juncheol Shin*

## **Preliminary**
### **Diffusion**
<p align="center">
  <img src="./diffusion.png" alt="."> 
</p>
<p align="center">
  Diffusion model
</p>
Diffusion models learn how to remove gaussian noise added to original image. 
Equation below shows how forward process proceeds. During the forward process, Gaussian noise is gradually added to original image for {{< katex >}}T{{< katex >}} times. Strength of the noise is controlled by the term {{< katex >}}\beta{{< katex >}}. {{< katex >}}x_t{{< katex >}} denotes corrupted image at time step {{< katex >}}t{{< katex >}}.  
In the reverse process, diffusion model tries to restore original image by estimating conditional distribution {{< katex >}}q(x_{t} | x_{t-1}){{< katex >}} with p_\theta(x_{t-1} | x_{t}). Reparameterization trick is used to estimate mean and variance of gaussian the distribution  
<p align="center">
  {{< katex >}}
    q(x_{t} | x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_{t}I) \\p_\theta(x_{t-1} | x_{t}) = \mathcal{N}(x_{t-1}; \tilde{\mu}_\theta(x_t, t) , \tilde{\beta}_{t}I )
  {{< /katex >}} 
</p>

### **Quantization**  
<p align="center">
  <img src="./quantization.png" alt="." width="500" height="300" > 
</p>
<p align="center">
  Quantization example
</p>
Quantization is an optimization technique which restricts data(weights, activations) in low precision. Not only does it reduce the memory footprint, but it also enables accelerated computations given hardware support to low-precision arithmetic.  

Values are represented as follows when linear symmetric quantization is applied:
<p align="center">
  {{< katex >}}
    q(x) = clamp(\lfloor \frac{x}{\Delta},-2^{b-1}, 2^{b-1}-1) \rceil,\ \ \Delta = \frac{\alpha}{2^{b-1}-1} 
  {{< /katex >}} 
</p>
x is value to be quantized and {{< katex >}}\alpha{{< katex >}} denotes trunctaion range. {{< katex >}}b{{< katex >}} denotes bit-width. {{< katex >}}\Delta{{< katex >}} is step-size, interval between quantization levels.
Binarization is extreme case of quantization, which only utilizes 1 bit ({{< katex >}}b = 1{{< katex >}}).

## **Motivation** 
While diffusion models achieved great success in generation tasks, its iterative nature act as a bottleneck to real-world application. Data must processed through heavy diffusion models for multiple steps and requires huge latency and memory footprint.

Quantization is one reasonable choice for the optimization of diffusion models. Especially when binarization is applied to weight, floating point operations can be substituted with cheap addition and memory footprint of the model can be reduced greatly. 

However binary models are hard to binarize in two aspects. One arises from perspective of representation, as binarization is extreme case of quantization which only uses 1bit to represent data. Naive binarization introduces severe degredation in quality of output. Another aspect arises from perspective of optimization. Training becomes unstable with binarized representation and hinders convergence of the model.

This work tackles binarization of diffusion models by handling aformentioned two aspects. By introducing *Learnable Multi-basis Binarizer(LMB)* and *Low-rank representation mimicking(LRM)*, BinaryDM is able to achieve **16.0× and 27.1× reductions on FLOPs and size**.  

## **Methodology**
### **Learnable Multi-basis binarizer**
Typical binarization of the weight can be described as follows:  
<p align="center">
  {{< katex >}}
    w^{bi} = \sigma sign(w) = \left\{\begin{matrix}
 \sigma, & if w\geq 0 \\ 
 -\sigma, & otherwise 
\end{matrix}\right.
  {{< /katex >}} 
</p>
In this work, authors propose a learnable multi-basis binarizer(LMB) to maintain quality of representations. Instead of using single base, multiple bases are utilized for the binarization. {{< katex >}}\sigma_{\MakeUppercase{\romannumeral1}}{{< katex >}} and {{< katex >}}\sigma_{\MakeUppercase{\romannumeral2}}{{< katex >}} denote learnable scalar values and initialized as {{< katex >}}\sigma_{\MakeUppercase{\romannumeral1}}^0 = \frac{\left \| w  \right \|}{n}{{< katex >}} and {{< katex >}}\sigma_{\MakeUppercase{\romannumeral2}}^0 = \frac{\left \| w - \sigma_{\MakeUppercase{\romannumeral1}}sign(w)  \right \|}{n}{{< katex >}} 
<p align="center">
  {{< katex >}}
    w_{LMB}^{bi} = \sigma_{\MakeUppercase{\romannumeral1}} sign(w) + \sigma_{\MakeUppercase{\romannumeral2}} sign(w - \sigma_{\MakeUppercase{\romannumeral1}} sign(w) 
  {{< /katex >}} 
</p>  
Gradient of learnable scalar values can be computed as follows:  
<p align="center">
  {{< katex >}}
    \frac{\partial w_{LMB}^{bi}}{\partial \sigma_{\MakeUppercase{\romannumeral1}}} = \left\{\begin{matrix}
 sign(w)(1-\sigma_{\MakeUppercase{\romannumeral2}} sign(w))), & if \ \sigma_{\MakeUppercase{\romannumeral1}}sign(w) \in (w-1, w+1)  \\ 
 sign(w), & otherwise 
\end{matrix}\right. 
  {{< /katex >}} 
</p>  

During the inference, computation for each bases are indepedent to each other and can be parallely computed. Thus, diffusion model can be fully accelerated with LMB.  


It is important to note that LMB is applied at only crucial parts of difusion model. Only modules where features cale is greater than or equal to {{< katex >}}\frac{1} {2}{{< katex >}} input scale. In other words, some of the first consecutive layers and last consecutive layers are binarized with LMB. The binarized modules close to input or output play important role, as they extract patterns from original data or directly influence the final result.  Figure below shows result of naive binarization and LMB applied to weights.  
<p align="center">
  <img src="./lmb.png" alt="." width="500" height="300" > 
</p>
<p align="center">
  Weight quantizatio result of naive binarization and LMB
</p>

### **Low-rank representation mimicking**
Binarization of weights makes the training hard and hiders convergence. Since full precision model is available, it is natural to align intermediate representations of binarized diffusion model and original model as additional supervision. However, fine-grained alignment of high-dimensional representation leads to blurry optimization direction and binarization makes the model hard to mimic full precision model.   

Authors propose Low-rank Representation Mimicking(LRM) to handle these problems. LRM utilize principal component analysis(PCA) to project representations to low-rank space. Then representaiton aligning is done in low-rank space by minimizing mean squared error (MSE)  

First, covariance matrix of ith module {{< /katex >}}C_i{{< /katex >}} is computed with representation of full-precision diffusion model,{{< /katex >}}\hat{\epsilon}_{\theta_{i}}^{FP}(x_t, t) \in \mathbb{R}^{h\times w\times c}{{< /katex >}} . Then eigenvector matrix {{< /katex >}}E_i{{< /katex >}} can be obtained and first {{< /katex >}}\lfloor\frac{c}{k}\rceil{{< /katex >}} column eighenvectors are used to compute projected representations, {{< katex >}}{{< katex >}} and {{< katex >}}{{< katex >}}.  
<p align="center">
  {{< katex >}}
    C_{i} = \frac{1}{{(h \times w)}^2}\hat{\epsilon}_{\theta_{i}}^{FP}(x_t, t) \hat{\epsilon}_{\theta_{i}}^{FP^T}(x_t, t), \\
    E_{i}^{T}C_{i}E_{i} = \Lambda_{i}, \\
    \mathcal{R}_i^{FP}(x_t, t) = E_{i}^{\lfloor \frac{c}{K}\rceil}\hat{\epsilon}_{\theta_{i}}^{FP}(x_t, t), \\
    \mathcal{R}_i^{bi}(x_t, t) = E_{i}^{\lfloor \frac{c}{K}\rceil}\hat{\epsilon}_{\theta_{i}^{bi}}^{bi}(x_t, t)
  {{< /katex >}} 
</p>  
LRM loss and total loss can be expressed as follows:  

Since computation of transformation matrix {{< katex >}}E_i{{< katex >}} is expensive, it is computed with the first batch of input and fixed during entire traning. As shown in the figure below, LRM stabilize training process, accelerating convergence.  
<p align="center">
  <img src="./lrm.png" alt="." width="500" height="300" > 
</p>

### **Progressive binarization**
Despite the enhanced methodology, training process remains slow and unstable. Authors additionally apply progressive binarization strategy to further stabilize convergence. M/2 th time stepping module is quantized in first iteration and m/2-i-th and m/2+i-th modules are quantized in next i-th iteration. As show in the figure, benefit coming from progressive binarization is significant compared to baseline traning process.  
<p align="center">
  <img src="./progressive.png" alt="." width="500" height="300" > 
</p>
