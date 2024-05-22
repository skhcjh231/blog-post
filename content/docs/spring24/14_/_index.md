---
type: docs
bookToc: True
weight: 1
---

# BinaryDM: Towards Accurate Binarization of Diffusion Model
## Preliminary
### Diffusion
Diffusion models learn how to remove gaussian noise added to original image. 
Equation below shows how forward process proceeds. In the forward process, Gaussian noise is gradually added to original image for T times. Strength of the noise is controlled by the term \beta x_t denotes corrupted image at time step t.  
In the reverse process, diffusion model tries to restore original image by estimating conditional distribution (). Reparameterization trick is used to estimate mean and variance of gaussian the distribution  

### Quantization  
Quantization is an optimization technique which restricts data(weights, activations) in low precision. Not only does it reduce the memory footprint, but it also enables accelerated computations given hardware support to low-precision arithmetic.  

Values are quantized and represented as integers as follows:

Binarization is extreme case of quantization, which only utilizes 1 bit.

## Motivation 
While diffusion models achieved great success in generation tasks, its iterative nature act as a bottleneck to real-world application. Data must processed through heavy diffusion models for multiple steps and requires huge latency and memory footprint.

Quantization is one reasonable choice for the optimization of diffusion models. Especially when binarization is applied to weight, floating point operations can be substituted with cheap addition and memory footprint of the model can be reduced greatly. 

However binary models are hard to binarize in two aspects. One arises from perspective of representation, as binarization is extreme case of quantization which only uses 1bit to represent data. Naive binarization introduces severe degredation in quality of output. Another aspect arises from perspective of optimization. Training becomes unstable with binarized representation and hinders convergence of the model.

This work tackles binarization of diffusion models by handling aformentioned two aspects. By introducing Learnable Multi-basis Binarizer(LMB) and Low-rank representation mimicking(LRM), BinaryDM is able to achieve 16.0× and 27.1× reductions on FLOPs and size.  

## Methodology
### Learnable Multi-basis binarizer
### Low-rank representation mimicking
### Progressive binarization
