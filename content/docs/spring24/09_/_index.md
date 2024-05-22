---
type: docs
bookToc: True
weight: 1
---

## MobileNetV4 - Universal Models for the Mobile Ecosystem
*Posted by JoonSeok Kim and DongGyu Kim*


## Main Contributions
1. Universal Inverted Bottleneck (UIB) seach block - Unifies the Inverted Bottleneck (IB), ConvNext, Feed Forward Netowork (FFN), and Extra Depthwise variant
2. Mobile MQA - Attention block tailored for mobile accelerators
3. NAS technique to improve performance
4. Achieves Pareto optimal acorss various devices such as CPUs, DSPs, GPUs, and TPUs
5. Novel distillation technique to boost accuracy. Achieves 87% accuracy on ImageNet-1k and 39x smaller model size

## Hardware-Independent Pareto Efficiency
<p align="center">
    <img src='./Fig2.png' width="900">
</p>
    
<p align="center">
    Fig. 2. Ridge Points and Latency/Accuracy Tradeoffs
</p>

<p align="center">
    <img src='./Fig3.png' width="900">
</p>
    
<p align="center">
    Fig. 3. Op Cost vs. Ridge Point
</p>
This research focuses on efficiency on various hardware targets such as DSPs, CPUs, and GPUs. To find whether the hardware is limited by its memory bottlenecked or compute bottlenecked, the Roofline Model of that hardware must be investigated. It is defined by the harware's peak computational throughput and its peak memory bandwidth. The optima of that hardware can be found in its ridge point, which is defined by the hardware's ratio of Peak MACs to Peak memory bandwidth. The algorithm's accuracy and latency are swept by the ridge point on various hardware on Fig. 2, and Fig. 3. The roodline model of MobileNetV4 achieves highest Pareto-optimal performance compared to other MobileNet models. 

MobileNetV4 is designed to achieve Pareto optimal and hence balances MAC operations and memory bandwidth. The initial layers are designed with high MAC intensity, so as to improve model capacity and downstream accuracy. The end layers use identically-sized FC layers to maximize accuracy. These two initial and end layers are balances so that MobileNetV4 should not see slowdowns at any hardware. 

## Universal Inverted Bottlenecks (UIB)
<p align="center">
    <img src='./Fig4.png' width="900">
</p>
    
<p align="center">
    Fig. 4. Univesal Inverted Bottleneck (UIB) blocks
</p>

The main advantage of UIB is its adaptability and flexibility, that mitigates seach complexity. Optional Depthwise (DW) convolution blocks are inserted before the expansion layer, and between the expansion and projection layer. In the NAS procedure, common components such as the pointwise expansion and projection are shared and DWs are added as search options. UIB has four possible instantiations as follows.
- Inverted Bottleneck (IB) : Spatial mixing on the expanded features activations, and provides higher model capacity
- ConvNext : Cheaper spatial mixing before the expansion with larger kernel size
- ExtraDW : Inexpensive increase of the network depth and the receptive field. Combined benefits of ConvNext and IB
- FFN : Stack of two 1x1 pointwise convolutions. Accelerator-friendly operation

## Mobile MQA
<p align="center">
    <img src='./Table1.png' width="900">
</p>
    
<p align="center">
    Table 1. Efficiency Gains by MQA
</p>


## Design of MobileNetV4 

## Results

<p align="center">
    <img src='./Table5.png' width="900">
</p>
    
<p align="center">
    Table 5. Classification results on ImageNet-1k
</p>

## Enhanced Distillation Recipe

## Conclusion


This paper propose a compression framework that leverages text information mainly by text-adaptive encoding and training with joint image-text loss. By doing so, they avoid decoding based on text-guided generative models---known for high generative diversity---and effectively utilize the semantic information of text at a global level. 

{{< figure src="./overall_architecture.png" alt="." width="600" height="600" >}}

## Example : Using KaTeX for math equation

KaTeX shortcode let you render math typesetting in markdown document. See [KaTeX](https://katex.org/)

Here is some inline example: {{< katex >}}\pi(x){{< /katex >}}, rendered in the same line. And below is `display` example, having `display: block`
{{< katex display=true >}}
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
{{< /katex >}}
Text continues here!!! 
