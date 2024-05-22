---
type: docs
bookToc: True
weight: 1
---

## MobileNetV4 - Universal Models for the Mobile Ecosystem
*Posted by JoonSeok Kim and DongGyu Kim*
- Qin, Danfeng and Leichner, Chas and Delakis, Manolis and Fornoni, Marco and Luo, Shixin and Yang, Fan and Wang, Weijun and Banbury, Colby and Ye, Chengxi and Akin, Berkin and others
- arXiv preprint arXiv:2404.1051


## Main Contributions
MobileNetV4 targets designing neural networks for mobile devices. Main objectives of designing inference models for mobile devices are 
- Acceptable test performance on widely-used datasets such as ImageNet-1k
- Low inference latency for utilization in mobile devices
- Minimization of the number of parameters for low memory utilization on mobile platforms
- Minimization in the number of MACs for high enegy efficiency
This paper mainly focuses on lowering inference latency while maintining the test accuracy up to SOTA mobile neural net performance. Since it targets mobile platforms, it analyzes performance of various mobile hardwares, and designs a neural network to fit the harwares maximum performance. The designing process was done by the NAS technique, where the intantiation of UIB blocks were set as the search space. The main contributions of this work can be states as follows. 
1. Universal Inverted Bottleneck (UIB) seach block - Unifies the Inverted Bottleneck (IB), ConvNext, Feed Forward Netowork (FFN), and Extra Depthwise variant
2. Mobile MQA - Attention block tailored for mobile accelerators
3. NAS technique to improve performance
4. Achieves Pareto optimal acorss various devices such as CPUs, DSPs, GPUs, and TPUs
5. Novel distillation technique to boost accuracy. Achieves 87% accuracy on ImageNet-1k and 39x smaller model size

## Preliminaries - Roofline Model and Hardware Efficiency
Algorithm running on hardware is composed of two parts - memory access and computation. The computation time is determined by the computation requirement and hardware performance. 
{{< katex display=true >}}
\text{runtime\_computation} = \frac{\text{Number of Operations}}{\text{FLOPS}}
{{< /katex >}}

Algorithm runtime can be limited by the memory access bottleneck or communication overhead

{{< katex display=true >}}
\text{runtime\_communication} = \frac{\text{Number of IO Bytes}}{\text{Bandwidth}}
{{< /katex >}}

Hardware performance is determined by the upper bound of the computation time or memory access latency

{{< katex display=true >}}
\text{performance} = \max(\text{runtime\_computation}, \text{runtime\_communication})
{{< /katex >}}

Below Fig. 1(a) and Fig. 1(b) illustrates the roofline model and its characteristics
<p align="center">
    <img src='./Fig0a.png' width="900">
</p>
    
<p align="center">
    Fig. 1 (a) Roofline Model
</p>

<p align="center">
    <img src='./Fig0b.png' width="900">
</p>
    
<p align="center">
    Fig. 1 (b) Roofline Model with Ceiling
</p>


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
    Fig. 4. Universal Inverted Bottleneck (UIB) blocks
</p>

The main advantage of UIB is its adaptability and flexibility, that mitigates seach complexity. Optional Depthwise (DW) convolution blocks are inserted before the expansion layer, and between the expansion and projection layer. In the NAS procedure, common components such as the pointwise expansion and projection are shared and DWs are added as search options. UIB has four possible instantiations as follows.
- Inverted Bottleneck (IB) : Spatial mixing on the expanded features activations, and provides higher model capacity
- ConvNext : Cheaper spatial mixing before the expansion with larger kernel size
- ExtraDW : Inexpensive increase of the network depth and the receptive field. Combined benefits of ConvNext and IB
- FFN : Stack of two 1x1 pointwise convolutions. Accelerator-friendly operation

## Mobile MQA
<p align="center">
    <img src='./Table1.png' width="500">
</p>
    
<p align="center">
    Table 1. Efficiency Gains by MQA
</p>

This paper considers the Operational Intensity (OI), which is the ratio of arithmetic operations to memory access, to enhance efficiency of vision models on mobile accelerators. Here, Multi-Query Attention (MQA) is proposed instead od Multi-Head Self Attention (MHSA), which is simplified by utilization of shared keys and values across all heads. This sharing of keys and values reduces memory access hence improving OI, especially when the batch size is small. Large language models does not have significant accuracy drop in this MQA case. Table 1 shows that by adding MHSA and MQA, the performace accuracy has increased whereas the inference latency for MQA is approximately x39 lower than that of MHSA. Hence, MQA can accelerate better in the mobile environment, with negligible performance degradation. 

The Spatial Reduction Attention (SRA) is applied, hence incorporating asymmetric spatial down-sampling, to downscale keys and values, and not queries. In hybrid models, there is a certain correlation between spatially adjacent tokens, hence necessitating spatial mixing convolution filters.

## Refined NAS for Enhanced Architectures
As shown above, the insitantiation of UIB blocks are in the neural architecture search process. TuNAS was adopted for the paper's search strategy. The paper uses a two-stage search operation, the coarse-grained search and fine-grained serach to address the variance in parameter counts between UIB's depthwise layers and other search options. The course-grained search process involves determining optimal filter sizes with fixed parameters. The fine-grained stage searches for the UIB's layer configuration. 

## Results

<p align="center">
    <img src='./Table5.png' width="900">
</p>
    
<p align="center">
    Table 5. Classification results on ImageNet-1k
</p>

<p align="center">
    <img src='./Table6.png' width="500">
</p>
    
<p align="center">
    Table 6. Object Detection results on the COCO validation set
</p>

Results on the classification performance on ImageNet-1k dataset show that the MobileNetV4 achieves the highest performance and smallest latency compareed to other models on various mobile platforms such as CPUs and DSPs of mobile phones. While other models have closely competitive latency with the MobileNetV4 model, their latency is much higher. 

The effectiveness of MobileNetV4 as backbone networks are tested on the COCO object detection experiment. The number of MACs were set to be similiar, and the Retina framework was used as the object detector. As the same as classification, MobileNetV4 achieves highest performance compared to other mobile-target modles, with the lowest CPU latency. Hence, the ability of MobileNetV4 for mobile devices can be shown. 

## Conclusion

- This paper proposes the MobileNet V4 series, a universal high-efficiency model that operates efficiently across a wide range of mobile environments.
- By introducing a new Universal Inverted Bottleneck and Mobile MQA layer and applying an enhanced NAS recipe, MobileNet V4 achieves near Pareto-optimal performance on various hardware, including mobile CPUs, GPUs, DSPs, and dedicated accelerators.
- Additionally, using the latest distillation techniques, it demonstrates cutting-edge performance in mobile computer vision by achieving 87% ImageNet-1K accuracy with a latency of 3.8ms on the Pixel 8 EdgeTPU.
- The paper also presents a theoretical framework and analysis for understanding the model's universality across heterogeneous devices, providing guidance for future design. 
