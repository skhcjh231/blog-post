---
type: docs
bookToc: True
weight: 1
---

## MobileNetV4 - Universal Models for the Mobile Ecosystem

## Main Contributions
1. Universal Inverted Bottleneck (UIB) seach block - Unifies the Inverted Bottleneck (IB), ConvNext, Feed Forward Netowork (FFN), and Extra Depthwise variant
2. Mobile MQA - Attention block tailored for mobile accelerators
3. NAS technique to improve performance
4. Achieves Pareto optimal acorss various devices such as CPUs, DSPs, GPUs, and TPUs
5. Novel distillation technique to boost accuracy



This paper propose a compression framework that leverages text information mainly by text-adaptive encoding and training with joint image-text loss. By doing so, they avoid decoding based on text-guided generative models---known for high generative diversity---and effectively utilize the semantic information of text at a global level. 

{{< figure src="./overall_architecture.png" alt="." width="600" height="600" >}}

## Example : Using KaTeX for math equation

KaTeX shortcode let you render math typesetting in markdown document. See [KaTeX](https://katex.org/)

Here is some inline example: {{< katex >}}\pi(x){{< /katex >}}, rendered in the same line. And below is `display` example, having `display: block`
{{< katex display=true >}}
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
{{< /katex >}}
Text continues here!!! 
