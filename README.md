# Quantization: Compressing Models

## Tensor Types

### Integer Types

- `torch.int8`: 8-bit signed integer, typically used when working with quantized models. 
- `torch.int16`: 16-bit signed integer, useful for scenarios requiring more precision than `int8` but still lower precision than typical 32-bit representations.
- `torch.int32`: Standard 32-bit signed integer, used for many typical computation tasks.

### Floating-Point Types

- `torch.float16`: 16-bit floating-point, often used for low-precision models or computations to save memory and improve performance.
- `torch.bfloat16`: 16-bit Brain Floating Point, designed for deep learning tasks with lower precision but retaining higher dynamic range than `float16`.
- `torch.float32`: Standard 32-bit floating-point, commonly used in high-precision computations.

![FP16 & BF16](https://images.contentstack.io/v3/assets/blt71da4c740e00faaa/blt40c8ab571893763a/65f370cc0c744dfa367c0793/EXX-blog-fp64-fp32-fp-16-5_(3).jpg?format=webp)

### Downcasting Error in Dot Product

- Dot product using `float32`: Higher precision result.
- Dot product using `bfloat16`: Lower precision, resulting in a small error compared to `float32`.

## Model Precision

Custom `SimpleCNN` and a pre-trained MobileViT model from Hugging Face were used to demonstrate the differences in memory usage, performance, and numerical precision. The models were tested for memory efficiency, numerical accuracy, and classification performance on an image classification task.

### Logits and Precision Comparison

The logits (outputs) of the models were calculated with inputs of various precisions. For `FP32` and `BF16`, the results were similar, but `FP16` raised an error on the CPU during convolutions, as it's not supported for certain operations without GPU support. 

**Differences**:
- The difference between `FP32` and `BF16` logits was minimal:
  - **Mean difference**: ~0.00022
  - **Max difference**: ~0.00061

This indicates that the `BF16` precision was nearly as accurate as `FP32` but with a significant reduction in memory usage.

### Hugging Face MobileViT Model

We also tested MobileViT in `FP32` and `BF16` formats:
- **FP32 Memory Footprint**: ~22.36 MB
- **BF16 Memory Footprint**: ~11.18 MB

The classification results on the same image were nearly identical, with the predicted class being "tabby cat" in both cases:
- **FP32 Probability**: ~0.2997
- **BF16 Probability**: ~0.2793

This shows that `BF16` offers a significant reduction in memory without a large tradeoff in accuracy.

### Key Takeaways

- **Memory Efficiency**: `BF16` models use significantly less memory than `FP32` models (about half), making them highly efficient for deployment.
- **Numerical Accuracy**: While `BF16` introduces a slight precision loss compared to `FP32`, the difference in logits and predictions is negligible, making it a viable option for large-scale model deployment.
- **FP16 Caution**: Using `FP16` precision on CPUs can result in unsupported operation errors (e.g., for convolution), which is why it is primarily suited for GPU-based models.