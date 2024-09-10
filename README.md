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

## Another project


## Another project
