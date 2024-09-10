import torch

def get_tensor_info():
    int_types = [torch.int8, torch.int16, torch.int32]
    float_types = [torch.float16, torch.bfloat16, torch.float32]
    
    print("Integer types and ranges:")
    for dtype in int_types:
        print(torch.iinfo(dtype), f"\tsize (Bytes): {int(torch.iinfo(dtype).bits / 8)}")
    
    print("\nFloating-point types and precision:")
    for dtype in float_types:
        print(torch.finfo(dtype), f"\tsize (Bytes): {int(torch.finfo(dtype).bits / 8)}")

def compare_precision():
    value = 2/3
    x_fp32 = torch.tensor(value, dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)
    x_bf16 = x_fp32.to(torch.bfloat16)

    print("\nPrecision comparison:")
    print(f"float32:  {x_fp32.item():.24f}")
    print(f"float16:  {x_fp16.item():.24f}")
    print(f"bfloat16: {x_bf16.item():.24f}")

def downcasting_dot_product():
    print("\nDowncasting: ")
    
    tensor1 = torch.randn(1000, dtype=torch.float32)
    tensor2 = torch.randn(1000, dtype=torch.float32)
    
    print(f"tensor1[:3]:  {tensor1[:3]}\ntensor2[:3]: {tensor2[:3]}")
    dot_fp32 = torch.dot(tensor1, tensor2)
    print(f"Dot product float32: {dot_fp32.item()}", end="\n\n")
    
    tensor1_bf16 = tensor1.to(torch.bfloat16)
    tensor2_bf16 = tensor2.to(torch.bfloat16)
    
    print(f"tensor1_bf16[:3]: {tensor1_bf16[:3]}\ntensor2_bf16[:3]: {tensor2_bf16[:3]}")
    dot_bf16 = torch.dot(tensor1_bf16, tensor2_bf16)
    print(f"Dot product bfloat16: {dot_bf16.item()}")
    
    print(f"\nDowncasting Error: {abs(dot_fp32 - dot_bf16)}")

def main():
    get_tensor_info()
    compare_precision()
    downcasting_dot_product()

if __name__ == "__main__":
    main()