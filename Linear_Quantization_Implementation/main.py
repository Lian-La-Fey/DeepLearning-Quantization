import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Asymmetric Quantization Function
def get_scale_and_zero_point(tensor: torch.Tensor, dtype: torch.dtype):
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    
    r_min = tensor.min().item()
    r_max = tensor.max().item()
    
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = int(round(q_min - (r_min / scale)))
    
    if zero_point < q_min or zero_point > q_max:
        zero_point = q_min
    
    return scale, zero_point

# Symmetric Quantization Function
def get_scale(tensor: torch.Tensor, dtype: torch.dtype):
    r_max = tensor.max().item()
    q_max = torch.iinfo(dtype).max
    return r_max / q_max, 0

def linear_quantization(tensor: torch.Tensor, scale, zero_point, dtype: torch.dtype):
    scaled_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor)
    
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    
    return q_tensor

def linear_dequantization(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)

def plot_matrices(tensor, q_tensor, deq_tensor, err_tensor):
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    matrices = [tensor, q_tensor, deq_tensor, err_tensor]
    titles = ["Original Tensor", "Quantized Tensor", "Dequantized Tensor", "Error Tensor"]

    for i, (ax, matrix, title) in enumerate(zip(axs, matrices, titles)):
        sns.heatmap(matrix.numpy(), annot=True, fmt=".2f", cmap='viridis', ax=ax, cbar=False, square=True)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Choose between asymmetric and symmetric quantization.")
    parser.add_argument(
        "--quant_type", 
        default="asymmetric",
        choices=["asymmetric", "symmetric"], 
    )
    args = parser.parse_args()
    
    tensor = torch.randn((4, 4))
    print(f"Tensor:\n{tensor}")
    
    if args.quant_type == "asymmetric":
        scale, zero_point = get_scale_and_zero_point(tensor, torch.int8)
    else:
        scale, zero_point = get_scale(tensor, torch.int8)
    
    q_tensor = linear_quantization(tensor, scale, zero_point, torch.int8)
    print(f"\nQuantized Tensor:\n{q_tensor}")

    deq_tensor = linear_dequantization(q_tensor, scale, zero_point)
    print(f"\nDequantized Tensor:\n{deq_tensor}")

    err_tensor = abs(tensor - deq_tensor)
    print(f"\nMean Absolute Quantization Error: {err_tensor.mean()}")
    print(f"\nQuantization Error:\n{err_tensor}")
    
    plot_matrices(tensor, q_tensor, deq_tensor, err_tensor)

if __name__ == "__main__":
    main()