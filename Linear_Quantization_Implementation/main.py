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

def get_per_channel_scales(tensor: torch.Tensor, dim, dtype: torch.dtype = torch.int8):
    scale_size = tensor.shape[dim]
    scales = torch.zeros(scale_size)
    zero_points = torch.zeros(scale_size)
    
    for i in range(scale_size):
        scales[i] = get_scale(tensor.select(dim, i), dtype=dtype)[0]
    
    scales_shape = [1] * tensor.dim()
    scales_shape[dim] = -1
    scales = scales.view(scales_shape)
    zero_points = zero_points.view(scales_shape)
    
    return scales, zero_points
    
def linear_quantization(tensor: torch.Tensor, scale, zero_point, dtype: torch.dtype):
    scaled_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor)
    
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    
    return q_tensor

def linear_dequantization(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)

def plot_matrices(tensor, q_tensor, deq_tensor, err_tensor, name):
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    matrices = [tensor, q_tensor, deq_tensor, err_tensor]
    titles = ["Original Tensor", "Quantized Tensor", "Dequantized Tensor", "Error Tensor"]

    for i, (ax, matrix, title) in enumerate(zip(axs, matrices, titles)):
        sns.heatmap(matrix.numpy(), annot=True, fmt=".2f", cmap='viridis', ax=ax, cbar=False, square=True)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"./{name}.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Choose between asymmetric and symmetric quantization.")
    parser.add_argument(
        "--quant_type", 
        default="asymmetric",
        choices=["asymmetric", "symmetric", "per_channel"],
        type=str
    )
    parser.add_argument(
        "--dim", 
        default=0,
        choices=[0, 1], 
        type=int
    )
    args = parser.parse_args()
    
    # tensor = torch.randn((4, 4))
    tensor = torch.tensor([
        [224.3, 112.4, -28.6, 53.9],
        [456.2, 110, 15.5, -29.6],
        [114.4, 256.3, 127.3, -99.4],
        [-24.3, 54.3, 65.8, 48.6]
    ])
    print(f"Tensor:\n{tensor}")
    
    if args.quant_type == "asymmetric":
        scale, zero_point = get_scale_and_zero_point(tensor, torch.int8)
    elif args.quant_type == "symmetric":
        scale, zero_point = get_scale(tensor, torch.int8)
    else:
        scale, zero_point = get_per_channel_scales(tensor, dim=args.dim)
    
    q_tensor = linear_quantization(tensor, scale, zero_point, torch.int8)
    print(f"\nQuantized Tensor:\n{q_tensor}")

    deq_tensor = linear_dequantization(q_tensor, scale, zero_point)
    print(f"\nDequantized Tensor:\n{deq_tensor}")

    err_tensor = abs(tensor - deq_tensor)
    print(f"\nMean Absolute Quantization Error: {err_tensor.mean()}")
    print(f"Mean Squared Quantization Error: {err_tensor.square().mean()}")
    print(f"\nQuantization Error:\n{err_tensor}")
    
    plot_matrices(tensor, q_tensor, deq_tensor, err_tensor, args.quant_type)

if __name__ == "__main__":
    main()