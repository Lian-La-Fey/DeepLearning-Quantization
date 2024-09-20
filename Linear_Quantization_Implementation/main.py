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
    
    if zero_point < q_min:
        zero_point = q_min
    
    if zero_point > q_max:
        zero_point = q_max
    
    return scale, zero_point

# Symmetric Quantization Function
def get_scale(tensor: torch.Tensor, dtype: torch.dtype):
    r_max = tensor.abs().max().item()
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

def get_per_group_scales(tensor: torch.Tensor, dim: int, group_size: int, dtype: torch.dtype = torch.int8):
    tensor_shape = tensor.shape
    assert tensor_shape[dim] % group_size == 0
    assert tensor.dim() == 2
    
    num_groups = tensor_shape[0] * tensor_shape[1] // group_size
    scales = torch.zeros(num_groups)
    zero_points = torch.zeros(num_groups)
    
    tensor = tensor.view(-1, group_size)
    for i in range(num_groups):
        scales[i] = get_scale(tensor.select(dim, i), dtype=dtype)[0]
    
    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)
    
    return scales, zero_points
    
def linear_quantization(tensor: torch.Tensor, scale, zero_point, dtype: torch.dtype, group_size: int = 0):
    if group_size != 0:
        scaled_and_shifted_tensor = tensor.reshape(-1, group_size) / scale + zero_point
        scaled_and_shifted_tensor = scaled_and_shifted_tensor.reshape(tensor.shape)
    else:
        scaled_and_shifted_tensor = tensor / scale + zero_point
    
    rounded_tensor = torch.round(scaled_and_shifted_tensor)
    
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    
    return q_tensor

def linear_dequantization(q_tensor: torch.Tensor, scale, zero_point, group_size: int = 0):
    if group_size != 0:
        return (q_tensor.reshape(-1, group_size) * scale - zero_point).reshape(q_tensor.shape)
    return scale * (q_tensor.float() - zero_point)

def plot_matrices(tensor: torch.Tensor, q_tensor, deq_tensor, err_tensor, name):
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    matrices = [tensor, q_tensor, deq_tensor, err_tensor]
    titles = ["Original Tensor", "Quantized Tensor", "Dequantized Tensor", "Error Tensor"]

    for i, (ax, matrix, title) in enumerate(zip(axs, matrices, titles)):
        if i == 1:
            fmt: str = ".0f"
            vmin, vmax = q_tensor.min(), q_tensor.max()
        else:
            vmin, vmax = tensor.min(), tensor.max()
            fmt: str = ".2f"
        
        sns.heatmap(
            matrix.numpy(), annot=True, fmt=fmt, 
            cmap='inferno', ax=ax, cbar=False, square=True,
            vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"./{name}.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant_type", 
        default="asymmetric",
        choices=["asymmetric", "symmetric", "per_channel", "per_group"],
        type=str
    )
    parser.add_argument("--dim", default=0, choices=[0, 1], type=int)
    parser.add_argument("--group_size", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    tensor = torch.randn((6, 6))
    print(f"Tensor:\n{tensor}")
    
    if args.quant_type == "asymmetric":
        scale, zero_point = get_scale_and_zero_point(tensor, torch.int8)
    elif args.quant_type == "symmetric":
        scale, zero_point = get_scale(tensor, torch.int8)
    elif args.quant_type == "per_channel":
        scale, zero_point = get_per_channel_scales(tensor, dim=args.dim)
    else:
        if args.group_size <= 0:
            raise ValueError("Group size must be greater than 0 for per_group quantization.")
        
        scale, zero_point = get_per_group_scales(tensor, dim=args.dim, group_size=args.group_size)
    
    q_tensor = linear_quantization(tensor, scale, zero_point, torch.int8, group_size=args.group_size)
    print(f"\nQuantized Tensor:\n{q_tensor}")

    deq_tensor = linear_dequantization(q_tensor, scale, zero_point, group_size=args.group_size)
    print(f"\nDequantized Tensor:\n{deq_tensor}")

    err_tensor = abs(tensor - deq_tensor)
    rel_err_tensor = err_tensor / (torch.abs(tensor) + 1e-8)
    
    print(f"\nMean Absolute Quantization Error: {err_tensor.mean()}")
    print(f"Mean Squared Quantization Error: {err_tensor.square().mean()}")
    print(f"Mean Relative Quantization Error: {rel_err_tensor.mean()}")
    print(f"\nQuantization Error:\n{err_tensor}")
    
    plot_matrices(tensor, q_tensor, deq_tensor, err_tensor, args.quant_type)

if __name__ == "__main__":
    main()