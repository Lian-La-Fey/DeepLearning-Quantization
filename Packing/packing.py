import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

class Quantized4BLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.packed_weights = nn.Parameter(
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8), 
            requires_grad=False
        )
        self.scale = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True) if bias else None
    
    def pack_weights(self, tensor: torch.Tensor):
        assert tensor.dtype == torch.int8
        tensor = tensor + 8
        high_bits = (tensor[:, ::2] & 0xF) << 4
        low_bits = tensor[:, 1::2] & 0xF
        packed = high_bits | low_bits
        return packed
    
    def unpack_weights(self, tensor: torch.Tensor):
        high_bits = (tensor >> 4) & 0xF
        low_bits = tensor & 0xF # 0xF (0000 1111) -> mask
        unpacked = torch.zeros(tensor.size(0), tensor.size(1) * 2, dtype=torch.int8)
        unpacked[:, ::2] = high_bits
        unpacked[:, 1::2] = low_bits
        return unpacked - 8
    
    def quantize(self, weights: torch.Tensor):
        self.scale.data = weights.abs().max(dim=1, keepdim=True).values / 7
        q_weights = torch.round(weights / self.scale).clamp(-8, 7).to(torch.int8)
        self.packed_weights.data = self.pack_weights(q_weights)
    
    def forward(self, x: torch.Tensor):
        device = x.device
        dtype = x.dtype
        weight = self.unpack_weights(self.packed_weights).to(device=device, dtype=dtype)
        output = F.linear(x, weight) * self.scale.to(device=device, dtype=dtype).view(1, -1)
        if self.bias is not None:
            output += self.bias
        return output
        
def replace_linear_with_quantized(module: nn.Module, exclude: list):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name not in exclude:
            bias = child.bias
            new_layer = Quantized4BLinearLayer(child.in_features, child.out_features, bias is not None)
            new_layer.quantize(child.weight)
            if bias is not None:
                new_layer.bias.data = bias.data
            setattr(module, name, new_layer)
        else:
            replace_linear_with_quantized(child, exclude=exclude)

def generate_text(model, tokenizer, input_text, device="cpu"):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    start_time = time.time()
    output = model.generate(input_ids, max_new_tokens=20, do_sample=True)
    end_time = time.time()
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Inference time for the original model: {end_time - start_time:.3f} seconds")
    print(generated_text, end='\n\n')

device = "cuda"
model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_text = "What have you prepared for breakfast?"
print(f"Footprint of the model is: {model.get_memory_footprint() * 1e-6:.2f} MBs.")
generate_text(model, tokenizer, input_text, device=device)

replace_linear_with_quantized(model, exclude=["lm_head"])

print(f"Footprint of the quantized model is: {model.get_memory_footprint() * 1e-6:.2f} MBs.")
generate_text(model, tokenizer, input_text, device=device)