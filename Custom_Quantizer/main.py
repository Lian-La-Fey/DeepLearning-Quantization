import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class QuantizedLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantized_weights = nn.Parameter(
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8), 
            requires_grad=False
        )
        self.scales = nn.Parameter(torch.randn(out_features), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(out_features), requires_grad=True) if bias else None
    
    def quantize_per_channel(self, weights: torch.Tensor):
        max_vals = torch.abs(weights).max(dim=1, keepdim=True).values
        self.scales.data = max_vals / 127
        quantized_weights = torch.round(weights / self.scales).clamp(-128, 127).to(torch.int8)
        self.quantized_weights.data = quantized_weights
    
    def forward(self, x: torch.Tensor):
        device = x.device
        dtype = x.dtype
        dequantized_weight = self.quantized_weights.to(device).to(dtype) * self.scales.to(device).view(-1, 1).to(dtype)
        
        if self.bias is not None:
            bias = self.bias.to(device).to(dtype)
            return F.linear(x, dequantized_weight, bias)
        return F.linear(x, dequantized_weight)

def replace_linear_with_quantized(module: nn.Module, exclude: list):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name not in exclude:
            bias = child.bias
            new_layer = QuantizedLinearLayer(child.in_features, child.out_features, bias is not None)
            new_layer.quantize_per_channel(child.weight)
            if bias is not None:
                new_layer.bias = bias
            setattr(module, name, new_layer)
        else:
            replace_linear_with_quantized(child, exclude=exclude)

model_id = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(f"Model:\n{model}")
print(f"Footprint of the model is: {model.get_memory_footprint() * 1e-6:.2f} MBs.")
print(pipe("What have you prepared for breakfast?", max_new_tokens=20, do_sample=True)[0]['generated_text'])

replace_linear_with_quantized(model, exclude=["lm_head"])

print(f"\nQuantized Model:\n{model}")
print(f"Footprint of the model is: {model.get_memory_footprint() * 1e-6:.2f} MBs.\n")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("What have you prepared for breakfast?", max_new_tokens=20, do_sample=True)[0]['generated_text'])