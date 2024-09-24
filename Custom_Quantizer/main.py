import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class QuantizedLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantized_weights = nn.Parameter(
            torch.zeros((out_features, in_features), dtype=torch.int8), 
            requires_grad=False
        )
        self.scales = nn.Parameter(torch.randn(out_features), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(out_features), requires_grad=False) if bias else None
    
    def quantize_per_channel(self, weights: torch.Tensor):
        max_vals = weights.abs().max(dim=1, keepdim=True).values
        self.scales.data = max_vals / 127
        quantized_weights = torch.round(weights / self.scales).clamp(-128, 127).to(torch.int8)
        self.quantized_weights.data = quantized_weights
    
    def forward(self, x: torch.Tensor):
        device = x.device
        dtype = x.dtype
        weight = self.quantized_weights.to(device=device, dtype=dtype)
        output = F.linear(x, weight) * self.scales.to(device=device, dtype=dtype).view(1, -1)
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
            output = output + bias
        return output

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

def generate_text(model, tokenizer, input_text, max_new_tokens=20, device="cpu"):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

device = "cuda"
model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# print(f"Model:\n{model}")
print(f"Footprint of the model is: {model.get_memory_footprint() * 1e-9:.2f} GBs.")
text = "What have you prepared for breakfast?"
start_time = time.time()
gen_text = generate_text(model, tokenizer, text, max_new_tokens=20, device=device)
end_time = time.time()
print(f"Inference time for the original model: {end_time - start_time:.3f} seconds")
print(gen_text, end='\n\n')


replace_linear_with_quantized(model, exclude=["lm_head"])

# print(f"Quantized Model:\n{model}")
print(f"Footprint of the model is: {model.get_memory_footprint() * 1e-9:.2f} GBs.")
start_time = time.time()
gen_text = generate_text(model, tokenizer, text, max_new_tokens=20, device=device)
end_time = time.time()
print(f"Inference time for the custom quantized model: {end_time - start_time:.3f} seconds")
print(gen_text)