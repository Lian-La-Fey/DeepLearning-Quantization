import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig

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
        low_bits = tensor & 0xF
        unpacked = torch.zeros(tensor.size(0), tensor.size(1) * 2, dtype=torch.int8)
        unpacked[:, ::2] = high_bits
        unpacked[:, 1::2] = low_bits
        return unpacked - 8
    
    def quantize(self, weights: torch.Tensor):
        self.scale.data = torch.abs(weights).max(dim=1, keepdim=True).values / 7
        q_weights = torch.round(weights / self.scale).clamp(-8, 7).to(torch.int8)
        self.packed_weights.data = self.pack_weights(q_weights)
    
    def forward(self, x: torch.Tensor):
        device = x.device
        dequantized_weights = self.unpack_weights(self.packed_weights).to(device) * self.scale.to(device)
        if self.bias is not None:
            self.bias.to(device)
            return F.linear(x, dequantized_weights, self.bias)
        return F.linear(x, dequantized_weights)
            
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

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(f"Model:\n{model}")
print(f"Footprint of the model is: {model.get_memory_footprint() * 1e-6:.2f} MBs.")
print(pipe("What have you prepared for breakfast?", max_new_tokens=20, do_sample=True)[0]['generated_text'])

replace_linear_with_quantized(model, exclude=["lm_head"])

print(f"\nQuantized Model:\n{model}")
print(f"Footprint of the model is: {model.get_memory_footprint() * 1e-6:.2f} MBs.")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("What have you prepared for breakfast?", max_new_tokens=20, do_sample=True)[0]['generated_text'])

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map="auto")
pipe = pipeline("text-generation", model=model_nf4, tokenizer=tokenizer)
print(f"\nModel:\n{model_nf4}")
print(f"Footprint of the model is: {model_nf4.get_memory_footprint() * 1e-6:.2f} MBs.")
print(pipe("What have you prepared for breakfast?", max_new_tokens=20, do_sample=True)[0]['generated_text'])
