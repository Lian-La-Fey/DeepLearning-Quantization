import torch
import warnings

from model import SimpleCNN
from copy import deepcopy
from utils import (
    load_image, 
    print_param_dtype, 
    compute_differences, 
    print_memory_footprint, 
    print_classification_result
)
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

warnings.filterwarnings("ignore")

print("==== SimpleCNN Model Overview ====")
model = SimpleCNN()
print(model)
print_param_dtype(model)

print("\n==== Half-Precision (FP16) Model ====")
model_fp16 = SimpleCNN().half()
print_param_dtype(model_fp16)

print("\n==== Half-Precision (BF16) Model ====")
model_bf16 = deepcopy(model).to(torch.bfloat16)
print_param_dtype(model_bf16)

input_fp32 = torch.randn(1, 1, 28, 28)
input_fp16 = input_fp32.to(dtype=torch.float16)
input_bf16 = input_fp32.to(dtype=torch.bfloat16)

print("\n==== Logits Calculations ====")
logits_fp32 = model(input_fp32)
print(f"Logits FP32: {logits_fp32}")

# for FP16 (RuntimeError on CPU)
try:
    logits_fp16 = model_fp16(input_fp16)
    print(f"Logits FP16: {logits_fp16}")
except Exception as error:
    print(f"Error: {type(error).__name__}: {error}\n")

logits_bf16 = model_bf16(input_bf16)
print(f"Logits BF16: {logits_bf16}")

print("\n==== Difference Between BF16 and FP32 Logits ====")
compute_differences(logits_bf16, logits_fp32)

print("\n==== Hugging Face MobileViT Model Precision ====")
feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
mobilevit_fp32 = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
mobilevit_bf16 = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small", torch_dtype=torch.bfloat16)

print("\n==== Memory Footprints ====")
print_memory_footprint(mobilevit_fp32)
print_memory_footprint(mobilevit_bf16)

image = load_image()
inputs_fp32 = feature_extractor(images=image, return_tensors="pt")
inputs_bf16 = {k: v.to(torch.bfloat16) for k, v in inputs_fp32.items()}

print("\n==== Classification Results ====")
print_classification_result(model=mobilevit_fp32, inputs=inputs_fp32)
print_classification_result(model=mobilevit_bf16, inputs=inputs_bf16)