import requests
import torch
import torch.nn.functional as F

from PIL import Image

def print_param_dtype(model):
    print("\nModel Parameters DataTypes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    print()

def compute_differences(logits_bf16, logits_fp32):
    mean_diff = torch.abs(logits_bf16 - logits_fp32).mean().item()
    max_diff = torch.abs(logits_bf16 - logits_fp32).max().item()
    print(f"Mean diff: {mean_diff} | Max diff: {max_diff}")
    
def print_memory_footprint(model):
    mem_footprint = model.get_memory_footprint()
    print(f"Footprint of the {model.dtype} model in MBs: {mem_footprint / 1e+6}")

def load_image():
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(img_url, stream=True).raw)
    return image

def print_classification_result(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class_index = probs.argmax(-1).item()
    max_prob = probs.max().item()
    print(f"Predicted class {model.dtype}: {model.config.id2label[predicted_class_index]} with probability {max_prob:.4f}")