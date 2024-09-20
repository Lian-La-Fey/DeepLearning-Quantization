import torch
import warnings
import re
import time

from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.quanto import QuantizedTransformersModel, qint8

# https://stackoverflow.com/questions/78905356/how-to-properly-quantize-model-with-quanto
def named_module_tensors(module, recurse=False):
    for name, tensor in module.named_parameters(recurse=recurse):
        if hasattr(tensor, "_data") or hasattr(tensor, "_scale"):
            if hasattr(tensor, "_data"):
                yield name + "._data", tensor._data
            if hasattr(tensor, "_scale"):
                yield name + "._scale", tensor._scale
        else:
            yield name, tensor

    for name, buffer in module.named_buffers(recurse=recurse):
        yield name, buffer

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) of a single element of a given dtype.
    """
    if dtype == torch.bool:
        return 1 / 8  # 1 bit per boolean
    bit_size = int(re.search(r'\d+', str(dtype)).group())
    return bit_size // 8

def compute_module_sizes(model):
    """
    Computes the size of each submodule in a model based on its tensors.
    """
    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
        size = tensor.numel() * dtype_byte_size(tensor.dtype)
        parts = name.split(".")
        for i in range(len(parts) + 1):
            submodule_name = ".".join(parts[:i])
            module_sizes[submodule_name] += size

    return module_sizes
    

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")
print(f"The model size is {compute_module_sizes(model)[''] * 1e-6:.2f} MBs")

input_text = "Turkey is located in "
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

start_time = time.time()
outputs = model.generate(input_ids)
end_time = time.time()
print(f"Inference time for the original model: {end_time - start_time:.3f} seconds")
print(tokenizer.decode(outputs[0]))

qmodel = QuantizedTransformersModel.quantize(model, weights=qint8, activations=None)
print(f"\nThe quantized model size is {compute_module_sizes(qmodel)[''] * 1e-6:.2f} MBs")

start_time = time.time()
outputs = qmodel.generate(input_ids)
end_time = time.time()
print(tokenizer.decode(outputs[0]))
print(f"Inference time for the quantized model: {end_time - start_time:.3f} seconds")