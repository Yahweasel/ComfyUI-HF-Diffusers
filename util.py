# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or distribute
# this software, either in source code form or as a compiled binary, for any
# purpose, commercial or non-commercial, and by any means.
#
# In jurisdictions that recognize copyright laws, the author or authors of this
# software dedicate any and all copyright interest in the software to the public
# domain. We make this dedication for the benefit of the public at large and to
# the detriment of our heirs and successors. We intend this dedication to be an
# overt act of relinquishment in perpetuity of all present and future rights to
# this software under copyright law.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json

import diffusers
import torch
import transformers

DEVICES = ["default", "auto", "cpu"]
DEFAULT_DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICES.append("cuda")
    for i in range(torch.cuda.device_count()):
        DEVICES.append(f"cuda:{i}")
    if torch.cuda.device_count() > 0:
        DEFAULT_DEVICE = "cuda:0"
DTYPES = ("default", "float32", "bfloat16", "float16", "bitsandbytes_8bit", "bitsandbytes_4bit")

def get_device(device):
    if device == "default":
        return DEFAULT_DEVICE
    else:
        return device

def mkkwargs(kwargs):
    """
    Return kwargs appropriate for HuggingFace models.
    """
    ret = {}
    if "kwargs" in kwargs and kwargs["kwargs"] != "":
        ret = json.loads(kwargs["kwargs"])
    for key in kwargs:
        if key == "kwargs":
            continue
        if key not in ret and kwargs[key] is not None:
            ret[key] = kwargs[key]
    return ret

def apply_device(kwargs, device, dtype, enable_model_cpu_offload=False, quant="pipeline"):
    """
    Apply device and dtype properties to the kwargs. Quantizes in pipeline,
    transformers, or diffusers mode. Returns the device that the result should
    be moved to with `to`, or `None` if not needed.
    """
    device = get_device(device)
    to_device = None
    if not enable_model_cpu_offload:
        if ":" in device:
            to_device = device
        else:
            kwargs["device_map"] = device
    kwargs["torch_dtype"] = torch.bfloat16

    if dtype[0:13] == "bitsandbytes_":
        if quant == "transformers" or quant == "diffusers":
            qc = {}
            if dtype == "bitsandbytes_4bit":
                qc["load_in_4bit"] = True
            else:
                qc["load_in_8bit"] = True
            if quant == "transformers":
                qc = transformers.BitsAndBytesConfig(**qc)
            else:
                qc = diffusers.BitsAndBytesConfig(**qc)
            kwargs["quantization_config"] = qc

        else: # "pipeline"
            qc = {
                "quant_backend": dtype
            }
            if dtype == "bitsandbytes_4bit":
                qc["quant_kwargs"] = {"load_in_4bit": True}
            else:
                qc["quant_kwargs"] = {"load_in_8bit": True}
            kwargs["quantization_config"] = diffusers.PipelineQuantizationConfig(**qc)

    elif dtype == "float32":
        kwargs["torch_dtype"] = torch.float32
    elif dtype == "float16":
        kwargs["torch_dtype"] = torch.float16

    return to_device

