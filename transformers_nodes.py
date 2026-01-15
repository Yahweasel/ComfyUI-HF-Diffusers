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

import torch
import transformers

from . import util
from .util import DEVICES, DTYPES

class HFTLoadPipeline:
    """
    Load a HuggingFace Transformers pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": ("STRING", {"default": "text-to-text"}),
                "pipeline_class": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "HuggingFaceH4/zephyr-7b-beta"}),
                "device": (DEVICES,),
                "dtype": (DTYPES,),
                "model_kwargs": ("STRING",),
                "kwargs": ("STRING",),
            }
        }

    RETURN_TYPES = ("HFT_PIPELINE",)
    FUNCTION = "load"

    CATEGORY = "huggingface-transformers"

    def load(
        self, task, pipeline_class, device, dtype, **kwargs
    ):
        kwargs = util.mkkwargs(kwargs)
        model_kwargs = kwargs["model_kwargs"]
        if isinstance(model_kwargs, str):
            model_kwargs = util.mkkwargs(model_kwargs)
            kwargs["model_kwargs"] = model_kwargs
        to_device = util.apply_device(
            kwargs, device, dtype, quant="transformers"
        )
        if "quantization_config" in kwargs:
            model_kwargs["quantization_config"] = kwargs["quantization_config"]
            del kwargs["quantization_config"]
        if to_device:
            kwargs["device"] = to_device

        if pipeline_class != "":
            kwargs["pipeline_class"] = getattr(transformers, pipeline_class)

        pipeline = transformers.pipeline(**kwargs)

        return (pipeline,)

class HFTCreateConversation:
    """
    Create or append to a transformers conversation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": ("STRING", {"default": "user"}),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "previous": ("HFT_CONVERSATION",),
                "image": ("PIL_IMAGE",),
            }
        }

    RETURN_TYPES = ("HFT_CONVERSATION",)
    FUNCTION = "make"

    CATEGORY = "huggingface-transformers"

    def make(self, role, text, previous=None, image=None):
        conv = previous[:] if previous else []
        content = []
        conv.append({
            "role": role,
            "content": content
        })
        if image:
            while isinstance(image, list) or isinstance(image, tuple):
                image = image[0]
            content.append({"type": "image", "image": image})
        if text != "":
            content.append({"type": "text", "text": text})
        return (conv,)

class HFTUnpackConversation:
    """
    Unpack content from a transformers conversation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conversation": ("HFT_CONVERSATION",),
                "index": ("INT", {"default": -1, "min": -1024, "max": 1024}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "PIL_IMAGE")
    FUNCTION = "unpack"

    CATEGORY = "huggingface-transformers"

    def unpack(self, conversation, index):
        part = conversation[index]
        role = None
        text = None
        image = None

        if isinstance(part, str):
            text = part

        if isinstance(part, dict):
            if "role" in part:
                role = part["role"]

            if "content" in part:
                content = part["content"]
                if isinstance(content, str):
                    text = content

                if isinstance(content, list):
                    for cpart in content:
                        if isinstance(cpart, dict) and "type" in cpart:
                            if cpart["type"] == "text":
                                text = cpart["text"]
                            elif cpart["type"] == "image":
                                image = cpart["image"]

        return (role, text, image)

class HFTAutoModel:
    """
    Load a transformers model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_class": ("STRING", {"default": "AutoModel"}),
                "pretrained_model_name_or_path": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "subfolder": ("STRING", {"default": "text_encoder"}),
                "device": (DEVICES,),
                "dtype": (DTYPES,),
                "kwargs": ("STRING",),
            }
        }

    RETURN_TYPES = ("HFT_MODEL",)
    FUNCTION = "load"

    CATEGORY = "huggingface-transformers"

    def load(self, model_class, device, dtype, **kwargs):
        kwargs = util.mkkwargs(kwargs)
        to_device = util.apply_device(kwargs, device, dtype, quant="transformers")
        model = getattr(transformers, model_class).from_pretrained(**kwargs)
        if to_device:
            model.to(to_device)
        return (model,)

class HFTRunPipeline:
    """
    Run a HuggingFace Transformers pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HFT_PIPELINE",),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "kwargs": ("STRING",),
            },
            "optional": {
                "input": ("*",),
            }
        }

    RETURN_TYPES = ("HFT_RESULT", "HFT_CONVERSATION")
    FUNCTION = "generate"

    CATEGORY = "huggingface-transformers"

    def generate(self, pipeline, seed, input=None, **kwargs):
        kwargs = util.mkkwargs(kwargs)
        generator = torch.manual_seed(seed)
        r = pipeline(input, generator=generator, **kwargs)
        ret = [r, None]

        if isinstance(r, list) and len(r) > 0:
            first = r[0]
            if isinstance(first, dict) and "generated_text" in first:
                ret[1] = first["generated_text"]

        return tuple(ret)
