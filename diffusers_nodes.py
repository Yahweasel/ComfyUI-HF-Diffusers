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
import diffusers.image_processor as diffusers_image_processor
import torch
import transformers

DEVICES = ["default", "auto", "cpu"]
DEFAULT_DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICES.append("cuda")
    for i in range(torch.cuda.device_count()):
        DEVICES.append(f"cuda:{i}")
    DEFAULT_DEVICE = "cuda"
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
            kwargs["device_map"] = get_device(device)
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

class HFDLoadPipeline:
    """
    Load a HuggingFace Diffusers pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline_class": ("STRING", {"default": "AutoPipelineForText2Image"}),
                "model": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "device": (DEVICES,),
                "enable_model_cpu_offload": ([False, True],),
                "dtype": (DTYPES,),
                "kwargs": ("STRING",),
            },
            "optional": {
                "vae": ("HFD_AUTOENCODERKL",),
                "text_encoder": ("HFT_MODEL",),
            }
        }

    RETURN_TYPES = ("HFD_PIPELINE", "HFD_AUTOENCODERKL")
    FUNCTION = "load"

    CATEGORY = "huggingface-diffusers"

    def load(
        self, pipeline_class, model, device, enable_model_cpu_offload, dtype,
        **kwargs
    ):
        kwargs = mkkwargs(kwargs)
        to_device = apply_device(
            kwargs, device, dtype,
            enable_model_cpu_offload=enable_model_cpu_offload
        )
        pipeline = getattr(diffusers, pipeline_class).from_pretrained(
            model, **kwargs
        )
        if to_device:
            pipeline.to(to_device)
        if enable_model_cpu_offload:
            pipeline.enable_model_cpu_offload()
        return (pipeline, pipeline.vae)

class HFDLoadLora:
    """
    Load a Lora into a HuggingFace pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HFD_PIPELINE",),
                "pretrained_model_name_or_path_or_dict": ("STRING", {"default": "TheLastBen/Papercut_SDXL"}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "kwargs": ("STRING",),
            }
        }

    RETURN_TYPES = ("HFD_PIPELINE",)
    FUNCTION = "load"

    CATEGORY = "huggingface-diffusers"

    def load(self, pipeline, weight, **kwargs):
        kwargs = mkkwargs(kwargs)
        pipeline.load_lora_weights(**kwargs)
        pipeline.fuse_lora(lora_scale=weight)
        pipeline.unload_lora_weights()
        return (pipeline,)

class HFDAutoencoderKL:
    """
    Load an Autoencoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "autoencoder_class": ("STRING", {"default": "AutoencoderKL"}),
                "pretrained_model_name_or_path": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "subfolder": ("STRING", {"default": "vae"}),
                "device": (DEVICES,),
                "dtype": (DTYPES,),
                "kwargs": ("STRING",),
            }
        }

    RETURN_TYPES = ("HFD_AUTOENCODERKL",)
    FUNCTION = "load"

    CATEGORY = "huggingface-diffusers"

    def load(self, autoencoder_class, device, dtype, **kwargs):
        kwargs = mkkwargs(kwargs)
        to_device = apply_device(kwargs, device, dtype, quant="diffusers")
        vae = getattr(diffusers, autoencoder_class).from_pretrained(**kwargs)
        if to_device:
            vae.to(to_device)
        return (vae,)

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
        kwargs = mkkwargs(kwargs)
        to_device = apply_device(kwargs, device, dtype, quant="transformers")
        model = getattr(transformers, model_class).from_pretrained(**kwargs)
        if to_device:
            model.to(to_device)
        return (model,)

class HFDRunPipeline:
    """
    Run HuggingFace pipelines.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HFD_PIPELINE",),
                "prompt": ("STRING", {"default": "a photo of an astronaut riding a horse on mars"}),
                "negative_prompt": ("STRING",),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "num_inference_steps": ("INT", {"default": 0, "min": 0, "max": 0x10000}),
                "output_type": (["pil", "latent"],),
                "kwargs": ("STRING",),
            },
            "optional": {
                "image": ("PIL_IMAGE",),
                "mask_image": ("PIL_IMAGE",),
                "latents": ("LATENT",),
                "prompt_embeds": ("TENSOR",),
                "pooled_prompt_embeds": ("TENSOR",),
                "negative_prompt_embeds": ("TENSOR",),
                "negative_pooled_prompt_embeds": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("PIL_IMAGE", "LATENT")
    FUNCTION = "generate"

    CATEGORY = "huggingface-diffusers"

    def generate(
        self, negative_prompt, pipeline, seed, num_inference_steps, **kwargs
    ):
        kwargs = mkkwargs(kwargs)
        if "prompt_embeds" in kwargs:
            del kwargs["prompt"]
        if "negative_prompt_embeds" not in kwargs and negative_prompt != "":
            kwargs["negative_prompt"] = negative_prompt
        if num_inference_steps > 0:
            kwargs["num_inference_steps"] = num_inference_steps
        with torch.no_grad():
            generator = torch.manual_seed(seed)
            r = pipeline(
                generator=generator,
                **kwargs
            )
        if kwargs["output_type"] == "pil":
            return (r.images[0], None)
        else:
            return (None, r.images)

class HFDEncodePrompt:
    """
    Encode a prompt using a HuggingFace pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HFD_PIPELINE",),
                "prompt": ("STRING", {"default": "a photo of an astronaut riding a horse on mars"}),
                "kwargs": ("STRING",),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "TENSOR")
    FUNCTION = "encode"

    CATEGORY = "huggingface-diffusers"

    def encode(self, pipeline, prompt, kwargs):
        kwargs = mkkwargs(kwargs)
        r = list(pipeline.encode_prompt(prompt, **kwargs))
        while len(r) < 4:
            r.append(None)
        return tuple(r)

class HFDVAEDecode:
    """
    VAE decoding using HuggingFace diffusers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "vae": ("HFD_AUTOENCODERKL",),
            }
        }

    RETURN_TYPES = ("PIL_IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "huggingface-diffusers"

    def decode(self, latents, vae):
        image_processor = diffusers_image_processor.VaeImageProcessor(
            vae_scale_factor=2**(len(vae.config.block_out_channels) - 1)
        )
        with torch.no_grad():
            image = vae.decode(
                (latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype),
                return_dict=False
            )[0]
        image = image_processor.postprocess(image)
        return (image,)

class HFDVAEEncode:
    """
    VAE encoding using HuggingFace diffusers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("PIL_IMAGE",),
                "vae": ("HFD_AUTOENCODERKL",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "huggingface-diffusers"

    def encode(self, image, vae):
        image_processor = diffusers_image_processor.VaeImageProcessor(
            vae_scale_factor=2**(len(vae.config.block_out_channels) - 1)
        )
        if isinstance(image, list):
            image = image[0]
        latents = image_processor.preprocess(
            image,
            height=image.height,
            width=image.width
        ).to(device=vae.device, dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(
                latents,
                return_dict=False
            )[0].sample()
        latents *= vae.config.scaling_factor
        return (latents,)
