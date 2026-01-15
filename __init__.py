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

from .diffusers_nodes import *
from .transformers_nodes import *

NODE_CLASS_MAPPINGS = {
    "HFDLoadPipeline": HFDLoadPipeline,
    "HFDLoadLora": HFDLoadLora,
    "HFDAutoencoderKL": HFDAutoencoderKL,
    "HFDRunPipeline": HFDRunPipeline,
    "HFDEncodePrompt": HFDEncodePrompt,
    "HFDVAEDecode": HFDVAEDecode,
    "HFDVAEEncode": HFDVAEEncode,

    "HFTLoadPipeline": HFTLoadPipeline,
    "HFTCreateConversation": HFTCreateConversation,
    "HFTUnpackConversation": HFTUnpackConversation,
    "HFTAutoModel": HFTAutoModel,
    "HFTRunPipeline": HFTRunPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFDLoadPipeline": "HF Diffusers load pipeline",
    "HFDLoadLora": "HF Diffusers load LoRA",
    "HFDAutoencoderKL": "HF Diffusers load AutoencoderKL (VAE)",
    "HFDRunPipeline": "HF Diffusers run pipeline",
    "HFDEncodePrompt": "HF Diffusers encode prompt",
    "HFDVAEDecode": "HF Diffusers VAE decode",
    "HFDVAEEncode": "HF Diffusers VAE encode",

    "HFTLoadPipeline": "HF Transformers load pipeline",
    "HFTCreateConversation": "HF Transformers create conversation",
    "HFTUnpackConversation": "HF Transformers unpack conversation",
    "HFTAutoModel": "HF Transformers load model",
    "HFTRunPipeline": "HF Transformers run pipeline",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
