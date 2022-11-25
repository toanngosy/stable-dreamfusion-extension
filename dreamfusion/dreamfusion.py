import gc
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint
from PIL import features
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, whoami
from six import StringIO

from extensions.stable_dreamfusion_extension.dreamfusion import conversion
from extensions.stable_dreamfusion_extension.dreamfusion.df_config import DreamfusionConfig
#from extensions.stable_dreamfusion_extension.dreamfusion.xattention import save_pretrained
from modules import paths, shared, devices, sd_models

try:
    cmd_dreamfusion_models_path = shared.cmd_opts.dreamfusion_models_path
except:
    cmd_dreamfusion_models_path = None


def get_df_models():
    model_dir = os.path.dirname(cmd_dreamfusion_models_path) if cmd_dreamfusion_models_path else paths.models_path
    out_dir = os.path.join(model_dir, "dreamfusion")
    output = []
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isdir(os.path.join(out_dir, found)):
                output.append(found)
    return output