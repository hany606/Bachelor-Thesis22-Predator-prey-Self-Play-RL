from bach_utils.logger import get_logger
clilog = get_logger()

import os
import torch
import random
import numpy as np

# Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
def check_cuda():
    if torch.cuda.is_available():
        clilog.info("## CUDA available")
        clilog.info(f"Current device: {torch.cuda.current_device()}")
        clilog.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        return 1
    else:
        clilog.warn("## CUDA not available")
        return 0

# Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
# When it is called it set the random seed
def make_deterministic(seed_value, cuda_check=False):
    seed = seed_value
    clilog.info(f"Make deterministic seed: {seed_value}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if(cuda_check):
        cuda_flag = check_cuda()
        if(cuda_flag):
            # see https://github.com/pytorch/pytorch/issues/47672
            cuda_version = torch.version.cuda
            if cuda_version is not None and float(torch.version.cuda) >= 10.2:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
            else:
                torch.set_deterministic(True)  # Not all Operations support this.
            # This is only for Convolution no problem
            torch.backends.cudnn.deterministic = True