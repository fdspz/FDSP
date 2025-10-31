import copy
import os

import numpy as np
import torch
import random
import argparse



def display_exp_setting(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 80)
    log_dict = cfg.__dict__.copy()
    for key in list(log_dict):
        if 'dir' in key or 'path' in key or 'dct' in key:
            if key == 'dct_l':
                continue
            else:
                del log_dict[key]
    logger.info(log_dict)
    logger.info('=' * 80)


def set_global_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # WARNING: if cudnn.enabled=False => greatly reduces training/inference speed.
    torch.backends.cudnn.enabled = True


def get_log_dir_index(out_dir, save_all=True):
    if save_all:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        dirs = [x[0] for x in os.listdir(out_dir)]
        if '.' in dirs:  # minor change for .ipynb
            dirs.remove('.')
        log_dir_index = str(len(dirs))
    else:
        log_dir_index = 'temp'

    return log_dir_index

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")