import torch
import torch.nn as nn
import re
import os
from datetime import datetime
import yaml
from argparse import ArgumentParser, ArgumentTypeError
from .types import Dict
import wandb


def init_weight(m, init_type='normal', init_gain=0.02):
    """
    Initializes the input module with the given parameters.

    Only deals with `Conv2d`, `ConvTranspose2d`, `Linear` and `BatchNorm2d` layers.

    Parameters
    ----------
    m : torch.nn.Module
        Module to initialize.
    init_type : str
        'normal', 'xavier', 'kaiming', or 'orthogonal'. Orthogonal initialization types for convolutions and linear
        operations. Ignored for batch normalization which uses a normal initialization.
    init_gain : float
        Gain to use for the initialization.
    """
    classname = m.__class__.__name__
    if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'BatchNorm2d':
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def get_config_and_setup_dirs(filename: str = 'config.yaml'):
    with open(filename, 'r') as fp:
        config = yaml.safe_load(fp)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config['exp_root_dir'] = config['exp_root_dir'].format(
        dataset=config['dataset'].lower(),
        timestamp=timestamp
    )

    config['log_dir'] = os.path.join(config['exp_root_dir'], 'logs')
    config['ckpt_dir'] = os.path.join(config['exp_root_dir'], 'ckpts')
    os.makedirs(config['log_dir'])
    os.makedirs(config['ckpt_dir'])

    wandb_id = wandb.util.generate_id()
    config['wandb_id'] = wandb_id

    return config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r'[^\w\-]', "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            #print(f'WARNING: Skipping key {k}!')
            continue
        if val_type == bool:
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser