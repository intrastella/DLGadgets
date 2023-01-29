import argparse
import glob
import inspect
import logging
import random
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import cmapy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yaml import SafeLoader


cwd = Path().absolute()
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class NormData:
    model_info = {'weight': [], 'bias': [], 'activation': [], 'stack_traces': []}

    ACTIVATION_lAYERS = [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.Linear,
        nn.ReLU,
        nn.ReLU6,
        nn.Softmax,
        nn.Tanh,
        nn.Sigmoid,
        nn.LeakyReLU,
        nn.Hardswish,
    ]

    def __init__(self, model: nn.Module, dataloader: DataLoader, section: str):
        self.model = model
        self.dataloader = dataloader
        self.section = section

        self.module_name_stacked = ''
        self.stack_traces = {}
        self.cached_sections = {}
        self.current_section = None
        self.to_cache = False
        self.model_data = pd.DataFrame(self.model_info)

    def cache_preparation(self):
        if self.section == 'inference':
            if self.module_name_stacked == self.section:
                self.to_cache = True
                self.current_section = self.module_name_stacked

        else:
            if len(self.module_name_stacked.split('.')) == 2:
                if not self.current_section:
                    self.current_section = self.module_name_stacked.split('.')[1]

                else:
                    if self.current_section != self.module_name_stacked.split('.')[1]:
                        self.current_section = self.module_name_stacked.split('.')[1]

            if self.section == self.current_section:
                self.cached_sections[self.section] = {}
                self.to_cache = True
            else:
                self.to_cache = False

    def caching(self, module_name: str):
        if (module_name == self.current_section) and self.to_cache:
            self.cached_sections[self.current_section] = self.model_data

    def update_stacked_names(self, module_name: str, pytorch_name: str, module_stack: str):
        self.stack_traces[pytorch_name] = module_stack
        if len(list(self.module_name_stacked)) == 0:
            self.module_name_stacked = module_name
        else:
            self.module_name_stacked += f'.{module_name}'

    def remove_last_stack(self):
        self.stack_traces.popitem()
        list_names = self.module_name_stacked.split('.')
        list_names.pop()
        self.module_name_stacked = '.'.join(list_names)

    def get_norms(self, tensor: torch.Tensor) -> list:
        np_type = tensor.cpu().detach().numpy()
        flatten_value = np.linalg.norm(np_type)  # frobenius norm
        norm1 = np.sum(np.abs(np_type))  # 1 norm
        return [flatten_value, norm1]

    def get_activation(
        self,
        in_tensor: Union[torch.Tensor, tuple],
        out_tensor: Union[torch.Tensor, tuple],
        module_name: str,
        key: str,
    ) -> list:
        imgs = []

        for tensor in [in_tensor, out_tensor]:
            if torch.is_tensor(tensor):
                imgs.append(self.get_norms(tensor))

            elif isinstance(tensor, tuple):
                imgs.append(self.get_norms(torch.stack(list(tensor), dim=0)))

        # ||f * g|| =< ||f|| ||g|| for any p norm, choose f=weight, g=image (Young's convolution inequality)
        if isinstance(self.cached_sections[key].at[module_name, 'weight'], list):
            upper_bound: float = imgs[0][1] * self.cached_sections[key].at[module_name, 'weight'][0]
            img_1norm = deepcopy(imgs[1])
            img_1norm.append(upper_bound)
            return img_1norm

        else:
            return imgs[1]

    def save_activation(self, module_name: str, key: str):
        def hook_fn(module, in_tensors, out_tensors):
            module._forward_hooks: Dict[int, Callable] = OrderedDict()
            self.cached_sections[key].at[module_name, 'activation'] = self.get_activation(
                in_tensors, out_tensors, module_name, key
            )

        return hook_fn

    def access_layers(self, net: torch.nn.Module = None):
        if not net:
            net = self.model

        for module_name, module in net.named_children():
            if module.__class__.__name__ not in ['QuantStub', 'DeQuantStub']:
                self.update_stacked_names(
                    module_name, module.__class__.__name__, inspect.getfile(module.__class__)
                )
                self.cache_preparation()

                if isinstance(module, tuple(self.ACTIVATION_lAYERS)):
                    if self.to_cache:

                        bias = None
                        weight = None
                        for name, params in module.named_parameters():
                            if name == 'bias':
                                bias = self.get_norms(params)
                            if name == 'weight':
                                weight = self.get_norms(params)
                        trace = deepcopy(self.stack_traces)
                        self.model_data.loc[self.module_name_stacked] = [weight, bias, None, trace]
                        module.register_forward_hook(
                            self.save_activation(self.module_name_stacked, self.current_section)
                        )

                else:
                    self.access_layers(module)
                    self.caching(module_name)

                self.remove_last_stack()

    def run(self):
        self.access_layers()
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                images = batch
                self.model.inference(images)

