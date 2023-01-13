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


class ParameterNorm:
    def __init__(
        self,
        mlflow_run_ids: List[str],
        ckpt_names: List[str],
        bucket_ids: List[str],
        sections: str,
        norm: str,
        upper_bound: float,
        lower_bound: float,
    ):
        self.mlflow_run_ids = mlflow_run_ids
        self.ckpt_names = ckpt_names
        self.bucket_ids = bucket_ids
        self.norm = norm
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.sections = 'inference' if sections == 'all' else sections

        now = datetime.now()
        self.plot_id = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.plot_norm_dir = Path(f'/nfs/publicNAS/tensorboard_plots/trained_model_plots')
        self.tasks = []
        self.dataframes = []

    def load_model(self, index: int):
        expt = get_exp_dir_from_mlflow_run_id(self.mlflow_run_ids[index])
        config = expt.config
        self.tasks.append(config['task'])
        ckpt_path = f'{expt.checkpoint_dir}/{self.ckpt_names[index]}'

        logger.info(f'Loading model with checkpoint : {self.ckpt_names[index]}')

        fkeys = get_frame_keys_for_bucket(self.bucket_ids[index])
        n_frames = expt.config.data.get('input_n_frames', None)

        loader_kwargs = deepcopy(config.data.loader)
        loader_kwargs['num_workers'] = 6
        loader_kwargs['batch_size'] = 6
        frames_root = None
        if config.data.get('diff_mode', None):
            n_frames = 2
        fkey_dataset_args = {
            'cache_stem': config.data.cache.stem,
            'loader_transforms': [ImageEqualSizeResizer(size=max(config.data.image_size))],
            'frame_prep_fn': get_frame_prep_fn_by_config(config),
            'n_frames': n_frames,
            'stride': expt.config.data.get('input_frame_stride', 1),
        }
        loader_params = dict(worker_init_fn=worker_init_fn, **loader_kwargs)

        dataloader = get_frames_loader(
            fkeys=fkeys, frames_root=frames_root, **fkey_dataset_args, **loader_params
        )

        model_class = resolve_od_model_class(
            expt.config.training.od_model
        )  # specific for OD but general case ?
        model = model_class.load_from_checkpoint(ckpt_path)

        '''File "/code/deployment/machine_learning/garuda/models/object_detection/centernet/model/model.py", line 109, in forward
        images, targets = batch
        ValueError: too many values to unpack (expected 2)
        q_model = post_training_quantization(model, dataloader)'''
        return model, dataloader

    def find_outliers(self, val: float) -> bool:
        if (val < self.lower_bound) or (val > self.upper_bound):
            return True
        else:
            return False

    def get_color(self):
        x: int = np.round(256 / len(self.ckpt_names), 0)
        colors_in_hex = [
            '#%02x%02x%02x'
            % tuple(cmapy.color('viridis', random.randrange(x * i, x * (i + 1)), rgb_order=True))
            for i in range(len(self.ckpt_names))
        ]
        return colors_in_hex

    def matplotlib_plot(self, indices: List[int]):
        plt.rcdefaults()
        colors = self.get_color()
        fig = plt.figure()
        ax = fig.add_subplot(211)

        x1 = self.dataframes[0].iloc[indices[0], indices[1]][indices[2]]
        x2 = self.dataframes[1].iloc[indices[0], indices[1]][indices[2]]
        y = np.arange(len(self.ckpt_names))

        ax.barh(y, [x1, x2], align='center', color=colors)
        ax.set_yticks(y)
        ax.invert_yaxis()
        ax.set_xlabel('Norm')

        patches = [
            mpatches.Patch(
                color=colors[i],
                label="{mlflow} - ckpt: {ckpt_name}\n".format(
                    ckpt_name=self.ckpt_names[i], mlflow=self.mlflow_run_ids[i]
                ),
            )
            for i in range(len(self.ckpt_names))
        ]
        ax.legend(
            handles=patches,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.3),
            ncol=1,
            fancybox=True,
            prop={'size': 8},
        )

        text_kwargs1 = dict(
            ha='left', va='center', fontsize=7, color='black', transform=plt.gcf().transFigure
        )
        text_kwargs2 = dict(
            ha='left', va='center', fontsize=8, color='black', transform=plt.gcf().transFigure
        )

        stack_keys = list(self.dataframes[0].iloc[indices[0], 3].keys())
        end_line = 0
        for j, key in enumerate(stack_keys):
            stack = f'({j}) {key}:    {self.dataframes[0].iloc[indices[0], 3][key]}'
            fig.text(0.02, 0.04 + 0.035 * j, stack, **text_kwargs1)
            end_line = 0.04 + 0.035 * j + 0.045
        fig.text(0.02, end_line, 'Stack Traces', **text_kwargs2)
        fig.text(0.02, end_line + 0.05, self.dataframes[0].iloc[indices[0], :].name, **text_kwargs1)
        fig.text(0.02, end_line + 0.085, 'Layer', **text_kwargs2)
        fig.text(0.02, end_line + 0.14, f'TASK :     {self.tasks[0]}', **text_kwargs2)
        plt.show()
        return fig

    def tensorboard_plots(self):
        summary = self.plot_norm_dir / f'plot{self.plot_id}/tensorboard_data'
        summary.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(summary.as_posix()))

        logger.info(f'Creating tensorboard plots.')

        n_layers = len(self.dataframes[0].index)
        layout = {}
        for j, plotType in zip([0, 1, 2], ['Parameter', 'Bias', 'Activation']):
            scalars = [f'{plotType}/FrobeniusNorm', f'{plotType}/1Norm']
            for i in range(n_layers):
                if isinstance(self.dataframes[0].iloc[i, j], np.ndarray) or isinstance(
                    self.dataframes[0].iloc[i, j], list
                ):
                    if self.norm == 'FrobeniusNorm':
                        if self.find_outliers(
                            self.dataframes[0].iloc[i, j][0]
                        ) or self.find_outliers(self.dataframes[1].iloc[i, j][0]):
                            fig = self.matplotlib_plot([i, j, 0])
                            writer.add_figure(f'{plotType}/FrobeniusNorm', fig)
                    writer.add_scalars(
                        f'{plotType}/FrobeniusNorm',
                        {
                            self.ckpt_names[0]: self.dataframes[0].iloc[i, j][0],
                            self.ckpt_names[1]: self.dataframes[1].iloc[i, j][0],
                        },
                        i,
                    )

                    if self.norm == '1Norm':
                        if self.find_outliers(
                            self.dataframes[0].iloc[i, j][1]
                        ) or self.find_outliers(self.dataframes[1].iloc[i, j][1]):
                            fig = self.matplotlib_plot([i, j, 1])
                            writer.add_figure(f'{plotType}/1Norm', fig)
                    writer.add_scalars(
                        f'{plotType}/1Norm',
                        {
                            self.ckpt_names[0]: self.dataframes[0].iloc[i, j][1],
                            self.ckpt_names[1]: self.dataframes[1].iloc[i, j][1],
                        },
                        i,
                    )

                    if plotType == 'Activation':
                        if len(self.dataframes[0].iloc[i, 2]) == 3:
                            scalars.append(f'{plotType}/UpperBound')
                            writer.add_scalars(
                                f'{plotType}/UpperBound',
                                {
                                    self.ckpt_names[0]: self.dataframes[0].iloc[i, 2][2],
                                    self.ckpt_names[1]: self.dataframes[1].iloc[i, 2][2],
                                },
                                i,
                            )

            layout[f'{self.sections}{j}'] = {f'{plotType}': ['Multiline', scalars]}
            writer.add_custom_scalars(layout)
        writer.close()

    def data_exists(self):
        logger.info(f'Searching for cached data.')
        cached_results = []
        for file in glob.glob(f'{self.plot_norm_dir}/**/*.yaml'):
            with open(file) as f:
                config = yaml.load(f, Loader=SafeLoader)
                for i in range(len(self.ckpt_names)):
                    if (config[i][f'model{i}']['mlflow_id'] in self.mlflow_run_ids) and (
                        config[i][f'model{i}']['ckpt_name'] in self.ckpt_names and
                        config[i][f'model{i}']['section'] == self.sections
                    ):
                        data = pd.read_csv(Path(file).parents[0] / f'model{i}.csv')
                        data.fillna(value='None', inplace=True)
                        for col in ['weight', 'bias', 'activation', 'stack_traces']:
                            data[col] = data[col].apply(literal_eval)
                        data.replace('None', None, inplace=True)
                        data.set_index(data.iloc[1:, 0].name, inplace=True)
                        cached_results.append(data)
                        self.tasks.append(config[i][f'model{i}']['task'])

        if len(cached_results) == len(self.ckpt_names):
            self.dataframes = cached_results

    def cache_dataframes(self):
        dict_file = [
            {f'model{i}': {'mlflow_id': self.mlflow_run_ids[i], 'ckpt_name': self.ckpt_names[i], 'task': self.tasks[i]}, 'section': self.sections}
            for i in range(len(self.ckpt_names))
        ]
        config_file = self.plot_norm_dir / f'plot{self.plot_id}/config_file.yaml'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as file:
            yaml.dump(dict_file, file)

        for i in range(len(self.ckpt_names)):
            filepath = self.plot_norm_dir / f'plot{self.plot_id}/model{i}.csv'
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.dataframes[i].to_csv(filepath, index=True)

    def make_plots(self):
        self.data_exists()

        if len(self.dataframes) == 0:
            for index in range(len(self.mlflow_run_ids)):
                model, dataloader = self.load_model(index)
                logger.info(f'Collecting data ...')
                norm_getter = NormData(model, dataloader, self.sections)
                norm_getter.run()
                self.dataframes.append(norm_getter.cached_sections[self.sections])

            self.cache_dataframes()

        self.tensorboard_plots()


def get_norm_cli_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_run_ids', nargs="+", required=True)
    parser.add_argument('--ckpt_names', nargs="+", required=True)
    parser.add_argument('--bucket_ids', nargs="+", required=True)
    parser.add_argument(
        '--sections', required=False, default='all', help='Choices are all or named module.'
    )
    parser.add_argument(
        '--norm', required=False, default='FrobeniusNorm', choices=['FrobeniusNorm', '1Norm']
    )
    parser.add_argument('--upper_bound', required=False, default=500, type=float)
    parser.add_argument('--lower_bound', required=False, default=0.5, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_norm_cli_args()
    ParameterNorm(**vars(args)).make_plots()
