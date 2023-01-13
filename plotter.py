import argparse
import json
import logging
from typing import Union

import yaml
from omegaconf import OmegaConf
from yaml.loader import SafeLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import random

from datetime import datetime
from pathlib import Path

from garuda.core.logs import setup_logging

logger = logging.getLogger(__name__)


def get_legend(colors):
    custom_lines = []
    for color in colors:
        custom_lines.append(Line2D([0], [0], color=color, lw=4))

    return custom_lines


def get_plotting_cli_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--backbone_type', required=True)
    return parser.parse_args()


def get_latest_profiler_results(
    profiler_dir: Path,
    model_type: str,
    backbone_type: str,
    model_id: str,
    model_id_val: float = None,
):
    all_dir = []
    prefix_name = f'{model_type}_{backbone_type}_'
    for d in profiler_dir.glob(f'{prefix_name}*'):
        all_dir.append(d.stem)
    all_dir.sort(
        key=lambda date: datetime.strptime(date.replace(prefix_name, ''), "%d-%m-%Y_%H-%M-%S")
    )
    latest_dir = profiler_dir / all_dir.pop() / 'profiling_results'
    if model_id_val:
        if model_id == 'width-mult':
            number2str = f'{int(model_id_val)}_{int(model_id_val * 10) - int(model_id_val) * 10}'
        else:
            number2str = model_id_val
        print(number2str)
        return list(latest_dir.glob(f'{model_id}_{number2str}*'))[0]
    else:
        return list(latest_dir.glob(f'{model_id}*'))


def get_data(model_type: str, backbone_type: str, model_id: str, model_id_val: float = None):
    profiler_result_path = Path('/nfs/publicNAS/tensorboard_profiler') / model_type / backbone_type
    file_names = get_latest_profiler_results(
        profiler_result_path, model_type, backbone_type, model_id, model_id_val
    )
    file_name = file_names
    if not model_id_val:
        file_name = file_names[0]

    arch_file_name = file_name.stem.split('.')[0]
    return arch_file_name, file_names


class ProfilingData:
    def __init__(self, file, arch_file_name):
        self.file_name = file
        self.arch_file_name = arch_file_name
        self.model_parameters = None
        self.arch = None
        self.blocks = None
        self.all_blocks = None
        self.plot_name = None

        self.operation_table = None

    def get_table(self):
        with open(self.file_name) as f:
            data = json.load(f)

        pd.options.display.max_columns = 10
        df = pd.json_normalize(data, record_path=['traceEvents'])
        operations = ['conv2d', 'add', 'relu', 'upsample_nearest2d']

        df = df[['name', 'ph', 'args.Input Dims', 'dur', 'ts']]
        df = df.loc[df['ph'] == 'X']
        df['name'] = df['name'].str.replace('aten::', '')
        df['name'] = df['name'].str.replace('upsample_nearest2d', 'upsample')
        df['dur'] = pd.to_numeric(df['dur'])
        df = df[['name', 'args.Input Dims', 'dur', 'ts']]
        df = df[df['name'].isin(operations)]

        df['input_dim'] = df['args.Input Dims'].apply(lambda x: x[0])
        df['ms_ratio'] = df['dur'] / df['input_dim'].apply(lambda x: np.prod(x))
        df['weight'] = df['args.Input Dims'].apply(lambda x: x[1:3] if len(x[1:3]) > 1 else 0)
        operation_table = df[['name', 'dur', 'input_dim', 'ms_ratio', 'weight', 'ts']]

        operation_table['block'] = ""
        operation_table.sort_values(by='ts', ascending=True, inplace=True)

        self.operation_table = operation_table
        self.get_blocks()

    def get_blocks(self):
        with open(f'{self.file_name.parents[1]}/config/{self.arch_file_name}_arch.yaml') as f:
            self.arch = yaml.load(f, Loader=SafeLoader)

        k = -1
        for component in self.arch['block'].keys():
            k = k * 1
            for block in self.arch['block'][component]:
                k = k * 1
                for _ in self.arch['construction'][block]:
                    k += 1
                    self.operation_table.iloc[k, -1] = component
                    if (self.operation_table.iloc[k, 0] == 'add') or (
                        self.operation_table.iloc[k, 0] == 'upsample'
                    ):
                        while (self.operation_table.iloc[k, 0] == 'add') or (
                            self.operation_table.iloc[k, 0] == 'upsample'
                        ):
                            k += 1
                            self.operation_table.iloc[k, -1] = component

        self.plot_name = self.file_name.stem.split('.')[0]
        model_config = open(f'{self.file_name.parents[1]}/config/{self.plot_name}_config.yaml')
        self.model_parameters = yaml.load(model_config, Loader=yaml.FullLoader)

        self.all_blocks = self.operation_table['block'].unique()
        self.blocks = {self.all_blocks[i]: i for i in range(len(self.all_blocks))}
        self.arch = self.operation_table['block'].tolist()


class BasePlot:
    arch_colors = {0: '#ffcc66', 1: '#66ffcc', 2: '#6666ff'}

    def __init__(self, n_sgridspec: list, n_axes: int):

        self.fig = plt.figure(figsize=(18, 12), facecolor='#ffffff')
        gs0 = gridspec.GridSpec(1, 1, figure=self.fig, hspace=0.0)
        self.area = gs0[0].subgridspec(n_sgridspec[0], n_sgridspec[1])

        self.n_axes = n_axes

        for i in range(1, n_axes + 1):
            setattr(self, f'ax{i}', None)

    def init_axes(self, value):
        for i in range(1, self.n_axes + 1):
            setattr(self, f'ax{i}', value[i - 1])

    def _set_title(self):
        pass

    def _set_axes(self):
        pass

    def _get_xy_data(self):
        pass

    def _feed_plot(self):
        pass

    def create_plot(self):
        self._get_xy_data()
        self._set_title()
        self._set_axes()
        self._feed_plot()


class HyperparameterPlots(BasePlot):
    x_width = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    x_aggregation = [48, 56, 64, 72, 80, 88, 96, 104]
    x_hidden = [144, 168, 192, 216, 240, 264, 288, 312]

    paras = {0: 'width-mult', 1: 'agg-out-chs', 2: 'hidden-dims'}
    X = {0: x_width, 1: x_aggregation, 2: x_hidden}
    lw = {0: 0.2 / 2, 1: 8 / 2, 2: 24 / 2}

    def __init__(self, plt_data, block_names):
        BasePlot.__init__(self, [1, 3], 3)

        Axes = [
            self.fig.add_subplot(self.area[:, 0]),
            self.fig.add_subplot(self.area[:, 1]),
            self.fig.add_subplot(self.area[:, 2]),
        ]

        self.plt_data = plt_data
        self.block_names = block_names
        self.init_axes(Axes)

    def _set_axes(self):
        self.__dict__['ax1'].set_ylabel('in ms', fontsize=14)
        for i, ax in enumerate([self.__dict__[f'ax{i+1}'] for i in range(3)]):
            ax.set_xlim(np.min(self.X[i]) * (1 - 0.1), np.max(self.X[i]) * (1 + 0.05))
            ax.set_xticks(self.X[i])

            height = 0
            for b in self.plt_data[self.paras[i]].keys():
                if b != 'other_para':
                    height += np.max(self.plt_data[self.paras[i]][b])
            ax.set_ylim(0, height)

    def _feed_plot(self):
        legend_lines = get_legend([self.arch_colors[i] for i in range(3)])
        plt.figlegend(
            legend_lines,
            [b for b in self.block_names.keys()],
            loc='upper center',
            ncol=3,
            labelspacing=0.1,
            fontsize=15,
        )

        block_names = [b for b in self.block_names.keys()]
        arch_colors = {block_names[i]: self.arch_colors[i] for i in range(len(block_names))}
        axes = {i: self.__dict__[f'ax{i + 1}'] for i in range(3)}

        for j in range(3):
            print(self.__dict__[f'ax{j+1}'])
            dur = np.zeros(len(self.X[j]))
            for idx, block in enumerate(self.plt_data[self.paras[j]].keys()):
                if block != 'other_para':
                    axes[j].bar(
                        self.X[j],
                        self.plt_data[self.paras[j]][block],
                        color=arch_colors[block],
                        bottom=dur,
                        width=self.lw[j],
                    )
                    axes[j].set_title(self.paras[j])
                    text = ''
                    for para in self.plt_data[self.paras[j]]['other_para'].keys():
                        value = self.plt_data[self.paras[j]]['other_para'][para]
                        text += f'{para} = {value}  '
                    axes[j].text(
                        0.5,
                        -0.05,
                        text,
                        fontsize=14,
                        ha='center',
                        va='center',
                        transform=axes[j].transAxes,
                    )
                    dur += np.array(self.plt_data[self.paras[j]][block])
        plt.show()


class GeneralPlots(BasePlot):
    def __init__(self, plt_data):
        BasePlot.__init__(self, [7, 1], 3)

        Axes = [
            self.fig.add_subplot(self.area[:3, :]),
            self.fig.add_subplot(self.area[3, :], sharex=self.__dict__['ax1']),
            self.fig.add_subplot(self.area[4:, :], sharex=self.__dict__['ax2']),
        ]

        self.plt_data = plt_data
        self.init_axes(Axes)

        self.percentage = None
        self.dur_input_ratio = None
        self.X = None

    def _set_title(self):
        width = self.plt_data.model_parameters['backbone']['width_mult']
        aggr_chs = self.plt_data.model_parameters['aggregation']['out_channels']
        hidden_dim = self.plt_data.model_parameters['head']['hidden_dim']

        fig_title = self.fig.suptitle(
            f'{self.plt_data.file_name.parents[2].stem} - {self.plt_data.file_name.parents[3].stem}'
            f'\n width mult. = {width}'
            f' || aggr. out channels = {aggr_chs}'
            f' || hidden dims = {hidden_dim}',
            fontsize="x-large",
        )

        fig_title.set_y(0.95)
        self.fig.subplots_adjust(top=0.85)

    def _set_axes(self):
        self.__dict__['ax1'].set_ylim(0, np.max(self.percentage) * (1 + 0.1))
        self.__dict__['ax1'].set_xlim(0, len(self.percentage) + 5)

        self.__dict__['ax3'].set_ylim(0, np.max(self.dur_input_ratio) * (1 + 0.1))
        self.__dict__['ax3'].set_xlim(0, len(self.dur_input_ratio) + 5)

        self.__dict__['ax2'].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            direction="in",
            pad=-55,
            labeltop=True,
            labelbottom=False,
        )

        self.__dict__['ax2'].tick_params(
            axis='y', which='both', left=False, right=False, labelleft=False
        )

        self.__dict__['ax2'].set_ylim(0, 1)
        self.__dict__['ax2'].xaxis.grid(True)
        # self.ax2.set_xlim(0, len(self.percentage) + 5)
        self.__dict__['ax2'].set_xticks(len(self.percentage))
        self.__dict__['ax2'].set_xticklabels(
            self.plt_data.operation_table['name'].tolist(), rotation=90
        )

        for i in [self.__dict__['ax1'], self.__dict__['ax3']]:
            i.spines['right'].set_color('black')
            i.spines['bottom'].set_color('black')
            i.spines['left'].set_color('black')

            i.grid()

            i.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            i.xaxis.set_ticks(self.X)
            if i == self.__dict__['ax1']:
                i.set_ylabel('in Percentage %')

            if i == self.__dict__['ax3']:
                i.set_ylabel('ms per pixel')

    def _get_xy_data(self):
        dur = self.plt_data.operation_table['dur'].to_numpy()
        total_time = np.sum(dur)
        self.percentage = np.round(dur / total_time, 3) * 100
        self.dur_input_ratio = self.plt_data.operation_table['ms_ratio'].to_numpy()
        self.X = np.arange(2, 5 * len(dur + 1), 5)

    def _feed_plot(self):
        color_ax1 = [
            self.arch_colors[self.plt_data.blocks[a_block]] for a_block in self.plt_data.arch
        ]

        legend_lines = get_legend([self.arch_colors[i] for i in range(3)])

        barlist1 = self.__dict__['ax1'].bar(self.X, self.percentage, 4)
        barlist2 = self.__dict__['ax3'].bar(self.X, self.dur_input_ratio, 4)
        barlist3 = self.__dict__['ax2'].bar(
            self.X, np.ones(len(self.X)), 4, edgecolor='black', color='none'
        )

        for i in range(len(color_ax1)):
            barlist1[i].set_color(color_ax1[i])
            barlist2[i].set_color(color_ax1[i])
            barlist3[i].set_color(color_ax1[i])

        self.__dict__['ax1'].legend(legend_lines, [b for b in self.plt_data.all_blocks])
        plt.savefig(f'{self.plt_data.file_name.parent.parent}/plots/{self.plt_data.plot_name}.png')
        plt.show()


class PlottingProfiler:
    def __init__(
        self, model_type: str, backbone_type: str, model_id: str = None, model_id_val: float = None
    ):

        self.model_type = model_type
        self.backbone_type = backbone_type
        self.model_id = model_id
        self.model_id_val = model_id_val

        self.comparision_data = {}
        self.block_names = None

    def parameter_plot(self):
        model_ids = ['width-mult', 'hidden-dims', 'agg-out-chs']

        for model_id in model_ids:
            arch_file_name, file_names = get_data(self.model_type, self.backbone_type, model_id)
            plot_dir = file_names[0].parent.parent / 'plots'
            plot_dir.mkdir(exist_ok=True, parents=True)

            for file in file_names:

                plt_data = ProfilingData(file, arch_file_name)
                plt_data.get_table()

                self.block_names = plt_data.blocks
                plot_type = plt_data.plot_name.split('_')[0]
                para_values = None

                if plot_type == 'width-mult':
                    para_values = {
                        'aggr-out-chs': plt_data.model_parameters['aggregation']['out_channels'],
                        'hidden-dims': plt_data.model_parameters['head']['hidden_dim'],
                    }
                if plot_type == 'agg-out-chs':
                    para_values = {
                        'width_mult': plt_data.model_parameters['backbone']['width_mult'],
                        'hidden-dims': plt_data.model_parameters['head']['hidden_dim'],
                    }
                if plot_type == 'hidden-dims':
                    para_values = {
                        'aggr-out-chs': plt_data.model_parameters['aggregation']['out_channels'],
                        'width_mult': plt_data.model_parameters['backbone']['width_mult'],
                    }

                if not plot_type in self.comparision_data:
                    self.comparision_data[plot_type] = {'other_para': para_values}

                for block in plt_data.all_blocks:

                    df = plt_data.operation_table[plt_data.operation_table['block'] == block]
                    time_per_model = np.round(np.sum(df['dur'].to_numpy()), 2)

                    if not block in self.comparision_data[plot_type]:
                        self.comparision_data[plot_type][block] = []

                    self.comparision_data[plot_type][block].append(time_per_model)

        self.plot_hyperparameter()

    def general_plot(self):
        arch_file_name, file_name = get_data(
            self.model_type, self.backbone_type, self.model_id, self.model_id_val
        )
        plot_dir = file_name.parent.parent / 'plots'
        plot_dir.mkdir(exist_ok=True, parents=True)

        plt_data = ProfilingData(file_name, arch_file_name)
        plt_data.get_table()

        self.operations_vs_time(plt_data)

    def plot_hyperparameter(self):
        plot = HyperparameterPlots(self.comparision_data, self.block_names)
        plot.create_plot()

    def operations_vs_time(self, plt_data):
        plot = GeneralPlots(plt_data)
        plot.create_plot()


if __name__ == '__main__':
    setup_logging('INFO')
    dict_args = vars(get_plotting_cli_args())
    # model_id = width-mult, hidden-dims, agg-out-chs
    # dict_args['model_id'] = 'agg-out-chs'
    # dict_args['model_id_val'] = 48

    plot = PlottingProfiler(**dict_args)
    # plot.general_plot()
    plot.parameter_plot()
