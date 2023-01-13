

class GradientHistogramLogger(Callback):
    def __init__(self, cache_size: int = 1024):
        self.activations = defaultdict(list)
        self.tb_logger = None
        self.label_to_name = None
        self.layer_hook_holder = LayerForwardHook()
        self.hook_caches = {}
        self.cache_size = cache_size
        self.output_dirs = {}

    @staticmethod
    def plot_layer_output(
            hook_key: str,
            layer: nn.Module,
            grad_input: Union[torch.Tensor, tuple],
            grad_output: Union[torch.Tensor, tuple],
            **kwargs,
    ):
        hook_cache = kwargs['hook_cache']
        stage = kwargs['stage']

        if isinstance(grad_output, tuple):
            hook_cache[stage][f'{hook_key}_activation'].append(grad_output[0])
            hook_cache[stage][f'{hook_key}_activation'].append(grad_output[1])

        else:
            hook_cache[stage][f'{hook_key}_activation'].append(grad_output)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._setup(trainer, pl_module, "train")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._teardown(trainer, pl_module, 'train')

    def on_validation_epoch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._setup(trainer, pl_module, 'valid')

    def on_validation_epoch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._teardown(trainer, pl_module, 'valid')

    def _setup(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None
    ) -> None:

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

        # register forward hooks for pl_module
        self.hook_caches[stage] = {}

        for module_name, module in pl_module.named_modules():
            pytorch_name = module.__class__.__name__
            if isinstance(module, tuple(ACTIVATION_lAYERS)):
                self.layer_hook_holder.register_layer_forward_hook(
                    f'{module_name}_{pytorch_name}',
                    module,
                    GradientHistogramLogger.plot_layer_output,
                    trainer=trainer,
                    stage=stage,
                    hook_cache=self.hook_caches,
                )

                self.hook_caches[stage][f'{module_name}_{pytorch_name}_activation'] = deque(
                    maxlen=self.cache_size
                )

    def _teardown(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None
    ) -> None:
        # aggregating statistics
        tb_logger = extract_logger(trainer.logger, TensorBoardLogger)
        for hook_key in self.hook_caches[stage].keys():
            tensor_list = self.hook_caches[stage][hook_key]
            tensors = torch.cat(list(tensor_list), dim=0)

            tb_logger.experiment.add_histogram(f'{stage}_{hook_key}', tensors, trainer.global_step)

            self.hook_caches[stage][hook_key].clear()
        self.hook_caches[stage] = {}
        self.layer_hook_holder.remove_all_forward_hooks()


class WeightPlot:

    def __init__(self, weight_data, model_data):
        self.weight_data = weight_data
        self.model_data = model_data

        self.figs = [plt.figure() for _ in range(weight_data.shape[0])]
        self.n_axes = weight_data.shape[0]
        self.layer_num = 0
        self.colors = self.get_color()

        for i in range(1, weight_data.shape[0] + 1):
            setattr(self, f'ax{i}', None)

    def get_color(self):
        colors_in_hex = ['#%02x%02x%02x' % tuple(cmapy.color('viridis', random.randrange(0, 256), rgb_order=True)) for _ in range(self.weight_data.shape[0])]
        return colors_in_hex

    def init_axes(self, value):
        setattr(self, f'ax{self.layer_num}', value)

    def make_plots(self):
        for i in range(self.weight_data.shape[0]):
            weight_data_row = self.weight_data.iloc[i, :]
            self.layer_num = i

            if isinstance(weight_data_row.iloc[0], np.ndarray):
                self.make_4d_plot(weight_data_row)

            else:
                if weight_data_row.iloc[0].size == 1:
                    self.make_2d_plot(weight_data_row)

                else:
                    pass

    def make_4d_plot(self, weight_data_row):
        self.init_axes(self.figs[self.layer_num].add_subplot(111, projection='3d'))
        patches = []
        a = np.min([np.min(weight_data_row.iloc[jdx]) for jdx in range(weight_data_row.size)])
        b = np.max([np.max(weight_data_row.iloc[jdx]) for jdx in range(weight_data_row.size)])
        x = np.linspace(a, b, 10)
        B, C = weight_data_row.iloc[0].shape
        plot_text = ''

        for idx in range(weight_data_row.size):
            verts = []
            for j in range(C):
                Y = np.zeros(10)
                for i in range(9):
                    Y[i] = abs(np.count_nonzero(weight_data_row.iloc[idx][..., j] < x[i]) - np.count_nonzero(weight_data_row.iloc[idx][..., j] < x[i + 1])) / B

                # plotting
                self.__dict__[f'ax{self.layer_num}'].plot(x, np.ones(10) * j, Y, label=f'{weight_data_row.index[idx]}', color=self.colors[idx], alpha=0.5)
                verts.append(polygon_under_graph(x, Y))

            poly = PolyCollection(verts, facecolors=[self.colors[idx]] * C, alpha=.3)
            self.__dict__[f'ax{self.layer_num}'].add_collection3d(poly, zs=np.arange(0, C, 1), zdir='y')
            patches.append(mpatches.Patch(color=self.colors[idx], label=weight_data_row.index[idx]))
            plot_text += "{ckpt_name}, mlflow: {mlflow}\n".format(ckpt_name=self.model_data.iloc[idx, 0],
                                                                    mlflow=self.model_data.iloc[idx, 2])

        self.figs[self.layer_num].legend(handles=patches, loc='upper left', prop={'size': 8})

        text_kwargs = dict(ha='left', va='top', fontsize=8, color='black', transform=plt.gcf().transFigure)
        self.figs[self.layer_num].text(0.01, 0.1, plot_text, **text_kwargs)

        self.__dict__[f'ax{self.layer_num}'].set_title(self.model_data.iloc[0, 1], y=1)
        self.__dict__[f'ax{self.layer_num}'].set_yticks(np.arange(C))

        plt.setp(self.__dict__[f'ax{self.layer_num}'].get_xticklabels(), fontsize=8)
        plt.setp(self.__dict__[f'ax{self.layer_num}'].get_yticklabels(), fontsize=8)
        plt.setp(self.__dict__[f'ax{self.layer_num}'].get_zticklabels(), fontsize=8)
        plt.show()

    def make_2d_plot(self, weight_data_row):
        self.init_axes(self.figs[self.layer_num].add_subplot(111))
        self.__dict__[f'ax{self.layer_num}'].bar(np.arange(0, weight_data_row.size, 1), weight_data_row, color=self.colors)
        patches = [mpatches.Patch(color=self.colors[i], label=weight_data_row.index[i]) for i in range(weight_data_row.size)]
        self.figs[self.layer_num].legend(handles=patches, loc='upper left', prop={'size': 8})
        plt.show()




def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]

