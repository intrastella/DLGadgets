

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
