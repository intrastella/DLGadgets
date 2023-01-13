class GeneralPlots:

    def __init__(self, plt_data):
        self.plt_data = plt_data
        self.fig = plt.figure(figsize=(18, 12), facecolor='#ffffff')
        gs0 = gridspec.GridSpec(1, 1, figure=self.fig, hspace=0.0)
        self.area = gs0[0].subgridspec(7, 1)

        self.ax1 = None
        self.ax2 = None
        self.name_bar = None

        self.percentage = None
        self.dur_input_ratio = None
        self.X = None

    def set_title(self):
        width = self.plt_data.model_parameters['backbone']['width_mult']
        aggr_chs = self.plt_data.model_parameters['aggregation']['out_channels']
        hidden_dim = self.plt_data.model_parameters['head']['hidden_dim']

        fig_title = self.fig.suptitle(f'{self.plt_data.file_name.parents[2].stem} - {self.plt_data.file_name.parents[3].stem}'
                                      f'\n width mult. = {width}'
                                      f' || aggr. out channels = {aggr_chs}'
                                      f' || hidden dims = {hidden_dim}', fontsize="x-large")

        fig_title.set_y(0.95)
        self.fig.subplots_adjust(top=0.85)

    def set_axes(self):
        self.ax1 = self.fig.add_subplot(self.area[:3, :])
        self.name_bar = self.fig.add_subplot(self.area[3, :], sharex=self.ax1)
        self.ax2 = self.fig.add_subplot(self.area[4:, :], sharex=self.name_bar)
        self.ax1.set_ylim(0, np.max(self.percentage) * (1 + .1))
        self.ax2.set_ylim(0, np.max(self.dur_input_ratio) * (1 + .1))
        self.ax1.set_xlim(0, len(self.percentage) + 5)
        self.ax2.set_xlim(0, len(self.dur_input_ratio) + 5)

        self.name_bar.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            direction="in",
            pad=-55,
            labeltop=True,
            labelbottom=False)

        self.name_bar.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)

        self.name_bar.xaxis.grid(True)
        self.name_bar.set_ylim(0, 1)
        self.name_bar.set_xticklabels(self.plt_data.operation_table['name'].tolist(), rotation=90)

        for i in [self.ax1, self.ax2]:
            i.spines['right'].set_color('black')
            i.spines['bottom'].set_color('black')
            i.spines['left'].set_color('black')

            i.grid()

            i.tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)

            i.xaxis.set_ticks(self.X)
            if i == self.ax1:
                i.set_ylabel('in Percentage %')

            if i == self.ax2:
                i.set_ylabel('ms per pixel')

    def get_xy_data(self):
        dur = self.plt_data.operation_table['dur'].to_numpy()
        total_time = np.sum(dur)
        self.percentage = np.round(dur / total_time, 3) * 100
        self.dur_input_ratio = self.plt_data.operation_table['ms_ratio'].to_numpy()
        self.X = np.arange(2, 5 * len(dur + 1), 5)

    def feed_plot(self):
        arch_colors = {0: '#ffcc66', 1: '#66ffcc', 2: '#6666ff'}
        color_ax1 = [arch_colors[self.plt_data.blocks[a_block]] for a_block in self.plt_data.arch]

        legend_lines = get_legend([arch_colors[i] for i in range(3)])

        barlist1 = self.ax1.bar(self.X, self.percentage, 4)
        barlist2 = self.ax2.bar(self.X, self.dur_input_ratio, 4)
        barlist3 = self.name_bar.bar(self.X, np.ones(len(self.X)), 4, edgecolor='black', color='none')

        for i in range(len(color_ax1)):
            barlist1[i].set_color(color_ax1[i])
            barlist2[i].set_color(color_ax1[i])
            barlist3[i].set_color(color_ax1[i])

        self.ax1.legend(legend_lines, [b for b in self.plt_data.all_blocks])
        plt.savefig(f'{self.plt_data.file_name.parent.parent}/plots/{self.plt_data.plot_name}.png')
        plt.show()