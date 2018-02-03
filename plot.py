import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Plotter:
    def __init__(self):
        self.plot_dir = 'plots'
        os.makedirs(self.plot_dir, exist_ok=True)
        self.fig, self.axes = plt.subplots(2, 1, sharex=True)

    def plot_sequences(self, train_data, file):
        for ax in self.axes:
            ax.clear()
        sns.tsplot(data=list(train_data[0, :100, 0]), ax=self.axes[0])
        sns.tsplot(data=list(train_data[len(train_data)//2, :100, 0]), ax=self.axes[1])

        plt.savefig(os.path.join(self.plot_dir, file))

    def plot_losses(self, losses, file):
        for ax in self.axes:
            ax.clear()
        sns.tsplot(data=losses[True], condition='Updating', color='r', ax=self.axes[0])
        sns.tsplot(data=losses[False], condition='NotUpdating', color='b', ax=self.axes[1])
        plt.savefig(os.path.join(self.plot_dir, file))

