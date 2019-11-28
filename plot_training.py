import numpy as np
import matplotlib.pyplot as plt


class RealTimePlotter(object):
    def __init__(self, series_labels):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.xs = [[] for _ in range(len(series_labels))]
        self.ys = [[] for _ in range(len(series_labels))]
        self.series_labels = series_labels

    def update(self, series_idx, x, y):
        self.xs[series_idx].append(x)
        self.ys[series_idx].append(y)
        self.redraw()

    def redraw(self):
        self.ax1.clear()
        for series_idx in range(len(self.series_labels)):
            self.ax1.plot(self.xs[series_idx], self.ys[series_idx])
        self.ax1.legend(self.series_labels)
        plt.pause(0.05)

    def save(self, filename):
        self.fig.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    plotter = RealTimePlotter(["a", "b"])
    for i in range(10):
        y = np.random.random()
        plotter.update(0, i, y)
        plotter.update(1, i, 2 * y)
    plt.show()
