import matplotlib.pyplot as plt


def _plot_trajectory(axs, spiral, color):
    axs[0].plot([point[0] for point in spiral], [point[1] for point in spiral], color=color)
    axs[1].plot(range(len(spiral)), [point[0] for point in spiral], color=color)
    axs[2].plot(range(len(spiral)), [point[1] for point in spiral], color=color)


def plot_trajectory(spiral, reference_trajectory=None):
    fig, axs = plt.subplots(3)
    _plot_trajectory(axs, spiral, "red")
    if reference_trajectory is not None:
        _plot_trajectory(axs, reference_trajectory, "green")
    axs[0].axis("equal")
    plt.show()
