import torch
import os
import numpy
from matplotlib import pyplot


def decay_learning_rate(optimizer, decay_rate):
    # learning rate annealing
    state_dict = optimizer.state_dict()
    lr = state_dict['param_groups'][0]['lr']
    lr *= decay_rate
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return optimizer


def save_checkpoint(epoch, model, validation_loss, optimizer, directory, filename='best.pt'):
    checkpoint = ({'epoch': epoch + 1,
                   'model': model.state_dict(),
                   'validation_loss': validation_loss,
                   'optimizer': optimizer.state_dict()
                   })
    try:
        torch.save(checkpoint, os.path.join(directory, filename))
    except:
        os.mkdir(directory)
        torch.save(checkpoint, os.path.join(directory, filename))


def _plot_trajectory(axes, trajectory, color="red"):
    delta_x = trajectory[:, 1]
    delta_y = trajectory[:, 2]
    x = numpy.cumsum(delta_x)
    y = numpy.cumsum(delta_y)

    cuts = numpy.where(trajectory[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        axes[0, 0].plot(delta_x[start:cut_value], delta_y[start:cut_value], 'k-', linewidth=3, color=color)
        axes[1, 0].plot(range(start, cut_value), delta_x[start:cut_value], 'k-', linewidth=3, color=color)
        axes[2, 0].plot(range(start, cut_value), delta_y[start:cut_value], 'k-', linewidth=3, color=color)
        axes[0, 1].plot(x[start:cut_value], y[start:cut_value], 'k-', linewidth=3, color=color)
        axes[1, 1].plot(range(start, cut_value), x[start:cut_value], 'k-', linewidth=3, color=color)
        axes[2, 1].plot(range(start, cut_value), y[start:cut_value], 'k-', linewidth=3, color=color)
        start = cut_value + 1


def plot_trajectory(trajectory, reference_trajectory=None, save_name=None):
    f, axes = pyplot.subplots(3, 2)
    _plot_trajectory(axes, trajectory)
    if reference_trajectory is not None:
        _plot_trajectory(axes, reference_trajectory, color="green")

    axes[0, 0].axis('equal')
    axes[0, 1].axis('equal')

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()
