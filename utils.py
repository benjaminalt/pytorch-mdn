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


def _plot_trajectory(ax, trajectory, color="red"):
    x = numpy.cumsum(trajectory[:, 1]) # Deltas to absolute coordinates
    y = numpy.cumsum(trajectory[:, 2])

    cuts = numpy.where(trajectory[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3, color=color)
        start = cut_value + 1


def plot_trajectory(trajectory, reference_trajectory=None, save_name=None):
    f, ax = pyplot.subplots()
    _plot_trajectory(ax, trajectory)
    if reference_trajectory is not None:
        _plot_trajectory(ax, reference_trajectory, color="green")

    ax.axis('equal')

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
