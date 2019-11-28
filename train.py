import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import tables
import sys

# plt.switch_backend('agg')

# import pytorch modules
import torch
from torch.autograd import Variable
import torch.utils.data

# import model and utilities
from model import LSTMRandWriter
from utils import decay_learning_rate, save_checkpoint
from plot_training import RealTimePlotter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    # prepare training data
    with tables.open_file(args.data_file) as data_file:
        train_labels = torch.from_numpy(data_file.get_node("/train_labels").read())
        train_inputs = torch.from_numpy(data_file.get_node("/train_inputs").read())
        train_masks = torch.from_numpy(data_file.get_node("/train_masks").read())
        validate_labels = torch.from_numpy(data_file.get_node("/validate_labels").read())
        validate_inputs = torch.from_numpy(data_file.get_node("/validate_inputs").read())
        validate_masks = torch.from_numpy(data_file.get_node("/validate_masks").read())
    seq_len = train_labels.size(1)
    timesteps = seq_len - 1
    print("Sequence length: {} --> {} timesteps".format(seq_len, timesteps))
    train_data = torch.utils.data.TensorDataset(train_inputs, train_labels, train_masks)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validate_data = torch.utils.data.TensorDataset(validate_inputs, validate_labels, validate_masks)
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # define model and optimizer
    model = LSTMRandWriter(input_size=train_labels.size(-1) + train_inputs.size(-1), output_size=train_labels.size(-1), cell_size=args.cell_size, num_clusters=args.num_clusters)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # initialize null hidden states and memory states
    init_states = [torch.zeros((1, args.batch_size, args.cell_size), device=device)] * 8
    init_states = [Variable(state, requires_grad=False) for state in init_states]
    h1_init, c1_init, h2_init, c2_init, h3_init, c3_init, h4_init, c4_init = init_states

    t_loss = []
    v_loss = []
    best_validation_loss = 1E10

    plotter = RealTimePlotter(["training", "validation"])

    # update training time
    start_time = time.time()

    for epoch in range(args.num_epochs):
        train_loss = 0
        model.train()
        for batch_idx, (inputs, labels, masks) in enumerate(train_loader):
            # add inputs to each datapoint in the input sequence
            param_seq = inputs.unsqueeze(1).repeat(1, seq_len, 1)
            data = torch.cat((labels, param_seq), dim=2).to(device)
            step_back = data.narrow(1, 0, timesteps)
            x = Variable(step_back, requires_grad=False)
            # Masks are used to mask out the loss where the sequence is shorter than seq_len
            masks = masks.narrow(1, 0, timesteps).to(device)
            masks = Variable(masks, requires_grad=False)

            optimizer.zero_grad()
            # feed forward
            outputs = model(x, (h1_init, c1_init), (h2_init, c2_init), (h3_init, c3_init), (h4_init, c4_init))
            end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, prev, prev2, prev3, prev4 = outputs

            # supervision
            data = data.narrow(1, 1, timesteps)
            y = Variable(data, requires_grad=False)
            loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks) / torch.sum(masks)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if np.isnan(loss.item()):
                print("Detected loss is NAN, max(rho) = {}, min(rho) = {}".format(torch.max(rho), torch.min(rho)))
                sys.exit(1)

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch + 1, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item()))

        avg_train_loss = train_loss / (len(train_loader.dataset) // args.batch_size)
        plotter.update(0, epoch, avg_train_loss)
        # update training performance
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch + 1, avg_train_loss))
        t_loss.append(avg_train_loss)

        # validation
        validation_loss = 0
        model.eval()
        for validation_inputs, validation_labels, validation_masks in validate_loader:
            validation_param_seq = validation_inputs.unsqueeze(1).repeat(1, seq_len, 1)
            validation_data = torch.cat((validation_labels, validation_param_seq), dim=2).to(device)
            step_back2 = validation_data.narrow(1, 0, timesteps)
            validation_masks = validation_masks.narrow(1, 0, timesteps).to(device)
            validation_masks = Variable(validation_masks, requires_grad=False)
            x = Variable(step_back2, requires_grad=False)

            outputs = model(x, (h1_init, c1_init), (h2_init, c2_init), (h3_init, c3_init), (h4_init, c4_init))
            end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, prev, prev2, prev3, prev4 = outputs

            validation_data = validation_data.narrow(1, 1, timesteps)
            y = Variable(validation_data, requires_grad=False)

            loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, validation_masks) / torch.sum(validation_masks)
            validation_loss += loss.item()
        avg_validation_loss = validation_loss / (len(validate_loader.dataset) // args.batch_size)
        plotter.update(1, epoch, avg_validation_loss)
        print('====> Epoch: {} Average validation loss: {:.4f}'.format(epoch + 1, avg_validation_loss))
        v_loss.append(avg_validation_loss)

        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            save_checkpoint(epoch, model, avg_validation_loss, optimizer, args.model_dir, "best.pt")

        # # learning rate annealing
        # if (epoch+1)%10 == 0:
        #     optimizer = decay_learning_rate(optimizer)

        # checkpoint model and training
        filename = 'epoch_{}.pt'.format(epoch + 1)
        save_checkpoint(epoch, model, avg_validation_loss, optimizer, args.model_dir, filename)

        print('wall time: {}s'.format(time.time() - start_time))

    plotter.save("loss_curves")


# training objective
def log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks):
    # targets
    y_0 = y.narrow(-1, 0, 1)
    y_1 = y.narrow(-1, 1, 1)
    y_2 = y.narrow(-1, 2, 1)

    # end of stroke prediction
    end_loglik = (y_0 * end + (1 - y_0) * (1 - end)).log().squeeze()

    # new stroke point prediction
    const = 1E-20  # to prevent numerical error
    pi_term = torch.Tensor([2 * np.pi]).to(device)
    pi_term = -Variable(pi_term, requires_grad=False).log()

    z = (y_1 - mu_1) ** 2 / (log_sigma_1.exp() ** 2) \
        + ((y_2 - mu_2) ** 2 / (log_sigma_2.exp() ** 2)) \
        - 2 * rho * (y_1 - mu_1) * (y_2 - mu_2) / ((log_sigma_1 + log_sigma_2).exp())
    mog_lik1 = pi_term - log_sigma_1 - log_sigma_2 - 0.5 * ((1 - rho ** 2).log())
    mog_lik2 = z / (2 * (1 - rho ** 2))
    mog_loglik = ((weights.log() + (mog_lik1 - mog_lik2)).exp().sum(dim=-1) + const).log()

    return (end_loglik * masks).sum() + ((mog_loglik) * masks).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str)
    parser.add_argument('--cell_size', type=int, default=800, help='size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=50, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--model_dir', type=str, default='save', help='directory to save model to')
    parser.add_argument('--learning_rate', type=float, default=8E-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='lr decay rate for adam optimizer per epoch')
    parser.add_argument('--num_clusters', type=int, default=20,
                        help='number of gaussian mixture clusters for stroke prediction')
    args = parser.parse_args()
    main(args)
