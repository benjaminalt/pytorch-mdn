import joblib
import numpy as np
import torch
from torch.autograd import Variable
from utils import plot_trajectory
from model import LSTMRandWriter
import os
import argparse

import generate_data_spirals

# find gpu 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def generate_unconditionally(input_scaler, label_scaler, state_dict_file, cell_size=400, num_clusters=20,
                             timesteps=800, random_seed=700):
    inputs, spiral = next(generate_data_spirals.generate_spirals(timesteps))
    scaled_inputs = torch.from_numpy(input_scaler.transform([inputs])).float().unsqueeze(0)
    input_dim = len(inputs) + 3
    output_dim = 3

    model = LSTMRandWriter(input_dim, output_dim, cell_size, num_clusters)
    # load trained model weights
    model.load_state_dict(torch.load(state_dict_file)['model'])
    model.to(device)

    np.random.seed(random_seed)
    initial_input = torch.cat((torch.zeros((1, 1, 3)), scaled_inputs), -1).to(device)
    # initialize null hidden states and memory states
    init_states = [torch.zeros((1,1, cell_size), device=device)]*4
    x = Variable(initial_input)
    init_states = [Variable(state, requires_grad = False) for state in init_states]
    h1_init, c1_init, h2_init, c2_init = init_states
    prev = (h1_init, c1_init)
    prev2 = (h2_init, c2_init)
    
    record = [np.array([0] * output_dim)]

    for i in range(timesteps):
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, p, prev, prev2 = model(x, prev, prev2)

        # sample end stroke indicator
        prob_end = end[0,0].item()
        sample_end = np.random.binomial(1,prob_end)

        sample_index = np.random.choice(range(num_clusters), p=weights[0,0].detach().cpu().numpy())
        
        # sample new stroke point
        mu = np.array([mu_1[0, 0, sample_index].item(), mu_2[0, 0, sample_index].item()])
        v1 = log_sigma_1.exp()[0, 0, sample_index].item()**2
        v2 = log_sigma_2.exp()[0, 0, sample_index].item()**2
        c = p[0, 0, sample_index].item()*log_sigma_1.exp()[0, 0, sample_index].item() * log_sigma_2.exp()[0, 0, sample_index].item()
        cov = np.array([[v1,c],[c,v2]])
        sample_point = np.random.multivariate_normal(mu, cov)

        out = np.insert(sample_point,0,sample_end)
        record.append(out)
        x = torch.cat((torch.from_numpy(out).float().reshape((1, 1, len(out))), scaled_inputs), dim=-1).to(device)
        x = Variable(x, requires_grad=False)

    # Prepare / scale actual spiral
    spiral_batch = [spiral]
    generate_data_spirals.add_eos(spiral_batch)
    spiral = np.array(spiral_batch[0])
    # spiral[:, 1:] = label_scaler.transform(spiral[:, 1:])
    record = np.array(record)
    record[:, 1:] = label_scaler.inverse_transform(record[:, 1:])
    plot_trajectory(record, spiral)


def main(args):
    input_scaler = joblib.load(args.input_scaler_filepath)
    label_scaler = joblib.load(args.label_scaler_filepath)
    generate_unconditionally(input_scaler, label_scaler, args.state_dict_filepath, timesteps=args.timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict_filepath", type=str)
    parser.add_argument("input_scaler_filepath", type=str)
    parser.add_argument("label_scaler_filepath", type=str)
    parser.add_argument("timesteps", type=int)
    main(parser.parse_args())