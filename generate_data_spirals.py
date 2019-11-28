import argparse
import multiprocessing
import tables
import os
import numpy as np
import math
import random
import joblib

from sklearn.preprocessing import MinMaxScaler


def generate_spiral(vel=0.001, initial_radius=0, turning_distance=0.01, x_extent=0.001, y_extent=0.001):
    b = turning_distance / (2 * math.pi)
    points = []
    t = 1
    x = 0
    y = 0
    while True:
        omega_t = vel / (2 * math.pi * t)
        c = (initial_radius + b * t)
        dx = c * math.cos(t) - x
        dy = c * math.sin(t) - y
        points.append([dx, dy])
        x += dx
        y += dy
        if abs(x) > x_extent or abs(y) > y_extent:
            break
        t += omega_t
    return points[2:] # Throw away degenerate first points


def generate_spirals(max_len = 1000):
    while True:
        path_increment = random.uniform(0.0001, 0.001)
        x_extent = random.uniform(0.0005, 0.002)
        y_extent = random.uniform(0.0005, 0.002)
        velocity = random.uniform(10, 12)
        spiral = generate_spiral(vel=velocity, turning_distance=path_increment, x_extent=x_extent, y_extent=y_extent)
        if len(spiral) < max_len:
            break
    yield (x_extent, y_extent, path_increment, velocity), spiral


def generate_data(n, max_length, q):
    for _ in range(n):
        labels_and_spirals = next(generate_spirals(max_length))
        q.put(labels_and_spirals)


def pad_and_build_masks(labels, seq_len):
    masks = np.zeros((len(labels), seq_len))
    for i in range(len(labels)):
        masks[i][0:len(labels[i]) - 1] = 1
        labels[i] = np.vstack([labels[i], np.zeros(((seq_len + 1) - len(labels[i]), 3))])  # pad to seq_len + 1
    return np.array(labels, dtype=np.float32), masks


def add_eos(labels):
    for seq_idx in range(len(labels)):
        labels[seq_idx] = [[0] + dp for dp in labels[seq_idx]]
        labels[seq_idx][-1][0] = 1


def preprocess_data(raw_inputs, raw_labels, seq_len):
    labels, masks = pad_and_build_masks(raw_labels, seq_len)
    input_scaler = MinMaxScaler(feature_range=(-1,1))
    input_scaler.fit(raw_inputs)
    inputs = np.array(input_scaler.transform(raw_inputs), dtype=np.float32)
    label_scaler = MinMaxScaler(feature_range=(-1,1))
    labels_flattened = labels[:,:,1:].reshape((labels.shape[0] * labels.shape[1], labels.shape[2] - 1))
    label_scaler.fit(labels_flattened)
    labels[:, :, 1:] = label_scaler.transform(labels_flattened).reshape((labels.shape[0], labels.shape[1], labels.shape[2] - 1))
    return inputs, labels, masks, input_scaler, label_scaler


def main(args):
    n_train = int(0.8 * args.n)
    print("Train/validate: {}/{}".format(n_train, args.n - n_train))
    num_processes = multiprocessing.cpu_count() - 1
    print("Generating data using {} processes".format(num_processes))
    items_per_process = int(args.n / num_processes)
    queue = multiprocessing.Queue()
    processes = [multiprocessing.Process(target=generate_data, args=(items_per_process, args.max_length, queue)) for _ in range(num_processes)]
    for process in processes:
        process.start()
    inputs = []
    spirals = []
    while len(inputs) < items_per_process * num_processes:
        params, spiral = queue.get()
        inputs.append(params)
        spirals.append(spiral)
    for process in processes:
        process.join()

    add_eos(spirals)
    inputs, labels, masks, input_scaler, label_scaler = preprocess_data(inputs, spirals, args.max_length)
    num_train_data = int(0.8 * len(labels))
    output_filepath = os.path.join(args.output_dir, "spirals_{}_{}.h5".format(args.max_length, args.n))
    with tables.open_file(output_filepath, "w") as output_file:
        output_file.create_array("/", "train_labels", labels[:num_train_data])
        output_file.create_array("/", "train_inputs", inputs[:num_train_data])
        output_file.create_array("/", "train_masks", masks[:num_train_data])
        output_file.create_array("/", "validate_labels", labels[num_train_data:])
        output_file.create_array("/", "validate_inputs", inputs[num_train_data:])
        output_file.create_array("/", "validate_masks", masks[num_train_data:])
    joblib.dump(label_scaler, os.path.join(args.output_dir, "spirals_{}_{}_label_scaler.pkl".format(args.max_length, args.n)))
    joblib.dump(input_scaler, os.path.join(args.output_dir, "spirals_{}_{}_input_scaler.pkl".format(args.max_length, args.n)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--max_length", type=int, default=1000)
    main(parser.parse_args())
