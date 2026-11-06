"""Deterministic CPU mini-batches, gradient clipping and internal early stopping."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import row_dataset


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def predict_network(network, numeric, categorical, batch_size=1024):
    network.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(numeric), batch_size):
            a = torch.from_numpy(numeric[start:start+batch_size])
            b = torch.from_numpy(categorical[start:start+batch_size])
            predictions.append(network(a, b).numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def train_rows(network, arrays, y, *, epochs, batch_size, learning_rate, weight_decay,
               seed, validation=None, patience=5, verbose=False, sample_weights=None):
    dataset = row_dataset(*arrays, y)
    if sample_weights is None:
        sample_weights = np.ones(len(y), dtype=np.float32)
    dataset = torch.utils.data.TensorDataset(*dataset.tensors, torch.as_tensor(sample_weights, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history, best, best_epoch, stale = [], float("inf"), 1, 0
    for epoch in range(1, epochs + 1):
        network.train()
        total = 0.0
        for numeric, categorical, target, weight in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = ((network(numeric, categorical) - target).square() * weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite NN training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(target)
        record = {"epoch": epoch, "train_mse": total / len(y)}
        if validation is not None:
            valid_arrays, raw_y, target_center, target_scale, rent_scale = validation
            pred = predict_network(network, *valid_arrays, batch_size)
            pred = (pred * target_scale + target_center) * rent_scale
            score = float(np.sqrt(np.mean((raw_y - pred) ** 2)))
            record["inner_rmse_sgd"] = score
            if score < best - 1e-5:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(record)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"    epoch {epoch}/{epochs}: {record}", flush=True)
        if validation is not None and stale >= patience:
            break
    return history, best_epoch if validation is not None else epochs
