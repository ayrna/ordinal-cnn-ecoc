import random
from copy import deepcopy
from numbers import Real
from typing import Callable, Iterator, Dict

import numpy as np
import torch
import torch.utils.data as tdata
from torch import Tensor
from torch.optim import Optimizer

from network_vgg import ExperimentModel
from timer import Timer


def train(net: ExperimentModel, train_dataset: tdata.Dataset, val_dataset: tdata.Dataset,
          batch_size: int, max_epochs: int, patience: int, criterion: Callable[[Tensor, Tensor], Tensor],
          optimizer: Optimizer, device: torch.device, verbose: bool = False) -> Iterator[Dict[str, Real]]:
    def seed_worker(worker_id):
        worker_seed = (torch.initial_seed() * worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dl = tdata.DataLoader(train_dataset, batch_size, shuffle=True,
                                num_workers=10, prefetch_factor=8,
                                worker_init_fn=seed_worker, pin_memory=True)
    validation_dl = tdata.DataLoader(val_dataset, batch_size, shuffle=False,
                              num_workers=10, prefetch_factor=8,
                              worker_init_fn=seed_worker, pin_memory=True)
    net = net.to(device)
    epochs_without_improvement = 0
    train_losses = list()
    validation_losses = list()

    on_batch_end = getattr(net, "on_batch_end", lambda: None)
    # Baseline validation loss
    with torch.no_grad():
        net.eval()
        val_loss = 0.0
        n_val_samples = 0
        for batch in validation_dl:
            inputs, targets = map(lambda t: t.to(device), batch)
            n_val_samples += inputs.size()[0]

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    best_validation_loss = val_loss / n_val_samples
    best_parameters = deepcopy(net.state_dict())

    for epoch in range(max_epochs):
        net.train()
        # Run epoch for every batch from the train DataLoader
        train_loss = 0.0
        n_train_samples = 0
        train_timer = Timer()
        with train_timer:
            for i_batch, batch in enumerate(train_dl):
                inputs, targets = map(lambda t: t.to(device), batch)
                n_train_samples += inputs.size()[0]

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                on_batch_end()

        train_loss = train_loss / n_train_samples
        train_losses.append(train_loss)

        # Validation step
        with torch.no_grad():
            net.eval()
            val_loss = 0.0
            n_val_samples = 0
            val_timer = Timer()
            with val_timer:
                for batch in validation_dl:
                    inputs, targets = map(lambda t: t.to(device), batch)
                    n_val_samples += inputs.size()[0]

                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss = val_loss / n_val_samples
            validation_losses.append(val_loss)
        if verbose:
            print(f'[Epoch {epoch + 1}]\tval loss: {val_loss:.4f}, train loss: {train_loss:.4f}')

        yield {'train_loss': train_loss, 'val_loss': val_loss,
               'train_time': train_timer.elapsed_time, 'val_time': val_timer.elapsed_time}

        # Check for early stopping
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_parameters = deepcopy(net.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f'{patience} epochs without improvement, restoring best '
                      f'parameters and stopping training')
            net.train(False)
            net.load_state_dict(best_parameters)
            break

