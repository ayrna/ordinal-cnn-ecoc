import random
import sys
import time
import traceback
import warnings
from datetime import timedelta
from inspect import signature
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
from flow import FlowProject
from multiprocess import Pool  # type: ignore
from scipy.special import softmax
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from metrics import metrics
from network import clm_models, ecoc_models, nominal_models
from network_common import (CumulativeLinkLoss, QWKLoss,
                            ExperimentModel, ordinal_ecoc_distance_loss)
from train import train


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback

experiment = FlowProject.make_group(name='experiment')


def load_partition(job) -> Tuple[ImageFolder, ImageFolder]:
    base_folder = Path(job.sp.base_folder) / f'{job.sp.partition}'
    train_path = base_folder / 'train'
    test_path = base_folder / 'test'

    train_ds = ImageFolder(str(train_path), ToTensor())
    test_ds  = ImageFolder(str(test_path) , ToTensor())

    return train_ds, test_ds


activation_function_by_name: Dict[str, Callable[[], nn.Module]] = {
    'relu': lambda: nn.ReLU(inplace=True),
    'elu': lambda: nn.ELU(inplace=True),
    'softplus': lambda: nn.Softplus(),
}


def training_components(model: str, classifier_type: str, n_classes: int, learning_rate: float,
                        activation_function: str, l2_penalty: float,
                        device: torch.device, class_weights: Optional[Sequence[float]] = None)\
        -> Tuple[ExperimentModel, Callable[[Tensor, Tensor], Tensor], torch.optim.Optimizer]:
    """
    Obtain training components from the configuration.
    Parameters
    ----------
    model
        Name of the model
    classifier_type
        Type of classifier (``'nominal'``, ``ordinal_qwk``, ``'ordinal_clm'`` or ``'ordinal_ecoc'``)
    n_classes
        Number of classes to classify into
    learning_rate
        Learning rate
    activation_function
        Which activation function to use
    class_weights
        Optional class weights for loss function
    Returns
    -------
    net: nn.Module
        CNN model to train
    criterion: nn.Module
        Optimization criterion
    optimizer: optim.Optimizer
        Optimizer method
    """
    if class_weights is not None:
        assert all(isinstance(f, float) for f in class_weights)
        class_weights = torch.tensor(class_weights).float().to(device)  # type: ignore

    activation_function_factory = activation_function_by_name[activation_function]

    if classifier_type == 'nominal':
        net = nominal_models[model](num_classes=n_classes, activation_function=activation_function_factory)
        criterion = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)  # type: ignore
    elif classifier_type == 'ordinal_qwk':
        net = nominal_models[model](num_classes=n_classes, activation_function=activation_function_factory)
        criterion = QWKLoss(n_classes, weight=class_weights).to(device)  # type: ignore
    elif classifier_type == 'ordinal_ecoc':
        net = ecoc_models[model](num_classes=n_classes, activation_function=activation_function_factory)
        criterion = ordinal_ecoc_distance_loss(n_classes, device, class_weights=class_weights)  # type: ignore
    elif classifier_type == 'ordinal_clm':
        net = clm_models[model](num_classes=n_classes, activation_function=activation_function_factory)
        criterion = CumulativeLinkLoss(reduction='sum', class_weights=class_weights)  # type: ignore
    else:
        raise NotImplementedError
    net: ExperimentModel = net.to(device)
    optimizer = torch.optim.Adam([
            {'params': net.non_regularized_parameters()},
            {'params': net.regularized_parameters(), 'weight_decay': l2_penalty},
        ], lr=learning_rate)
    return net, criterion, optimizer


def predictions(net: ExperimentModel, test_ds, device, batch_size=32):
    """
    Use the whole CNN to estimate the class labels from a test dataset.
    Parameters
    ----------
    net : VGG
        CNN model
    test_ds : tdata.Dataset
        Test dataset
    device : torch.device
    batch_size : int
        Batch size to use during testing
    Returns
    -------
    true_labels: np.ndarray
        True labels
    pred_labels: np.ndarray
        Predicted labels
    pred_probas: np.ndarray
        Predicted class probabilities
    """
    net.eval()
    test_dl = tdata.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        true_labels = list()
        pred_labels = list()
        pred_probas = list()
        for batch in test_dl:
            inputs, tl = map(lambda t: t.to(device), batch)
            pl, pp = net.predict(inputs)

            true_labels.append(tl.cpu())
            pred_labels.append(pl)
            pred_probas.append(pp)
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    pred_probas = np.concatenate(pred_probas)
    return true_labels, pred_labels, pred_probas


def seed_from_str(s: str) -> int:
    """
    Obtains an integer seed from a string using the hash function
    """
    return hash(s) % (2 ** 32)


def determinism(seed):
    """
    Uses a given seed to ensure determinism when launching a new experiment
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.manual_seed(seed)


@FlowProject.label
def results_saved(job):
    return ('confusion_matrix' in job.doc.keys()) and \
           ('result_metrics' in job.doc.keys()) and \
           ('result_metrics_per_class' in job.doc.keys())


def evaluation_metrics(ytrue, ypred, probas):
    """
    Compute all the evaluation metrics into a dictionary.
    ytrue
        True labels
    ypred
        Predicted labels
    probas
        Predicted class probabilities
    Returns
    -------
    x:
        Dictionary containing all computed metrics (as listed in C{metrics.metric_list})
    """
    evaluated_metrics = dict()
    for metric_type in ('multinomial', 'ordinal'):
        for metric_name, m in metrics[metric_type]['labels'].items():
            evaluated_metrics[metric_name] = m(ytrue, ypred)
        for metric_name, m in metrics[metric_type]['scores'].items():
            evaluated_metrics[metric_name] = m(ytrue, probas)

    evaluated_metrics_per_class = dict()
    for metric_name, m in metrics['multinomial_per_class']['labels'].items():
        evaluated_metrics_per_class[metric_name] = m(ytrue, ypred)
    for metric_name, m in metrics['multinomial_per_class']['scores'].items():
        evaluated_metrics_per_class[metric_name] = m(ytrue, probas)
    return evaluated_metrics, evaluated_metrics_per_class


def plot_history(job, history, filename):
    fig, ax = plt.subplots()
    ax.plot([h['train_loss'] for h in history], label='train loss')
    ax.plot([h['val_loss'] for h in history], label='val loss')
    ax.legend()
    fig.savefig(job.fn(filename))
    plt.close(fig)


@experiment
@FlowProject.operation.with_directives({'np': 3})  # type: ignore
@FlowProject.post.isfile('trainval_idx.npz')  # type: ignore
@FlowProject.post.isfile('class_weights.npz')  # type: ignore
def split_train_validation(job):
    determinism(job.sp.seed)
    random_state = check_random_state(job.sp.seed)

    trainval_ds, _ = load_partition(job)

    trainval_targets = trainval_ds.targets
    train_idx, val_idx = train_test_split(np.arange(len(trainval_targets)), test_size=job.sp.validation_ratio,
                                          stratify=trainval_targets, shuffle=True, random_state=random_state)
    np.savez(job.fn('trainval_idx.npz'), train_idx=train_idx, val_idx=val_idx)

    train_targets = np.array(trainval_targets)[train_idx]
    train_targets_counts = np.unique(train_targets, return_counts=True)[1]

    class_weights = softmax(-train_targets_counts * job.sp.class_weight_factor)
    np.savez(job.fn('class_weights.npz'), class_weights=class_weights)

@experiment
@FlowProject.operation.with_directives({'ngpu': 1, 'np': 3})  # type: ignore
@FlowProject.pre.after(split_train_validation)  # type: ignore
@FlowProject.post.isfile('trained_state.pt')  # type: ignore
def train_model(job):
    determinism(job.sp.seed)
    device = torch.device('cuda:0')

    trainval_ds, _ = load_partition(job)

    idxs = np.load(job.fn('trainval_idx.npz'))
    train_idx = idxs['train_idx']
    val_idx = idxs['val_idx']

    train_ds = tdata.Subset(trainval_ds, train_idx)
    val_ds = tdata.Subset(trainval_ds, val_idx)

    class_weights = np.load(job.fn('class_weights.npz'))['class_weights']

    net, criterion, optimizer = training_components(**{p: v for p, v in job.sp.items()
                                                       if p in signature(training_components).parameters},
                                                    class_weights=class_weights,
                                                    n_classes=job.doc.n_classes, device=device)

    last_plot = None
    async_pool = Pool(1)
    history = list()
    begin_time = time.time()
    for epoch in train(net, train_ds, val_ds, criterion=criterion, optimizer=optimizer, device=device,
                       **{p: v for p, v in job.sp.items() if p in signature(train).parameters}):
        history.append(epoch)
        if last_plot is not None:
            last_plot.wait()
        last_plot = async_pool.apply_async(plot_history, (job, history, 'training.png'))
    last_plot.wait()  # type: ignore
    end_time = time.time()
    elapsed = end_time - begin_time
    print(f'Ran for {len(history)} epochs. Elapsed time: {timedelta(seconds=elapsed)} (avg {timedelta(seconds=elapsed/len(history))} per epoch)')
    job.doc['train_network_elapsed_seconds'] = elapsed

    with job.stores.training_data as d:
        d['train_loss'] = np.array([e['train_loss'] for e in history])
        d['val_loss'] = np.array([e['val_loss'] for e in history])
        d['train_time'] = np.array([e['train_time'] for e in history])
        d['val_time'] = np.array([e['val_time'] for e in history])
    torch.save(net.state_dict(), job.fn(f'trained_state.pt'))


@experiment
@FlowProject.operation.with_directives({'ngpu': 1, 'np': 3})  # type: ignore
@FlowProject.pre.after(train_model)  # type: ignore
@FlowProject.post(results_saved)  # type: ignore
def evaluate_trained_model(job):
    if results_saved(job):
        return

    determinism(job.sp.seed)
    device = torch.device('cuda:0')

    _, test_ds = load_partition(job)

    net, _, _ = training_components(**{p: v for p, v in job.sp.items()
                                       if p in signature(training_components).parameters},
                                    n_classes=job.doc.n_classes, device=device)
    net.load_state_dict(torch.load(job.fn(f'trained_state.pt'), map_location=device))
    results = evaluate_model(net, test_ds, device)
    for k, v in results:
        job.doc[k] = v

    confusion_matrix = np.array(results['confusion_matrix'])
    weighted_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]
    cmd = ConfusionMatrixDisplay(weighted_confusion_matrix)
    fig, ax = plt.subplots()
    cmd.plot(ax=ax)
    fig.savefig(job.fn('confusion_matrix.png'))


def evaluate_model(net: ExperimentModel, test_ds: tdata.Dataset, device: torch.device) -> dict:
    ytrue, ypred, probas = predictions(net, test_ds, device)
    cm = confusion_matrix(ytrue, ypred)
    em, empc = evaluation_metrics(ytrue, ypred, probas)
    return {
        'confusion_matrix': cm.tolist(),
        'result_metrics': em,
        'result_metrics_per_class': empc
    }


def main():
    FlowProject().main()


if __name__ == '__main__':
    main()
