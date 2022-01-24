"""Initialize the signac project

This script loads the project data and initializes a signac project in the CWD.
"""

import argparse
from pathlib import Path

import signac
from PIL import Image
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('datasets_roots', type=Path, nargs='+', help='Root folder for the dataset partitions of each dataset (in ImageFolder format)')
args = parser.parse_args()

validation_ratio = 0.1
seed = 0

project = signac.init_project('ordinal-comparison')


def add_job(sp, doc):
    if project.find_jobs(sp):
        return 0
    job = project.open_job(sp).init()
    for k, v in doc.items():
        job.doc[k] = v
    return 1


for dataset_folder_path in args.datasets_roots:
    dataset_folder = str(dataset_folder_path)
    print(dataset_folder)

    partition_paths = list(dataset_folder_path.iterdir())
    n_partitions = len(partition_paths)
    first_partition_class_paths = list((partition_paths[0] / 'train').iterdir())
    n_classes = len(first_partition_class_paths)
    image_size = Image.open(list(first_partition_class_paths[0].iterdir())[0]).size

    common_doc = {
        'n_classes': n_classes,
        'image_shape': image_size,
    }

    grid = {
        'partition': list(range(n_partitions)),
        'base_folder': [dataset_folder],
        'classifier_type': ['nominal', 'ordinal_ecoc', 'ordinal_clm', 'ordinal_qwk'],
        'activation_function': ['relu'],
        'model': ['vgg11', 'resnet18', 'mobilenet_v3_large', 'shufflenet_x2_0'],
        'class_weight_factor': [3e-5],
        'l2_penalty': [0.0],
        'learning_rate': [1e-4],
        'batch_size': [72],
        'validation_ratio': [0.1],
        'max_epochs': [200],
        'patience': [5],
        'seed': [seed],
    }

    print('Adding jobs to project')
    total_added = 0
    for p in tqdm(ParameterGrid(grid)):

        if 'vgg' in p['model']:
            p['l2_penalty'] = 5e-4
        elif 'resnet' in p['model']:
            p['l2_penalty'] = 1e-4

        if 'densenet' in p['model']:
            p['batch_size'] = 32
            
        total_added += add_job(p, common_doc)
    print(f'Added {total_added} new jobs')
