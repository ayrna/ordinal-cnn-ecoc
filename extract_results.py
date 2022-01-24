from itertools import chain
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import signac
import argparse
from common import add_filters_to_parser, get_filters_from_args
from project import results_saved


parser = argparse.ArgumentParser()
parser.add_argument('output', help='Output .xlsx path', type=Path)
add_filters_to_parser(parser)
args = parser.parse_args()

filter = get_filters_from_args(args.filter)
doc_filter = get_filters_from_args(args.doc_filter)

project = signac.get_project()

df = defaultdict(lambda: list())
incomplete = False
for j in project.find_jobs(filter, doc_filter):
    if not results_saved(j):
        if not incomplete:
            print("WARNING: incomplete results")
            incomplete = True
        continue

    for key, value in chain(j.sp.items(), j.doc['result_metrics'].items()):
        df[key].append(value)
    df['confusion_matrix'].append(str(j.doc['confusion_matrix']))
    with j.stores.training_data as d:
        df['train_loss'].append(str(np.array(d['train_loss']).tolist()))
        df['val_loss'].append(str(np.array(d['val_loss']).tolist()))
    df['training_time'].append(j.doc['train_network_elapsed_seconds'])

df = pd.DataFrame(df)
df = df.sort_values(by=['base_folder', 'model', 'classifier_type', 'partition'])
df.to_excel(args.output.with_suffix('.xlsx'), index=False)
