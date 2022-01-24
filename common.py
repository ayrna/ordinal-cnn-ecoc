import argparse
from typing import List

from flow import FlowProject
from signac.contrib.filterparse import parse_filter_arg


# def get_frontier_distribution(job):
#     if job.sp.classifier_type == 'nominal':
#         return ""
#
#     if job.sp.distribution == 'beta':
#         xql = job.sp.ordinal_augment_beta_params.xql
#         ql = job.sp.ordinal_augment_beta_params.ql
#         xqu = job.sp.ordinal_augment_beta_params.xqu
#         qu = job.sp.ordinal_augment_beta_params.qu
#         return f'beta, P(X < {xql}) = {ql}, P(X < {xqu}) = {qu}'
#     elif 'ordinal_augment_gamma_params' in job.sp:
#         shape = job.sp.ordinal_augment_gamma_params.shape
#         scale = job.sp.ordinal_augment_gamma_params.scale
#         return f'gamma, shape={shape}, scale={scale}'
#     else:
#         raise NotImplementedError


def add_filters_to_parser(parser: argparse.ArgumentParser):
    FlowProject._add_job_selection_args(parser)


def get_filters_from_args(args: List[str]):
    filter = parse_filter_arg(args)
    return filter if filter is not None else dict()
