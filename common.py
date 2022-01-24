import argparse
from typing import List

from flow import FlowProject
from signac.contrib.filterparse import parse_filter_arg


def add_filters_to_parser(parser: argparse.ArgumentParser):
    FlowProject._add_job_selection_args(parser)


def get_filters_from_args(args: List[str]):
    filter = parse_filter_arg(args)
    return filter if filter is not None else dict()
