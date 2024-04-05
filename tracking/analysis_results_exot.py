import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import argparse
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.tracker.exotst_tracker import EXOTSTTracker
import importlib
from lib.test.evaluation.environment import env_settings
from pl_prac import Settings

trackers = []
parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_param', type=str, help='Name of config file.')
parser.add_argument('--dataset_name', type=str, help='Name of config file.')
parser.add_argument('--name', type=str, help='Name of tracker config file.')
args = parser.parse_args()

dataset_name = args.dataset_name

def get_parameters(name, parameter_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(name))
    params = param_module.parameters(parameter_name)
    return params

dataset = get_dataset(dataset_name)
params = get_parameters(args.name, args.tracker_param)
tracker = [EXOTSTTracker(params, args.name, args.tracker_param, dataset_name)]
 
# trackers.extend(trackerlist(name='stark_st', parameter_name=args.tracker_param, dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(tracker, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
