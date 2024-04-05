import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import argparse
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_param', type=str, help='Name of config file.')
parser.add_argument('--dataset', type=str, help='Name of config file.')
parser.add_argument('--name', type=str, help='Name of tracker config file.')
parser.add_argument('--modelname', type=str, default='exot_st2')
parser.add_argument('--ckpt_name', type=str, default='EXOTST_epoch=49.pth.tar')
args = parser.parse_args()

dataset_name = args.dataset

dataset = get_dataset(dataset_name)
trackers.extend(trackerlist(name=args.name, parameter_name=args.tracker_param, dataset_name=dataset_name, 
                            modelname = args.modelname, ckpt_name = args.ckpt_name,
                            run_ids=None, display_name=args.name))
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
if 'stark' in args.modelname: 
    print_results(trackers, dataset, dataset_name, merge_results=True, force_evaluation=True, plot_types=('success', 'prec', 'norm_prec')) #, 'main_tnr_auroc'))
elif 'exot' in args.modelname: 
    print_results(trackers, dataset, dataset_name, merge_results=True, force_evaluation=True, exclude_invalid_frames=True, plot_types=('success', 'prec', 'norm_prec', 'main_tnr_auroc'))
