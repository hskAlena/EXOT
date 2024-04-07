import torch
import matplotlib.pyplot as plt
import numpy as np
# from lib.pylight import LitEXOTActor, LitEXOTSTActor, LitODINActor, RobotDataModule
# from lib.models.exot import build_exotst_odin

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import _save_tracker_output
from lib.test.tracker.exotst_tracker import EXOTSTTracker
from collections import OrderedDict
import time
from lib.utils.lmdb_utils import decode_img
import argparse
import importlib
import cv2 as cv
from pl_prac import Settings
import tqdm
import wandb
import sys, os



def parse_args():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--main', type=str, default="main")
    parser.add_argument('--epsilon', type=float, default=0.005)

    args = parser.parse_args()
    return args

def parse_args_jup(args):
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=2)

    args = parser.parse_args(args)
    return args

def _results_exist(seq, tracker):
    if seq.object_ids is None:
        # if seq.dataset in ['robot']:
        #     name = '-'.join(['/'.join(seq.name.split('/')[:-2]), *seq.name.split('/')[-2:]])
        #     base_results_path = os.path.join(tracker.results_dir, name)
        # else:
        bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
        return os.path.isfile(bbox_file)
    else:
        bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
        missing = [not os.path.isfile(f) for f in bbox_files]
        return sum(missing) == 0

    

def run_tracker(dataset_name, name, parameter_name, epsilon):
    dataset = get_dataset(dataset_name)
    print(len(dataset))
    dataset_mean = []
    params = get_parameters(name, parameter_name)
    tracker = EXOTSTTracker(params, name, parameter_name, dataset_name)
    for seq in tqdm.tqdm(dataset):
        if _results_exist(seq, tracker):
            print('FPS: {}'.format(-1))
            continue
        output = track_out(tracker, seq, epsilon)
        print('Tracker: {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, seq.name))
        sys.stdout.flush()
        meanout = np.mean(np.array(output['odin_meanscore']))
        dataset_mean.append(meanout)
        print(meanout)
        wandb.log({"epsilon_seq": meanout})
    
    return np.mean(np.array(dataset_mean))


def track_out(tracker, seq, epsilon):    
    # print("PARAMSS", params)
    # Get init information
    init_info = seq.init_info()    

    output = {'target_bbox': [],
                'time': [], 
                'odin_meanscore': []}
    if tracker.params.save_all_boxes:
        output['all_boxes'] = []
        output['all_scores'] = []

    # Initialize
    image = _read_image(seq.frames[0])

    start_time = time.time()
    out = tracker.initialize(image, init_info)
    if out is None:
        out = {}

    prev_output = OrderedDict(out)
    init_default = {'target_bbox': init_info.get('init_bbox'),
                    'time': time.time() - start_time}
    if tracker.params.save_all_boxes:
        init_default['all_boxes'] = out['all_boxes']
        init_default['all_scores'] = out['all_scores']

    _store_outputs(out, output, init_default)

    for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
        image = _read_image(frame_path)

        start_time = time.time()
        # print(frame_num)
        info = seq.frame_info(frame_num)
        info['previous_output'] = prev_output

        out = tracker.odin_new_track(image, epsilon, info)
        prev_output = OrderedDict(out)
        _store_outputs(out, output, {'time': time.time() - start_time})
    
    for key in ['target_bbox', 'all_boxes', 'all_scores']:
        if key in output and len(output[key]) <= 1:
            output.pop(key)

    return output

def track_odin_test(tracker, seq, epsilon):
    
    # print("PARAMSS", params)

    # Get init information
    init_info = seq.init_info()

    

    output = {'target_bbox': [],
                'time': [], 
                "conf_score": [],
                'objgt': [], 
                'visgt': [], 
                'objconf': [], 
                'predobj': []
                }
    if tracker.params.save_all_boxes:
        output['all_boxes'] = []
        output['all_scores'] = []

    # Initialize
    image = _read_image(seq.frames[0])

    start_time = time.time()
    out = tracker.initialize(image, init_info)
    if out is None:
        out = {}

    prev_output = OrderedDict(out)
    init_default = {'target_bbox': init_info.get('init_bbox'),
                    'time': time.time() - start_time}
    if tracker.params.save_all_boxes:
        init_default['all_boxes'] = out['all_boxes']
        init_default['all_scores'] = out['all_scores']

    _store_outputs(out, output, init_default)

    for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
        image = _read_image(frame_path)

        start_time = time.time()

        info = seq.frame_info(frame_num)
        info['previous_output'] = prev_output

        out = tracker.odin_test_track(image, epsilon, info)
        prev_output = OrderedDict(out)
        _store_outputs(out, output, {'time': time.time() - start_time})
    
    for key in ['target_bbox', 'all_boxes', 'all_scores']:
        if key in output and len(output[key]) <= 1:
            output.pop(key)

    return output

#####################################################################################
def test_tracker(dataset_name, name, parameter_name, epsilon):
    dataset = get_dataset(dataset_name)
    print(len(dataset))
    params = get_parameters(name, parameter_name)
    tracker = EXOTSTTracker(params, name, parameter_name, dataset_name)
    for seq in dataset:
        if _results_exist(seq, tracker):
            print('FPS: {}'.format(-1))
            continue
        output = track_odin_test(tracker, seq, epsilon)
        print('Tracker: {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, seq.name))
        sys.stdout.flush()
        vispred = (np.array(output['conf_score'])>0.5).astype(np.float)
        visconf = np.array(output['conf_score'])
        objconf = np.array(output['objconf'])
        wandb.log({"negative num_frames": np.sum(np.array(output['visgt']))})
        # print("NEGATIVE NUM ", np.sum(np.array(output['visgt'])))
        exit_gt = 1 - np.array(output['visgt'])  # exit => 0, 
        exit_bool = np.array(output['visgt'], dtype=bool)  # exit ==true, 
        # print("NEGATIVE INDEX ", np.nonzero(exit_bool))
        conf_out = objconf[exit_bool].reshape(-1, 1)
        conf_in = objconf[~exit_bool].reshape(-1, 1)
        results, tp, fp, tnr_at_tpr95, neg_threshold = metric(conf_in, conf_out, "generalized_odin")
        objpred = (objconf>neg_threshold).astype(np.float)

        objlabel_gt = np.array(output['objgt'])
        objlabel_pred = np.array(output['predobj'])

        x_values = list(range(len(seq.frames)))

        def draw_wandb_plot(customname, y_values):
            plotname = seq.name
            plotTitle = customname
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({plotname+'/'+plotTitle : wandb.plot.line(table, "x", "y", title=plotTitle)})

        # plt.plot(vispred)
        # wandb.log(, plt)
        
        draw_wandb_plot("vispred", vispred)
        draw_wandb_plot("visconf", visconf)
        draw_wandb_plot("exit_gt", exit_gt)
        draw_wandb_plot("objconf", objconf)
        draw_wandb_plot("objlabel_gt", objlabel_gt)
        draw_wandb_plot("objlabel_pred", objlabel_pred)
        draw_wandb_plot("objpred", objpred)

        print(results)
        wandb.log({"tnr": results['generalized_odin']['TNR']})
        wandb.log({"auroc": results['generalized_odin']['AUROC']})

def test_tracker_print(dataset_name, name, parameter_name, epsilon):
    dataset = get_dataset(dataset_name)
    print(len(dataset))
    params = get_parameters(name, parameter_name)
    tracker = EXOTSTTracker(params, name, parameter_name, dataset_name)

    for seq in dataset:
        if _results_exist(seq, tracker):
            print('FPS: {}'.format(-1))
            continue
        output = track_odin_test(tracker, seq, epsilon)
        print('Tracker: {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, seq.name))
        sys.stdout.flush()
        _save_tracker_output(seq, tracker, output)
        

def _store_outputs(tracker_out: dict, output, defaults=None):
    defaults = {} if defaults is None else defaults
    for key in output.keys():
        val = tracker_out.get(key, defaults.get(key, None))
        if key in tracker_out or val is not None:
            output[key].append(val)


def _read_image(image_file: str):
    if isinstance(image_file, str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    elif isinstance(image_file, list) and len(image_file) == 2:
        return decode_img(image_file[0], image_file[1])
    else:
        raise ValueError("type of image_file should be str or list")

def get_parameters(name, parameter_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(name))
    params = param_module.parameters(parameter_name)
    return params

#############################################################################################
def main(args):    
    # args = parse_args()
    wandb.init(project="EXOT-cal_epsilon", name=f'exotst_testparam-{args.tracker_param}-{args.dataset_name}')

    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    num_gpu = args.num_gpus
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        #torch.cuda.set_device(gpu_id)
    except:
        pass
    epsilon_grid = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    '''
    RESULT  0.0025 is : -2.645319
    RESULT  0.005 is : -2.6748314
    RESULT  0.01 is : -2.702175
    RESULT  0.02 is : -2.6883633
    RESULT  0.04 is : -2.5800993
    RESULT  0.08 is : -2.4553664
    '''   
    
    mean_scores = {}

    for epsilon in epsilon_grid:
        epout = run_tracker(args.dataset_name, "exotst_testparam", args.tracker_param, epsilon)
        print("RESULT ", epsilon, "is :", epout)
        mean_scores[epsilon] = epout
        wandb.log({'epsilon_loop': epsilon})
        wandb.log({'mean_epsilon': epout})

    best_epsilon = min(mean_scores, key=(lambda key: mean_scores[key]))
    print(f"Epsilon: {best_epsilon / 2.}")

def test_main(args, epsilon):
    
    wandb.init(project="EXOT-visualize-predictions", name=f'exotst_testparam-{args.tracker_param}-{args.dataset_name}')

    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    num_gpu = args.num_gpus
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        #torch.cuda.set_device(gpu_id)
    except:
        pass

    test_tracker(args.dataset_name, "exotst_testparam", args.tracker_param, epsilon)

def testprint_main(args, epsilon):
    
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    num_gpu = args.num_gpus
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        #torch.cuda.set_device(gpu_id)
    except:
        pass

    test_tracker_print(args.dataset_name, "exotst_testparam", args.tracker_param, epsilon)



def calculate_auroc(confidence_in, confidence_out, stype):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()


    confidence_in.sort()
    confidence_out.sort()

    # print(confidence_in.shape, confidence_out.shape)
    # (1027, 1) (144, 1)
    if confidence_out.shape[0] == 0:
        print("No outlier exist")
        return
    # print(np.max(confidence_in))

    end = np.max([np.max(confidence_in), np.max(confidence_out)])
    start = np.min([np.min(confidence_in), np.min(confidence_out)])
    wandb.log({"max confidence": end})
    wandb.log({"min confidence": start})
    # print(end, start)

    num_k = confidence_in.shape[0]
    num_n = confidence_out.shape[0]
    tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
    fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
    tp[stype][0], fp[stype][0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[stype][l + 1 :] = tp[stype][l]
            fp[stype][l + 1 :] = np.arange(fp[stype][l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[stype][l + 1 :] = np.arange(tp[stype][l] - 1, -1, -1)
            fp[stype][l + 1 :] = fp[stype][l]
            break
        else:
            if confidence_out[n] < confidence_in[k]:
                n += 1
                tp[stype][l + 1] = tp[stype][l]
                fp[stype][l + 1] = fp[stype][l] - 1
            else:
                k += 1
                tp[stype][l + 1] = tp[stype][l] - 1
                fp[stype][l + 1] = fp[stype][l]

    # print(list(tp[stype]))
    # print(list(fp[stype]))
    # print()
    tpr95_pos = np.abs(tp[stype] / num_k - 0.95)
    tnr95_pos = np.abs(fp[stype] / num_n - 0.95)
    # print("95", list(tpr95_pos))
    wandb.log({"THRESHOLD confidence out": confidence_out[n-1]})
    neg_threshold = confidence_out[n-1]
    tpr95_pos = tpr95_pos.argmin()
    tnr95_pos = tnr95_pos.argmin()
    # print("ARGMIN", tpr95_pos)
    tnr_at_tpr95[stype] = 1.0 - fp[stype][tpr95_pos] / num_n

    return tp, fp, tnr_at_tpr95, neg_threshold

def metric(in_loader, out_loader, stype):
    tp, fp, tnr_at_tpr95, neg_threshold = calculate_auroc(in_loader, out_loader, stype)

    results = dict()
    results[stype] = dict()

    # TNR
    mtype = "TNR"
    results[stype][mtype] = tnr_at_tpr95[stype]

    # AUROC
    mtype = "AUROC"
    tpr = np.concatenate([[1.0], tp[stype] / tp[stype][0], [0.0]])
    fpr = np.concatenate([[1.0], fp[stype] / fp[stype][0], [0.0]])
    results[stype][mtype] = -np.trapz(1.0 - fpr, tpr)

    return results, tp, fp, tnr_at_tpr95, neg_threshold

if __name__ == "__main__":
    args = parse_args()
    if args.main == 'main':
        main(args)
    elif args.main == 'test':        
        test_main(args, args.epsilon)
    elif args.main == 'test_print':        
        testprint_main(args, args.epsilon)