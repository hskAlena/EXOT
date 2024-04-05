import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch
import wandb


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    '''2021.1.5 create new folder for these two datasets'''
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    elif seq.dataset in ['robot']:
        # name = '-'.join(['/'.join(seq.name.split('/')[:-2]), *seq.name.split('/')[-2:]])
        name = '/'.join(seq.name.split('/')[:-1])
        if not os.path.exists(os.path.join(tracker.results_dir, name)):
            os.makedirs(os.path.join(tracker.results_dir, name))
        base_results_path = os.path.join(tracker.results_dir, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key in [ 'visgt', 'objgt', 'predobj'] :
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)
                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_{}.txt'.format(base_results_path, obj_id, key)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                print(f"saving {key}...")
                bbox_file = '{}_{}.txt'.format(base_results_path, key)
                save_bb(bbox_file, data)
                #'conf_score', 'objconf',
        if key in [ 'conf_score', 'objconf'] :
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)
                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_{}.txt'.format(base_results_path, obj_id, key)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print(f"saving {key}...")
                bbox_file = '{}_{}.txt'.format(base_results_path, key)
                save_score(bbox_file, data)
        if key in [ 'results_old', 'neg_threshold', 'results_new', 'results_old_odin'] :
            # Single-object mode
            print(f"saving {key}...")
            bbox_file = '{}_{}.txt'.format(base_results_path, key)
            with open(bbox_file, 'w') as f:
                if isinstance(data, (dict, OrderedDict)):
                    for i, d in data['generalized_odin'].items():
                        f.write(i +'\t'+str(d)+'\n')
                else:
                    f.write(key+'\t'+str(data)+'\n')
            # save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    wandb.init(project="EXOT-visualize-predictions", name=f'{tracker.modelname}-{tracker.parameter_name}-{seq.dataset}')
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        #torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)

def run_epsilon(seq: Sequence, tracker: Tracker, epsilon_grid, version, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    wandb.init(project="EXOT-cal_epsilon", name=f'{tracker.modelname}-{tracker.parameter_name}-{seq.dataset}')
    
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        #torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.modelname, tracker.parameter_name, tracker.run_id, seq.name))

    mean_scores = {}
    if debug:
        for epsilon in epsilon_grid:
            output, epout = tracker.run_epsilon(seq, epsilon, version, debug=debug)
            print("RESULT ", epsilon, "is :", epout)
            mean_scores[epsilon] = epout
            wandb.log({'epsilon_loop': epsilon})
            wandb.log({'mean_epsilon': epout})
        best_epsilon = min(mean_scores, key=(lambda key: mean_scores[key]))
        final_eps = best_epsilon / 2.
        print(f"Epsilon: {best_epsilon / 2.}")
    else:
        try:
            for epsilon in epsilon_grid:
                output, epout = tracker.run_epsilon(seq, epsilon, version, debug=debug)
                print("RESULT ", epsilon, "is :", epout)
                mean_scores[epsilon] = epout
                wandb.log({'epsilon_loop': epsilon})
                wandb.log({'mean_epsilon': epout})
            best_epsilon = min(mean_scores, key=(lambda key: mean_scores[key]))
            final_eps = best_epsilon / 2.
            wandb.log({'final epsilon': final_eps})
            print(f"Epsilon: {best_epsilon / 2.}")
        except Exception as e:
            print(e)
            return
    
    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    # if not debug:
    #     _save_tracker_output(seq, tracker, output)

    return final_eps

def run_odin_test(seq: Sequence, tracker: Tracker, epsilon, version, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    wandb.init(project="EXOT-visualize-predictions", name=f'{tracker.modelname}-{tracker.parameter_name}-{seq.dataset}')

    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        #torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.modelname, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_odintest(seq, epsilon, version, debug=debug)
    else:
        try:
            output = tracker.run_odintest(seq, epsilon, version, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)

def run_dataset(dataset, trackers, run_fn, epsilon_grid, epsilon, version, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                if run_fn == 'run_seq':
                    run_sequence(seq, tracker_info, debug=debug)
                elif run_fn == 'run_epsilon':
                    epsilon = run_epsilon(seq, tracker_info, epsilon_grid, version, debug)
                    # elif run_fn == 'run_odin_test':
                    run_odin_test(seq, tracker_info, epsilon, version, debug)
                
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            if run_fn == 'run_seq':
                pool.starmap(run_sequence, param_list)
            elif run_fn == 'run_epsilon':
                pool.starmap(run_epsilon, param_list)
            elif run_fn == 'run_odin_test':
                pool.starmap(run_odin_test, param_list)
    print('Done')
