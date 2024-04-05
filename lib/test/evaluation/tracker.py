import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import wandb
import math
import pandas
import copy

def bsr3(ax,ay,az):
    robot_pose=[0,0,0,0,0,0]
    robot_pose[0]=-ax
    robot_pose[1]=(ay+az)/math.sqrt(2)
    robot_pose[2]=(-ay+az)/math.sqrt(2)
    return robot_pose

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def trackerlist(name: str, parameter_name: str, dataset_name: str, modelname = None, ckpt_name = None, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, modelname, ckpt_name, run_id, display_name, result_only) for run_id in run_ids]

def calculate_auroc(confidence_in, confidence_out, stype, wandb_=True):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    tpr_at_tnr95 = dict()


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

    if wandb_:
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

    if wandb_:
        wandb.log({"THRESHOLD confidence out": confidence_out[n-1]})
    neg_threshold = confidence_out[n-1]
    tpr95_pos = tpr95_pos.argmin()
    tnr95_pos = tnr95_pos.argmin()
    # print("ARGMIN", tpr95_pos)
    tnr_at_tpr95[stype] = 1.0 - fp[stype][tpr95_pos] / num_n
    tpr_at_tnr95[stype] = 1.0 - tp[stype][tnr95_pos] / num_k

    return tp, fp, tnr_at_tpr95, neg_threshold

def metric(in_loader, out_loader, stype, wandb_ = True):
    tp, fp, tnr_at_tpr95, neg_threshold = calculate_auroc(in_loader, out_loader, stype, wandb_)

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

class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, modelname = None, ckpt_name = None, run_id: int = None, display_name: str = None,
                 result_only=False, wandb_=True):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.modelname = modelname
        self.ckpt_name = ckpt_name
        self.tmp_name = self.ckpt_name.split('.')[0]#.split('/')[-1]
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.wandb_ = wandb_

        env = env_settings()
        if self.run_id is None:
            # self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            #tmp = self.ckpt_name.split('.')[0].split('/')[-1]
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.modelname, self.tmp_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.modelname, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.modelname)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        print("start of run seq")
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)

        x_values = list(range(len(seq.frames)))

        def draw_wandb_plot(customname, y_values):
            plotname = seq.name
            plotTitle = customname
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({plotname+'/'+plotTitle : wandb.plot.line(table, "x", "y", title=plotTitle)})

        vispred = (np.array(output['conf_score'])>0.5).astype(np.float)  #exit = 0
        visconf = np.array(output['conf_score'])
        # objconf = np.array(output['objconf'])
        
        # # print("NEGATIVE NUM ", np.sum(np.array(output['visgt'])))
        exit_gt = np.array(output['visgt'])  # exit => 0, 
        exit_bool = np.array(output['visgt'], dtype=bool)  # exit ==true, 
        # print("NEGATIVE INDEX ", np.nonzero(exit_bool))
        conf_out = visconf[exit_bool].reshape(-1, 1)
        conf_in = visconf[~exit_bool].reshape(-1, 1)
        results, tp, fp, tnr_at_tpr95, neg_threshold = metric(conf_in, conf_out, "generalized_odin", wandb_=False) #self.wandb_)
        vispred2 = (visconf>=neg_threshold).astype(np.float)  # exit

        output['results_old'] = results
        output['neg_threshold'] = neg_threshold

        if self.wandb_:
            draw_wandb_plot("vispred", vispred)
            draw_wandb_plot("visconf", visconf)
            draw_wandb_plot("exit_gt", exit_gt)

        conf_out = vispred[exit_bool].reshape(-1, 1)
        conf_in = vispred[~exit_bool].reshape(-1, 1)
        results2, tp, fp, tnr_at_tpr95, neg_threshold = metric(conf_in, conf_out, "generalized_odin", wandb_=False) #self.wandb_)
        output['results_new'] = results2

        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

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

        print("is track sequence on??")

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

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

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_epsilon(self, seq, epsilon, version, debug=None):
        # dataset = get_dataset(dataset_name)
        # print(len(dataset))
        # dataset_mean = []
        # params = get_parameters(name, parameter_name)
        # tracker = EXOTSTTracker(params, name, parameter_name, dataset_name)

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()
        dataset_mean = []

        tracker = self.create_tracker(params)
        output = self._track_epsilon(tracker, seq, epsilon, version, init_info)

        meanout = np.mean(np.array(output['odin_meanscore']))
        dataset_mean.append(meanout)
        # print(meanout)
        #wandb.log({"epsilon_seq": meanout})        
        
        return output, np.mean(np.array(dataset_mean))

    def _track_epsilon(self, tracker, seq, epsilon, version, init_info):
        output = {'target_bbox': [],
                'time': [], 
                'odin_meanscore': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

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

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.odin_track(image, epsilon, version, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def _track_odin_test(self, tracker, seq, epsilon, version, init_info):
    
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

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

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

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.odin_test_track(image, epsilon, version, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})
        
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_odintest(self, seq, epsilon, version, debug=None):
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_odin_test(tracker, seq, epsilon, version, init_info)

        vispred = (np.array(output['conf_score'])>0.5).astype(np.float)  #exit = 0
        visconf = np.array(output['conf_score'])
        objconf = np.array(output['objconf'])
        
        # print("NEGATIVE NUM ", np.sum(np.array(output['visgt'])))
        exit_gt = np.array(output['visgt'])  # exit => 0, 
        exit_bool = np.array(output['visgt'], dtype=bool)  # exit ==true, 
        # print("NEGATIVE INDEX ", np.nonzero(exit_bool))
        conf_out = objconf[exit_bool].reshape(-1, 1)
        conf_in = objconf[~exit_bool].reshape(-1, 1)
        results, tp, fp, tnr_at_tpr95, neg_threshold = metric(conf_in, conf_out, "generalized_odin")
        objpred = (objconf>=neg_threshold).astype(np.float)  # exit

        objlabel_gt = np.array(output['objgt'])
        objlabel_pred = np.array(output['predobj'])

        x_values = list(range(len(seq.frames)))
        output['results_new'] = results
        output['neg_threshold'] = neg_threshold

        conf_out = vispred[exit_bool].reshape(-1, 1)
        conf_in = vispred[~exit_bool].reshape(-1, 1)
        results2, tp, fp, tnr_at_tpr95, neg_threshold = metric(conf_in, conf_out, "generalized_odin")

        output['results_old'] = results2

        conf_out = visconf[exit_bool].reshape(-1, 1)
        conf_in = visconf[~exit_bool].reshape(-1, 1)
        results3, tp, fp, tnr_at_tpr95, neg_threshold = metric(conf_in, conf_out, "generalized_odin")

        output['results_old_odin'] = results3


        
        def draw_wandb_plot(customname, y_values):
            plotname = seq.name
            plotTitle = customname
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({plotname+'/'+plotTitle : wandb.plot.line(table, "x", "y", title=plotTitle)})

        # plt.plot(vispred)
        # wandb.log(, plt)
        wandb.log({"negative num_frames": np.sum(np.array(output['visgt']))})
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
        
        return output


    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_video_annot_odin(self, videofilepath, neg_thres=0.55, epsilon=0.005, avg_thres=20, version='cos', optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()
        #params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        # assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        # ", videofilepath must be a valid videofile"

        output_boxes = []
        conf_history = []
        print(videofilepath)
        if (os.path.isdir(videofilepath)):
            image_idx = 0
            images = sorted(list(Path(videofilepath).glob('*.jpg')))
            frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
            success = True
        elif (os.path.isfile(videofilepath)):
            cap = cv.VideoCapture(videofilepath)
            success, frame = cap.read()
        else:
            assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        display_name = 'Display: ' + videofilepath #tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            if 'TREK-150' in videofilepath:
                optional_box[:,0] = optional_box[:,0]/1920*456
                optional_box[:,1] = optional_box[:,1]/1080*256
                optional_box[:,2] = optional_box[:,2]/1920*456
                optional_box[:,3] = optional_box[:,3]/1080*256
            firstbox = list(optional_box[0])
            assert isinstance(firstbox, (list, tuple))
            assert len(firstbox) == 4, "valid box's foramt is [x,y,w,h]"
            print("INIT BOX", firstbox)
            tracker.initialize(frame, _build_init_info(firstbox))
            output_boxes.append(firstbox)
        else:
            # raise NotImplementedError("We haven't support cv_show now.")
            while True:
                if (os.path.isdir(videofilepath)):
                    image_idx +=1
                    frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
                elif (os.path.isfile(videofilepath)):
                    ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                # cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                cv.destroyAllWindows()
                break
                

        flag = 0
        while True:            
            if (os.path.isdir(videofilepath)):
                image_idx +=1
                if image_idx == len(images):
                    break
                frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
            elif (os.path.isfile(videofilepath)):
                ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            out = tracker.odin_test_track(frame, epsilon, version) #0.005

            basic_exit = out['conf_score']
            pred_exit = out['objconf']
            pred_obj = out['predobj']
            conf_history.append(pred_exit)
            if len(conf_history)>=avg_thres:
                exitscore = moving_average(np.array(conf_history[-avg_thres:]), avg_thres)
            else:
                tmpthres = len(conf_history)
                exitscore = moving_average(np.array(conf_history), tmpthres)
            state = [int(s) for s in out['target_bbox']]

            output_boxes.append(state)

            # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            #              (0, 255, 0), 5)

            font_color = (255, 0, 0)
            red_font = (0,0,255)

            if basic_exit<=0.5:
                exit_flag_b = True
            else:
                exit_flag_b = False

            if exitscore<neg_thres:
                pred_exit = exitscore #sigmoid(pred_exit)
                exit_flag_a = True
                flag = 1
            else:
                pred_exit = exitscore
                exit_flag_a = False
                flag = 0

            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            if optional_box is not None:
                if (os.path.isdir(videofilepath)):
                    cv.rectangle(frame_disp, (optional_box[image_idx][0], optional_box[image_idx][1]), 
                                (optional_box[image_idx][2] + optional_box[image_idx][0], optional_box[image_idx][3] + optional_box[image_idx][1]),
                            (255, 255, 255), 5)
                elif (os.path.isfile(videofilepath)):
                    # cv.rectangle(frame_disp, (optional_box[image_idx][0], optional_box[image_idx][1]), 
                    #             (optional_box[image_idx][2] + optional_box[image_idx][0], optional_box[image_idx][3] + optional_box[image_idx][1]),
                    #         (255, 255, 255), 5)
                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)
                
            if exit_flag_b:
                cv.putText(frame_disp, 'basic score: %.4f'%(basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        red_font, 1)
            else:
                cv.putText(frame_disp, 'basic score: %.4f'%(basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)
                # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                #          (0, 255, 0), 5)
            if exit_flag_a:
                cv.putText(frame_disp, 'pred score: %.4f'%(pred_exit), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        red_font, 1)
            else:
                cv.putText(frame_disp, 'pred score: %.4f'%(pred_exit), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)
            # cv.putText(frame_disp, 'pred label: '+str(pred_obj), (20, 105), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #             font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                output_boxes.append(state)
                cv.destroyAllWindows()
                break
            elif key == ord("p"):
                output_boxes.append(state)
                frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
                print("FRAMe indx :   ", frame_idx)
                cv.waitKey(-1)
                # break
            elif key == ord("n"):
                if flag == 0:
                    flag = 1
                    print("NEGATIVE START  ")
                else:
                    flag = 0
                    print("NEGATIVE END  ")
                output_boxes.append([-1, -1, -1, -1])
                frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
                print("FRAME indx :   ", frame_idx)
                cv.waitKey(-1)
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                # cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
            else:
                if flag == 0:
                    output_boxes.append(state)
                else:
                    output_boxes.append([-1, -1, -1, -1])

        # When everything done, release the capture

        if (os.path.isfile(videofilepath)):
            cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_video_annot_stark(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()
        #params = self.params
        optional_box = None

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output_boxes = []
        conf_history = []
        print(videofilepath)
        if (os.path.isdir(videofilepath)):
            image_idx = 0
            images = sorted(list(Path(videofilepath).glob('*.jpg')))
            frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
            
            success = True
        elif (os.path.isfile(videofilepath)):
            cap = cv.VideoCapture(videofilepath)
            success, frame = cap.read()
        else:
            assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        display_name = 'Display: ' + videofilepath #tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        # cv.imshow(display_name, frame)
        img_w, img_h = frame.shape[0], frame.shape[1]
        file_name = 'auto/BeigeCube-8-23-18-35-7'
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 30
        vid_filename = '_'.join(file_name.split('/'))
        out_mv = cv.VideoWriter(vid_filename+'.avi', fourcc, fps, (img_w, img_h))

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            if 'TREK-150' in videofilepath:
                optional_box[:,0] = optional_box[:,0]/1920*456
                optional_box[:,1] = optional_box[:,1]/1080*256
                optional_box[:,2] = optional_box[:,2]/1920*456
                optional_box[:,3] = optional_box[:,3]/1080*256
            firstbox = list(optional_box[0])
            assert isinstance(firstbox, (list, tuple))
            assert len(firstbox) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(firstbox))
            output_boxes.append(firstbox)
        else:
            # raise NotImplementedError("We haven't support cv_show now.")
            while True:
                if (os.path.isdir(videofilepath)):
                    image_idx +=1
                    frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
                elif (os.path.isfile(videofilepath)):
                    ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                # cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                cv.destroyAllWindows()
                break
                

        flag = 0
        image_idx = 0
        while True:  
                     
            if (os.path.isdir(videofilepath)):
                
                if image_idx == len(images):
                    break
                frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
            elif (os.path.isfile(videofilepath)):
                ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            out = tracker.track(frame) #0.005

            basic_exit = out['conf_score']

            conf_history.append(basic_exit)

            state = [int(s) for s in out['target_bbox']]

            output_boxes.append(state)
            font_color = (255, 0, 0)
            red_font = (0,0,255)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)
            if optional_box is not None:
                cv.rectangle(frame_disp, (optional_box[image_idx][0], optional_box[image_idx][1]), 
                             (optional_box[image_idx][2] + optional_box[image_idx][0], optional_box[image_idx][3] + optional_box[image_idx][1]), font_color, 5)

            

            if basic_exit<=0.5:
                exit_flag_b = True
                flag = 1
            else:
                exit_flag_b = False
                flag = 0

            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            if exit_flag_b:
                cv.putText(frame_disp, 'basic score: %.4f'%(basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        red_font, 1)
            else:
                cv.putText(frame_disp, 'basic score: %.4f'%(basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                output_boxes.append(state)
                cv.destroyAllWindows()
                break
            elif key == ord("p"):
                output_boxes.append(state)
                frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
                print("FRAMe indx :   ", frame_idx)
                cv.waitKey(-1)
                # break
            elif key == ord("n"):
                if flag == 0:
                    flag = 1
                    print("NEGATIVE START  ")
                else:
                    flag = 0
                    print("NEGATIVE END  ")
                output_boxes.append([-1, -1, -1, -1])
                frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
                print("FRAME indx :   ", frame_idx)
                cv.waitKey(-1)
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                # cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
            else:
                if flag == 0:
                    output_boxes.append(state)
                else:
                    output_boxes.append([-1, -1, -1, -1])

            cv.imshow(display_name, frame_disp)
            cv.imwrite(f'test/datas/{vid_filename}_{image_idx}.png', frame_disp)
            out_mv.write(frame_disp)
            image_idx +=1 
        
        # When everything done, release the capture
        if (os.path.isfile(videofilepath)):
            cap.release()
            out_mv.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_video_save_jpg(self, videofilepath, neg_thres=0.55, epsilon=0.005, avg_thres=20, version='cos', optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()
        #params = self.params
        file_name = 'auto/TennisBall-8-23-19-10-44'

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        print(self.name, self.ckpt_name.split('.')[0], file_name)
        tmp = self.ckpt_name.split('.')[0]
        bbox_txt = f'test/tracking_results/safe/{self.name}/{tmp}/{file_name}.txt'
        bboxes = pandas.read_csv(bbox_txt, delimiter='\t', header=None, dtype=np.int, na_filter=False, low_memory=False).values
        objconf_txt = f'test/tracking_results/safe/{self.name}/{tmp}/{file_name}_objconf.txt'
        objconf_history = pandas.read_csv(objconf_txt, header=None, dtype=np.float, na_filter=False, low_memory=False).values
        conf_txt = f'test/tracking_results/safe/{self.name}/{tmp}/{file_name}_conf_score.txt'
        conf_history = pandas.read_csv(conf_txt, header=None, dtype=np.float, na_filter=False, low_memory=False).values

        
        print(videofilepath)
        if (os.path.isdir(videofilepath)):
            image_idx = 0
            images = sorted(list(Path(videofilepath).glob('*.png')))
            frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
            success = True
            print(frame.shape)
            img_w, img_h = frame.shape[0], frame.shape[1]
        elif (os.path.isfile(videofilepath)):
            cap = cv.VideoCapture(videofilepath)
            success, frame = cap.read()
        else:
            assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        display_name = 'Display: ' + videofilepath #tracker.params.tracker_name

        def lengthen(target, images):
            if len(target)<len(images):
                head = np.expand_dims(copy.deepcopy(target[0]), 0)
                target = np.concatenate([head, target], axis=0) 
            return target
        objconf_history = lengthen(objconf_history, images)
        conf_history = lengthen(conf_history, images)
        bboxes = lengthen(bboxes, images)

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 30
        vid_filename = '_'.join(file_name.split('/'))
        out = cv.VideoWriter(vid_filename+'.avi', fourcc, fps, (img_w, img_h))

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
                

        flag = 0
        while True:            
            if (os.path.isdir(videofilepath)):
                image_idx +=1
                if image_idx == len(images):
                    break
                frame = cv.imread(str(images[image_idx]), cv.IMREAD_COLOR)
            elif (os.path.isfile(videofilepath)):
                ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()
            state = bboxes[image_idx]
            basic_exit = conf_history[image_idx]
            pred_exit = objconf_history[image_idx]
            
            font_color = (255, 0, 0)
            red_font = (0,0,255)

            if basic_exit<=0.5:
                exit_flag_b = True
                flag = 1
            else:
                exit_flag_b = False
                flag = 0
            if pred_exit<=neg_thres:
                exit_flag_a = True
                flag = 1
            else:
                exit_flag_a = False
                flag = 0

            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            if optional_box is not None:
                if (os.path.isdir(videofilepath)):
                    cv.rectangle(frame_disp, (optional_box[image_idx][0], optional_box[image_idx][1]), 
                                (optional_box[image_idx][2] + optional_box[image_idx][0], optional_box[image_idx][3] + optional_box[image_idx][1]),
                            (255, 255, 255), 5)
                elif (os.path.isfile(videofilepath)):
                    # cv.rectangle(frame_disp, (optional_box[image_idx][0], optional_box[image_idx][1]), 
                    #             (optional_box[image_idx][2] + optional_box[image_idx][0], optional_box[image_idx][3] + optional_box[image_idx][1]),
                    #         (255, 255, 255), 5)
                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)
                
            if exit_flag_b:
                cv.putText(frame_disp, 'basic score: %.4f'%(basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        red_font, 1)
            else:
                cv.putText(frame_disp, 'basic score: %.4f'%(basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)
                # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                #          (0, 255, 0), 5)
            if exit_flag_a:
                cv.putText(frame_disp, 'pred score: %.4f'%(pred_exit), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        red_font, 1)
            else:
                cv.putText(frame_disp, 'pred score: %.4f'%(pred_exit), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            cv.imwrite(f'test/datas/{vid_filename}_{image_idx}.png', frame_disp)
            out.write(frame_disp)
            

        # When everything done, release the capture
        if (os.path.isfile(videofilepath)):
            cap.release()
            out.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            # tracked_bb = np.array(output_boxes).astype(int)
            # bbox_file = '{}.txt'.format(base_results_path)
            # np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_video_robot(self, videofilepath, neg_thres=0.55, epsilon=0.005, avg_thres=20, version='cos', debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        import pyrealsense2 as rs
        import urx
        import time
        from basis import bs, bsr
        from Dependencies.urx_custom.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
        
        robot = urx.Robot('192.168.1.66')
        gripper = Robotiq_Two_Finger_Gripper(robot)

        params = self.get_parameters()
        #params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625, -0.8506844679461878, 0.4674954414367676],acc=0.3,vel=0.3,relative=False, threshold =1)
        print("robot ready")
        
        # Configure depth and color streams

        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
        self.frameSize = (640, 480)

        output_boxes = []
        conf_history = []
        # cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        # success, frame = cap.read()
        # cv.imshow(display_name, frame)



        def _build_init_info(box):
            return {'init_bbox': box}


        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            #convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            #colorwriter.write(color_image)
            #depthwriter.write(depth_colormap)
            
            cv.imshow('Stream', color_image)
            
            if cv.waitKey(1) == ord("p"):

                self.pipeline.stop()
                cv.destroyAllWindows()

                boxes = cv.selectROIs("first: pick, second: target", color_image)
                #first box: pick box, second box: place target
                optional_box = boxes[0].tolist()
                print(boxes[0], type(boxes[0]), "pick box type")
                print(boxes[1])
                #<class 'numpy.ndarray'>
                #x, y, w, h = cv.selectROI(color_image, False)
                #optional_box = [x, y, w, h]
                if optional_box is not None:
                    assert isinstance(optional_box, (list, tuple))
                    assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
                    tracker.initialize(color_image, _build_init_info(optional_box))
                    output_boxes.append(optional_box)
                cv.destroyAllWindows()
                break

        frame_num = 0
        gripper.gripper_action(0)
        self.pipeline.start(config)
        flag = 0
        cz_view=0
        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            #convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            frame = np.asanyarray(color_frame.get_data())

            if frames is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.odin_test_track(frame, epsilon, version)  # 0.005

            basic_exit = out['conf_score']
            pred_exit = out['objconf']
            pred_obj = out['predobj']
            conf_history.append(pred_exit)
            if len(conf_history) >= avg_thres:
                exitscore = moving_average(np.array(conf_history[-avg_thres:]), avg_thres)
            else:
                tmpthres = len(conf_history)
                exitscore = moving_average(np.array(conf_history), tmpthres)


            state = [int(s) for s in out['target_bbox']]
            initial_size = state[2]*state[3]

            output_boxes.append(state)

            # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            #              (0, 255, 0), 5)

            font_color = (255, 0, 0)
            red_font = (0, 0, 255)

            if basic_exit <= 0.5:
                exit_flag_b = True
            else:
                exit_flag_b = False

            if exitscore < neg_thres:
                #pred_exit = exitscore  # sigmoid(pred_exit)
                exit_flag_a = True
                #flag = 1
            else:
                #pred_exit = exitscore
                exit_flag_a = False
                #flag = 0

            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            if exit_flag_b:
                cv.putText(frame_disp, 'basic score: %.4f' % (basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        red_font, 1)
            else:
                cv.putText(frame_disp, 'basic score: %.4f' % (basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                #          (0, 255, 0), 5)
            if exit_flag_a:
                cv.putText(frame_disp, 'pred score: %.4f' % (pred_exit), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           red_font, 1)
                cv.putText(frame_disp, 'exit score: %.4f' % (exitscore), (20, 105), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           red_font, 1)
            else:
                cv.putText(frame_disp, 'pred score: %.4f' % (pred_exit), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'exit score: %.4f' % (exitscore), (20, 105), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            prev = []
            
            if frame_num % 10 ==0 and not exit_flag_a:
                cx = state[0]+state[2]/2
                icx = int(cx)
                cy = state[1]+state[3]/2
                icy = int(cy)
                print(icx, cx)
                print(icy, cy)
                print(depth_image.shape)

                print(depth_image[icy][icx])

                #cz = (depth_image[icy][icx]-588)*0.001*0.70 + 0.20
                cz = ((depth_image[icy][icx])*0.0011-0.16)
                #cx = cx*0.0011-0.266
                cx = cx*0.001-0.3
                #cy = cy*(-0.0011)+0.32
                cy = cy*(-0.001)+0.3

                dist = np.sqrt((cx**2+cy**2))

                if dist >= 0.03:
                    ax = cx/dist*0.03
                    ay = cy/dist*0.03
                else:
                    ax = cx
                    ay = cy


                max_depth = 0.27
                print(cx, cy, cz)
                # if flag ==1:
                #     x = input("if target found, press n, if not, press x")
                #     if x == 'n':
                #         flag = 2
                #     else:
                #         az = 0.03
                #         cz_view-=az
                #         #if cz_view + az >=max_depth:
                #         #    az=cz_view-max_depth
                #         #    cz_view=max_depth
                #         print(1, az, cz_view)
                #         robot.movel((0, -az,-az, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                #         #robot.movel(bsr3(0,0,-az),acc=0.2,vel=0.2,relative=True)
                #         frame_num +=1
                #         continue
                
                if cz < 0 :
                    print("z_value error")
                    print(2, cz_view)
                    robot.movel(bsr(ax, ay, 0, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                    #robot.movel(bsr3(ax,ay,0),acc=0.2,vel=0.2,relative=True)
                    prev.append((ax, ay))

                elif cz > 0.12 :
                    az = 0.05
                    cz_view+=az
                    #if cz_view <= max_depth and cz_view + az >=max_depth:
                    #    az = cz_view-max_depth
                    #    cz_view=max_depth
                    print(3, az, cz_view)
                    #robot.movel(bsr3(ax,ay,az),acc=0.2,vel=0.2,relative=True)
                    robot.movel(bsr(ax, ay, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    robot.movel((0, az,az, 0, 0, 0), acc=0.5, vel=0.5, relative=True)

                    prev.append((ax, ay))

                else:
                    az = cz
                    cz_view+=az+0.015
                    if cz_view + az >= max_depth:
                        az=cz_view-max_depth
                        cz_view=max_depth
                    print(4, az, cz_view)
                    #robot.movel(bsr(ax-0.036, ay+0.08, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    robot.movel(bsr(ax - 0.04, ay + 0.05, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    #robot.movel(bsr3(ax-0.031, ay+0.05, ax+0.015), acc=0.2, vel=0.2, relative=True)
                    #robot.movel((0, az / math.sqrt(2), az / math.sqrt(2), 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                    robot.movel((0, 0.03, 0.03, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    if flag == 0:
                        gripper.close_gripper()
                        flag = 1
                        tracker.initialize(color_image, _build_init_info(boxes[1].tolist()))
                        output_boxes.append(boxes[1].tolist())
                        time.sleep(2)
                        robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625,
                                     -0.8506844679461878, 0.4674954414367676], acc=0.3, vel=0.3, relative=False,
                                    threshold=1)

                    else:
                        gripper.open_gripper()
                        time.sleep(0.5)
                        robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625,
                                     -0.8506844679461878, 0.4674954414367676], acc=0.3, vel=0.3, relative=False,
                                    threshold=1)
                        x = input('type \"q then enter then ctrl+c\" to end')
                        
                    # x = input('type \"n\" to place the picked object on the target; otherwise type \"q then enter then ctrl+c\" to end')
                    # if x == 'n':
                    #     cz_view-=az+0.015
                    #     if cz_view>=max_depth:
                    #         az=cz_view-max_depth
                    #         cz_view=max_depth
                    #     robot.movel((0, -az-0.015,-az-0.015, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                    #     #robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625, -0.8506844679461878, 0.4674954414367676],acc=0.5,vel=0.5,relative=False, threshold =1)
                    #     print("robot ready", x)
                    #     flag = 1
                    #     tracker.initialize(color_image, _build_init_info(boxes[1].tolist()))
                    #     output_boxes.append(boxes[1].tolist())
                    # else:
                    #     gripper.open_gripper()
                    #     cz_view=0
                    
                

            frame_num += 1
            key = cv.waitKey(1)
            if key == ord('q'):
                self.pipeline.stop()
                cv.destroyAllWindows()
                break

            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()

            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)

            #     # cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        # self.pipeline.stop()
        # cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_stark_robot(self, videofilepath,  avg_thres=20, debug=None, visdom_info=None,  save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        import pyrealsense2 as rs
        import urx
        import time
        from basis import bs, bsr
        from Dependencies.urx_custom.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

        robot = urx.Robot('192.168.1.66')
        gripper = Robotiq_Two_Finger_Gripper(robot)

        params = self.get_parameters()
        # params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        robot.movej(
            [2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625, -0.8506844679461878,
             0.4674954414367676], acc=0.3, vel=0.3, relative=False, threshold=1)
        print("robot ready")

        # Configure depth and color streams

        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
        self.frameSize = (640, 480)

        output_boxes = []
        conf_history = []
        # cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name

        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        # success, frame = cap.read()
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # colorwriter.write(color_image)
            # depthwriter.write(depth_colormap)

            cv.imshow('Stream', color_image)

            if cv.waitKey(1) == ord("p"):

                self.pipeline.stop()
                cv.destroyAllWindows()

                boxes = cv.selectROIs("first: pick, second: target", color_image)
                # first box: pick box, second box: place target
                optional_box = boxes[0].tolist()
                print(boxes[0], type(boxes[0]), "pick box type")
                print(boxes[1])
                # <class 'numpy.ndarray'>
                # x, y, w, h = cv.selectROI(color_image, False)
                # optional_box = [x, y, w, h]
                if optional_box is not None:
                    assert isinstance(optional_box, (list, tuple))
                    assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
                    tracker.initialize(color_image, _build_init_info(optional_box))
                    output_boxes.append(optional_box)
                cv.destroyAllWindows()
                break

        frame_num = 0
        gripper.gripper_action(0)
        self.pipeline.start(config)
        flag = 0
        cz_view = 0
        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            frame = np.asanyarray(color_frame.get_data())

            if frames is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)  # 0.005

            basic_exit = out['conf_score']

            conf_history.append(basic_exit)
            if len(conf_history) >= avg_thres:
                exitscore = moving_average(np.array(conf_history[-avg_thres:]), avg_thres)
            else:
                tmpthres = len(conf_history)
                exitscore = moving_average(np.array(conf_history), tmpthres)

            state = [int(s) for s in out['target_bbox']]

            output_boxes.append(state)

            # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            #              (0, 255, 0), 5)

            font_color = (255, 0, 0)
            red_font = (0, 0, 255)

            if basic_exit <= 0.5:
                exit_flag_b = True
            else:
                exit_flag_b = False

            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            if exit_flag_b:
                cv.putText(frame_disp, 'basic score: %.4f' % (basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           red_font, 1)
                cv.putText(frame_disp, 'basic score: %.4f' % (exitscore), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           red_font, 1)
            else:
                cv.putText(frame_disp, 'basic score: %.4f' % (basic_exit), (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'basic score: %.4f' % (exitscore), (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)


            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            prev = []

            if frame_num % 5 == 0 and not exit_flag_b:
                cx = state[0] + state[2] / 2
                icx = int(cx)
                cy = state[1] + state[3] / 2
                icy = int(cy)
                print(icx, cx)
                print(icy, cy)
                print(depth_image.shape)

                print(depth_image[icy][icx])

                # cz = (depth_image[icy][icx]-588)*0.001*0.70 + 0.20
                cz = ((depth_image[icy][icx]) * 0.0011 - 0.16)
                # cx = cx*0.0011-0.266
                cx = cx * 0.001 - 0.3
                # cy = cy*(-0.0011)+0.32
                cy = cy * (-0.001) + 0.3

                dist = np.sqrt((cx ** 2 + cy ** 2))

                if dist >= 0.03:
                    ax = cx / dist * 0.03
                    ay = cy / dist * 0.03
                else:
                    ax = cx
                    ay = cy

                max_depth = 0.27
                print(cx, cy, cz)
                # if flag ==1:
                #     x = input("if target found, press n, if not, press x")
                #     if x == 'n':
                #         flag = 2
                #     else:
                #         az = 0.03
                #         cz_view-=az
                #         #if cz_view + az >=max_depth:
                #         #    az=cz_view-max_depth
                #         #    cz_view=max_depth
                #         print(1, az, cz_view)
                #         robot.movel((0, -az,-az, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                #         #robot.movel(bsr3(0,0,-az),acc=0.2,vel=0.2,relative=True)
                #         frame_num +=1
                #         continue

                if cz < 0:
                    print("z_value error")
                    print(2, cz_view)
                    robot.movel(bsr(ax, ay, 0, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                    # robot.movel(bsr3(ax,ay,0),acc=0.2,vel=0.2,relative=True)
                    prev.append((ax, ay))

                elif cz > 0.12:
                    az = 0.05
                    cz_view += az
                    # if cz_view <= max_depth and cz_view + az >=max_depth:
                    #    az = cz_view-max_depth
                    #    cz_view=max_depth
                    print(3, az, cz_view)
                    # robot.movel(bsr3(ax,ay,az),acc=0.2,vel=0.2,relative=True)
                    robot.movel(bsr(ax, ay, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    robot.movel((0, az, az, 0, 0, 0), acc=0.5, vel=0.5, relative=True)

                    prev.append((ax, ay))

                else:
                    az = cz
                    cz_view += az + 0.015
                    if cz_view + az >= max_depth:
                        az = cz_view - max_depth
                        cz_view = max_depth
                    print(4, az, cz_view)
                    #robot.movel(bsr(ax - 0.036, ay + 0.08, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    robot.movel(bsr(ax - 0.04, ay + 0.05, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    # robot.movel(bsr3(ax-0.031, ay+0.05, ax+0.015), acc=0.2, vel=0.2, relative=True)
                    # robot.movel((0, az / math.sqrt(2), az / math.sqrt(2), 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                    robot.movel((0, 0.03, 0.03, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
                    if flag == 0:
                        gripper.close_gripper()
                        flag = 1
                        tracker.initialize(color_image, _build_init_info(boxes[1].tolist()))
                        output_boxes.append(boxes[1].tolist())
                        time.sleep(2)
                        robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625,
                                     -0.8506844679461878, 0.4674954414367676], acc=0.3, vel=0.3, relative=False,
                                    threshold=1)

                    else:
                        gripper.open_gripper()
                        time.sleep(0.5)
                        robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625,
                                     -0.8506844679461878, 0.4674954414367676], acc=0.3, vel=0.3, relative=False,
                                    threshold=1)
                        x = input('type \"q then enter then ctrl+c\" to end')

                    # x = input('type \"n\" to place the picked object on the target; otherwise type \"q then enter then ctrl+c\" to end')
                    # if x == 'n':
                    #     cz_view-=az+0.015
                    #     if cz_view>=max_depth:
                    #         az=cz_view-max_depth
                    #         cz_view=max_depth
                    #     robot.movel((0, -az-0.015,-az-0.015, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
                    #     #robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625, -0.8506844679461878, 0.4674954414367676],acc=0.5,vel=0.5,relative=False, threshold =1)
                    #     print("robot ready", x)
                    #     flag = 1
                    #     tracker.initialize(color_image, _build_init_info(boxes[1].tolist()))
                    #     output_boxes.append(boxes[1].tolist())
                    # else:
                    #     gripper.open_gripper()
                    #     cz_view=0

            frame_num += 1
            key = cv.waitKey(1)
            if key == ord('q'):
                self.pipeline.stop()
                cv.destroyAllWindows()
                break

            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()

            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)

            #     # cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        # self.pipeline.stop()
        # cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.modelname, self.parameter_name, self.ckpt_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



