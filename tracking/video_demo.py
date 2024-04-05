import os
import sys
import argparse
import numpy as np
import pandas

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker
from pl_prac import Settings


# def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False):
#     """Run the tracker on your webcam.
#     args:
#         tracker_name: Name of tracking method.
#         tracker_param: Name of parameter file.
#         debug: Debug level.
#     """
#     tracker = Tracker(tracker_name, tracker_param, "video")
#     tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)


# def main():
#     parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
#     parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
#     parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
#     parser.add_argument('videofile', type=str, help='path to a video file.')
#     parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
#     parser.add_argument('--debug', type=int, default=0, help='Debug level.')
#     parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
#     parser.set_defaults(save_results=False)

#     args = parser.parse_args()

#     run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results)

def run_video(tracker_name, tracker_param, modelname, ckpt_name, videofile='', optional_box=None, save_dir=None,
              track_format='run_video', debug=None,
              save_results=False, neg_thres=0.55, epsilon=0.005, version='cos', avg_thres=20):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video", modelname=modelname, ckpt_name=ckpt_name)
    if track_format == 'run_video':
        tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)
    # elif track_format == 'run_video_raw':
    #     tracker.run_video_raw(videofilepath=save_dir, debug=debug, save_results=save_results)
    elif track_format == 'run_video_robot':
        tracker.run_video_robot(videofilepath=save_dir, neg_thres=neg_thres, epsilon=epsilon, version=version, avg_thres=avg_thres,  debug=debug, save_results=save_results)
    elif track_format == 'run_stark_robot':
        tracker.run_stark_robot(videofilepath=save_dir, avg_thres=avg_thres,  debug=debug, save_results=save_results)
    elif track_format == 'run_video_annot_odin':
        with open(save_dir, 'r') as f:
            folder_list = f.read().splitlines()
        for i in range(len(folder_list)):
            if 'TREK-150' in save_dir:
                annot_file = ('/').join([('/').join(save_dir.split('/')[:-1]), folder_list[i].split('-')[0], folder_list[i], 'groundtruth_rect.txt'])
                videofile = ('/').join([('/').join(save_dir.split('/')[:-1]), folder_list[i].split('-')[0], folder_list[i], 'img'])
                optional_box = pandas.read_csv(annot_file, delimiter=',', header=None, dtype=np.int, na_filter=False, low_memory=False).values
            elif 'robot' in save_dir:
                annot_file = ('/').join([('/').join(save_dir.split('/')[:-1]), folder_list[i], 'groundtruth.txt'])
                videofile = ('/').join(save_dir.split('/')[:-1])+'/'+folder_list[i]+'.avi'
                if os.path.isfile(annot_file):
                    optional_box = pandas.read_csv(annot_file, delimiter='\t', header=None, dtype=np.int, na_filter=False, low_memory=False).values
            else:
                optional_box = None
            tracker.run_video_annot_odin(videofilepath=videofile, neg_thres=neg_thres, epsilon=epsilon, version=version, avg_thres=avg_thres, optional_box=optional_box, debug=debug, save_results=save_results)
    elif track_format == 'run_video_annot_stark':
        with open(save_dir, 'r') as f:
            folder_list = f.read().splitlines()
        for i in range(len(folder_list)):
            if 'TREK-150' in save_dir:
                annot_file = ('/').join([('/').join(save_dir.split('/')[:-1]), folder_list[i].split('-')[0], folder_list[i], 'groundtruth_rect.txt'])
                videofile = ('/').join([('/').join(save_dir.split('/')[:-1]), folder_list[i].split('-')[0], folder_list[i], 'img'])
                optional_box = pandas.read_csv(annot_file, delimiter=',', header=None, dtype=np.int, na_filter=False, low_memory=False).values
            elif 'robot' in save_dir:
                annot_file = ('/').join([('/').join(save_dir.split('/')[:-1]), folder_list[i], 'groundtruth.txt'])
                videofile = ('/').join(save_dir.split('/')[:-1])+'/'+folder_list[i]+'.avi'
                if os.path.isfile(annot_file):
                    optional_box = pandas.read_csv(annot_file, delimiter='\t', header=None, dtype=np.int, na_filter=False, low_memory=False).values
            else:
                optional_box = None
            tracker.run_video_annot_stark(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)
    elif track_format == 'run_video_save_jpg':
        with open(save_dir, 'r') as f:
            folder_list = f.read().splitlines()
        for i in range(len(folder_list)):
            if 'robot' in save_dir:
                # annot_file = ('/').join([('/').join(save_dir.split('/')[:-1]), folder_list[i], 'groundtruth.txt'])
                videofile = ('/').join(save_dir.split('/')[:-1])+'/'+folder_list[i]#+'.avi'
                optional_box = None
            tracker.run_video_save_jpg(videofilepath=videofile, neg_thres=neg_thres, epsilon=epsilon, version=version, avg_thres=avg_thres, optional_box=optional_box, debug=debug, save_results=save_results)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--video_file', type=str, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--save_dir', type=str, default=None, help='save directory for run_video_raw')
    parser.add_argument('--track_format', type=str, default='run_video', help='annotate, evaluate video, evaluate video stream')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.add_argument('--neg_thres', type=float, default=0.55, help='threshold for odin exit')
    parser.add_argument('--epsilon', type=float, default=0.005, help='epsilon for odin prediction')
    parser.add_argument('--version', type=str, default='cos', help='ori, h, g')
    parser.add_argument('--avg_thres', type=int, default=20, help='length for moving average in exit')
    parser.add_argument('--modelname', type=str, default=None, help='save directory for run_video_raw')
    parser.add_argument('--ckpt_name', type=str, default=None, help='save directory for run_video_raw')
    parser.set_defaults(save_results=True)

    args = parser.parse_args()

    # tracker_params = {}
    # for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) != None, args.__dir__())):
    #     tracker_params[param.split('__')[1]] = getattr(args, param)
    # print(tracker_params)

    run_video(args.tracker_name, args.tracker_param, args.modelname, args.ckpt_name, args.video_file, args.optional_box, args.save_dir, args.track_format, args.debug,
              args.save_results, args.neg_thres, args.epsilon, args.version, args.avg_thres)


if __name__ == '__main__':
    main()
