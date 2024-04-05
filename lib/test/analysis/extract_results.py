import os
import sys
import numpy as np
from lib.test.utils.load_text import load_text
import torch
import pickle
from tqdm import tqdm

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings


def calc_err_center(pred_bb, anno_bb, normalized=False):
    
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)
    print("Cal err center  ", pred_center, anno_center)
    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    return err_center

def cal_exit(pred_bb, anno_bb, valid):
    whole = pred_bb.shape[0]
    # predicted exit (True: non-exit(positive), False: exit(negative))
    pred_valid = ((pred_bb[:, 2:] > 0.0).sum(1) == 2)
    pred_true_num = torch.count_nonzero(pred_valid).item()
    # number of positive
    true_num = torch.count_nonzero(valid).item()
    # print("PRED valid", pred_valid)
    # print("valid", valid)
    # correct guess of exit (True: correct, False: incorrect)
    ans = pred_valid==valid
    # print("ANS", ans)
    # ratio of correct guess in non-exit (True: true positive, False: false positive+false negative+true negative)
    pos_ans = torch.logical_and(ans, pred_valid)
    true_pos = torch.logical_and(valid, pred_valid)
    false_pos = torch.logical_and(~valid, pred_valid)
    # ratio of correct guess in exit (True: true negative, False: false positive+false negative+true positive)
    true_neg = torch.logical_and(~valid, ~pred_valid)
    false_neg = torch.logical_and(valid, ~pred_valid)
    neg_ans = torch.logical_and(ans, ~pred_valid)
    # print("True positive", pos_ans)
    # print("True negative", neg_ans)
    posans_num = torch.count_nonzero(pos_ans).item()
    negans_num = torch.count_nonzero(neg_ans).item()
    fp_num = torch.count_nonzero(false_pos).item()
    tn_num = torch.count_nonzero(true_neg).item()
    positive_ratio = fp_num #-(true_num -posans_num) # number of incorrect positive
    # if whole-true_num ==0:
    #     negative_ratio = - (whole-pred_true_num)/whole
    # else:
    negative_ratio = tn_num # - (whole-true_num -negans_num) # number of incorrect negative
    return positive_ratio, negative_ratio


def calc_iou_overlap(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union


def calc_seq_err_robust(pred_bb, anno_bb, dataset, target_visible=None):
    pred_bb = pred_bb.clone()

    # Check if invalid values are present
    if torch.isnan(pred_bb).any():
        raise Exception('Error: Invalid results')
    # print(pred_bb[:, 2:])
    # print(dataset)
    if (pred_bb[:, 2:] < 0.0).any():
        # print(dataset)
        if dataset in ['robot', 'trek150']:
            idx = torch.nonzero(pred_bb[:, 2:] < 0.0, as_tuple=True)[0]
            # print(pred_bb[idx])
            # print(anno_bb[idx])
            pass
        else:
            raise Exception('Error: Invalid results')

    if torch.isnan(anno_bb).any():
        if dataset == 'uav':
            pass
        else:
            raise Exception('Warning: NaNs in annotation')

    if (pred_bb[:, 2:] == 0.0).any():
        print("Replace width height 0 to previous one")
        for i in range(1, pred_bb.shape[0]):
            if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i-1, :]

    print(pred_bb.shape, anno_bb.shape)
    if pred_bb.shape[0] != anno_bb.shape[0]:
        print("Predicted shape not match with groundtruth!")
        if dataset == 'lasot':
            if pred_bb.shape[0] > anno_bb.shape[0]:
                # For monkey-17, there is a mismatch for some trackers.
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                raise Exception('Mis-match in tracker prediction and GT lengths')
        else:
            # print('Warning: Mis-match in tracker prediction and GT lengths')
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                pad = torch.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4)).type_as(pred_bb)
                pred_bb = torch.cat((pred_bb, pad), dim=0)

    pred_bb[0, :] = anno_bb[0, :]

    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)
        #print("WHAT is valid", valid)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)
    fp, fn = cal_exit(pred_bb, anno_bb, valid)

    print(err_center) # [0 ~ 200]
    print("Norm", err_center_normalized)
    print("Overlap", err_overlap) # [0~1]
    print(fp, fn)

    # handle invalid anno cases
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0
    if dataset in ['robot', 'trek150']:
        err_center[~valid] = -1.0
        err_center_normalized[~valid] = -1.0
        err_overlap[~valid] = -1.0

    if dataset == 'lasot':
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')

    print(err_center) # [0 ~ 200]
    print("Norm", err_center_normalized)
    print("Overlap", err_overlap) # [0~1]
    print(fp, fn)

    return err_overlap, err_center, err_center_normalized, [fp, fn], valid


def extract_results(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05,
                    exclude_invalid_frames=False):
    settings = env_settings()
    eps = 1e-16

    #result_plot_path = os.path.join(settings.result_plot_path, report_name)
    result_plot_path = os.path.join(settings.result_plot_path, trackers[0].name, trackers[0].parameter_name, trackers[0].tmp_name, report_name)

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    avg_overlap_all = torch.zeros((len(dataset), len(trackers)), dtype=torch.float64)
    avg_fpfn_all = torch.zeros((len(dataset), len(trackers), 2), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((len(dataset), len(trackers), threshold_set_overlap.numel()),
                                                dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                               dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                                    dtype=torch.float32)
    main_tnr_auroc = torch.zeros((len(dataset), len(trackers), 2), dtype=torch.float32)
    sub_tnr_auroc = torch.zeros((len(dataset), len(trackers), 2), dtype=torch.float32)
    odin_tnr_auroc = torch.zeros((len(dataset), len(trackers), 2), dtype=torch.float32)

    valid_sequence = torch.ones(len(dataset), dtype=torch.uint8)

    def print_tnr_auroc(path, seq_id, trk_id, saver):
        with open(path, 'r') as f:
            main = f.read().splitlines()
            tnr = float(main[0].split('\t')[-1])
            auroc = float(main[1].split('\t')[-1])
            saver[seq_id, trk_id, 0] = tnr
            saver[seq_id, trk_id, 1] = auroc
        return saver

    for seq_id, seq in enumerate(tqdm(dataset)):
        # Load anno
        print(seq_id, seq)
        anno_bb = torch.tensor(seq.ground_truth_rect)
        target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
        for trk_id, trk in enumerate(trackers):
            # Load results
            base_results_path = '{}/{}'.format(trk.results_dir, seq.name)
            sub_results_path = '{}/{}_results_old'.format(trk.results_dir, seq.name)
            main_results_path = '{}/{}_results_new'.format(trk.results_dir, seq.name)
            odin_results_path = '{}/{}_results_old_odin'.format(trk.results_dir, seq.name)
            results_path = '{}.txt'.format(base_results_path)
            subresults_path = '{}.txt'.format(sub_results_path)
            mainresults_path = '{}.txt'.format(main_results_path)
            odin_results_path = '{}.txt'.format(odin_results_path)
            print(results_path, "RESULT path")

            if os.path.isfile(results_path):
                pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
                main_tnr_auroc = print_tnr_auroc(mainresults_path, seq_id, trk_id, main_tnr_auroc)
                sub_tnr_auroc = print_tnr_auroc(subresults_path, seq_id, trk_id, sub_tnr_auroc)                
            if os.path.isfile(odin_results_path):
                odin_tnr_auroc = print_tnr_auroc(odin_results_path, seq_id, trk_id, odin_tnr_auroc)
            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    print("ODIN results not computed")
                    #raise Exception('Result not found. {}'.format(results_path))

            # Calculate measures
            print(pred_bb)
            print(anno_bb)
            err_overlap, err_center, err_center_normalized, fpfn, valid_frame = calc_seq_err_robust(
                pred_bb, anno_bb, seq.dataset, target_visible)

            avg_overlap_all[seq_id, trk_id] = err_overlap[valid_frame].mean()
            avg_fpfn_all[seq_id, trk_id, 0] = fpfn[0]
            avg_fpfn_all[seq_id, trk_id, 1] = fpfn[1]

            if exclude_invalid_frames:
                seq_length = valid_frame.long().sum()
            else:
                seq_length = anno_bb.shape[0]

            if seq_length <= 0:
                raise Exception('Seq length zero')

            print("AVG overlap all", avg_overlap_all)
            print('Threshold overlap', threshold_set_overlap)
            print('Threshold center', threshold_set_center)
            ave_success_rate_plot_overlap[seq_id, trk_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center[seq_id, trk_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center_norm[seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length
            print('avg success rat plot overla', ave_success_rate_plot_overlap)
            print('avg success rate plot centr', ave_success_rate_plot_center)
        # exit(0)
        #input('wait.... ')
    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    # Prepare dictionary for saving data
    seq_names = [s.name for s in dataset]
    tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                     for t in trackers]

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'main_tnr_auroc': main_tnr_auroc.tolist(),
                 'sub_tnr_auroc': sub_tnr_auroc.tolist(),
                 'odin_tnr_auroc': odin_tnr_auroc.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'avg_fpfn_all': avg_fpfn_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data
