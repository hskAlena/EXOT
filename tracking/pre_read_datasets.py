import _init_paths
import multiprocessing as mp
import argparse
import os
from lib.utils.lmdb_utils import decode_str
import time
import json


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--data_dir', type=str, help='directory where lmdb data is located')
    parser.add_argument('--dataset_str', type=str, help="which datasets to use")
    args = parser.parse_args()

    return args


def get_trknet_dict(trknet_dir):
    with open(os.path.join(trknet_dir, "seq_list.json"), "r") as f:
        seq_list = json.loads(f.read())
    res_dict = {}
    set_idx_pre = -1
    for set_idx, seq_name in seq_list:
        if set_idx != set_idx_pre:
            res_dict[set_idx] = "anno/%s.txt" % seq_name
            set_idx_pre = set_idx
    return res_dict


def target(lmdb_dir, key_name):
    _ = decode_str(lmdb_dir, key_name)


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    dataset_str = args.dataset_str
    
    print("Ready to pre load datasets")
    start = time.time()
    ps = []
    # deal with trackingnet
    if 't' in dataset_str:
        trknet_dict = get_trknet_dict(os.path.join(data_dir, "trackingnet_lmdb"))
        for set_idx, seq_path in trknet_dict.items():
            lmdb_dir = os.path.join(data_dir, "trackingnet_lmdb", "TRAIN_%d_lmdb" % set_idx)
            p = mp.Process(target=target, args=(lmdb_dir, seq_path))
            print("add %s %s to job queue" % (lmdb_dir, seq_path))
            ps.append(p)
    for p in ps:
        p.start()
    for p in ps:
        p.join()

    print("Pre read over")
    end = time.time()
    hour = (end - start) / 3600
    print("it takes %.2f hours to pre-read data" % hour)
