# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---



# +
train_list = 'data/robot-data/data_RGB/sequence_list.txt'
val_list = 'data/robot-data/data_RGB/test_seq.txt'
test_list = 'data/robot-data/data_RGB/test_seq2.txt'

# train_list = 'data/TREK-150/origin_seq.txt'
# val_list = 'data/TREK-150/test_seq.txt'
# test_list = 'data/TREK-150/test_seq2.txt'

# +
import glob

def read_txt_trek(lists):
    with open(lists, 'r') as f:
        train_folders =f.read().splitlines()
    train_txts = []
    for i in range(len(train_folders)):
        foldername = train_folders[i].split('-')[0]
        train_txts.append('data/TREK-150/'+foldername+'/'+train_folders[i]+'/groundtruth_rect.txt')
    
    train_pos = 0
    train_neg = 0
    train_lists = {}

    for j in range(len(train_txts)):      
        with open(train_txts[j], 'r') as f:
            whole_ = f.read().splitlines()
        name = '/'.join(train_txts[j].split('/')[-3:-1])
        train_lists[name+'_pos'] = 0
        train_lists[name+'_neg'] = 0

        for k in range(len(whole_)):
            num = int(whole_[k].split(',')[0])
            if num>=0:
                train_pos +=1
                train_lists[name+'_pos'] +=1
            else:
                train_neg +=1
                train_lists[name+'_neg'] +=1
    return train_pos, train_neg, train_lists

def read_noun_trek(lists):
    with open(lists, 'r') as f:
        train_folders =f.read().splitlines()
    train_txts = []
    for i in range(len(train_folders)):
        foldername = train_folders[i].split('-')[0]
        train_txts.append('data/TREK-150/'+foldername+'/'+train_folders[i]+'/action_target.txt')
    
    train_lists = {}

    for j in range(len(train_txts)):      
        with open(train_txts[j], 'r') as f:
            whole_ = f.read().splitlines()
        name = '/'.join(train_txts[j].split('/')[-3:-1])

        target_noun = int(whole_[2])
        if target_noun in train_lists:
            train_lists[target_noun] += 1
        else:
            train_lists[target_noun] = 1

    return train_lists

def read_txt(lists):
    with open(lists, 'r') as f:
        train_folders =f.read().splitlines()
    train_txts = []
    for i in range(len(train_folders)):
        train_txts.append('data/robot-data/data_RGB/'+train_folders[i]+'/groundtruth.txt')
    
    train_pos = 0
    train_neg = 0
    train_lists = {}

    for j in range(len(train_txts)):      
        with open(train_txts[j], 'r') as f:
            whole_ = f.read().splitlines()
        name = '/'.join(train_txts[j].split('/')[-3:-1])
        train_lists[name+'_pos'] = 0
        train_lists[name+'_neg'] = 0

        for k in range(len(whole_)):
            num = int(whole_[k].split('\t')[0])
            if num>=0:
                train_pos +=1
                train_lists[name+'_pos'] +=1
            else:
                train_neg +=1
                train_lists[name+'_neg'] +=1
    return train_pos, train_neg, train_lists



# +
#tp, tn, td = read_txt_trek(train_list)
vp, vn, vd = read_txt(val_list)
tp, tn, td = read_txt(test_list)
#noun = read_noun_trek(train_list)

#print(tp, tn, td)
# print(noun)
# print('\n\n')
# print(vp, vn, vd)
# print('\n\n')
# print(ttp, ttn, ttd)
# print('\n\n')

# print(ttp)
# -

sortednoun = list(noun.keys())
sortednoun.sort()
print(sortednoun)
print(len(sortednoun))


# +
import csv

header = ['noun_id','before_id']
with open('convert.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i, j in enumerate(sortednoun):
        writer.writerow([i, j])

# +
negatives = []
positives = []

human_negative = []
human_positive = []
for key, value in td.items():
    if 'neg' in key and value >0:
        #print(td[key[:-3]+'pos'])
        print(key, td[key[:-3]+'pos'], value)
        if 'human' in key:
            human_negative.append(value)
            human_positive.append(td[key[:-3]+'pos'])
        else:
            negatives.append(value)
            positives.append(td[key[:-3]+'pos'])
        
for key, value in vd.items():
    if 'neg' in key and value >0:
        #print(td[key[:-3]+'pos'])
        print(key, vd[key[:-3]+'pos'], value)
        if 'human' in key:
            human_negative.append(value)
            human_positive.append(vd[key[:-3]+'pos'])
        else:
            negatives.append(value)
            positives.append(vd[key[:-3]+'pos'])

# +
import numpy as np
negatives =np.array(negatives)
positives =np.array(positives)
eal = np.mean(negatives)
avl = np.mean(negatives + positives)
mieal = np.min(negatives)
mael = np.max(negatives)

print(eal, avl, mieal, mael)

# +
negatives =np.array(human_negative)
positives =np.array(human_positive)
eal = np.mean(negatives)
avl = np.mean(negatives + positives)
mieal = np.min(negatives)
mael = np.max(negatives)

print(eal, avl, mieal, mael)

# +
from cProfile import label
import glob
from sre_constants import SRE_INFO_CHARSET
# file_path = glob.glob('test/tracking_results/stark_st/*/*/*/*_conf_score.txt')
# gt_path = glob.glob('test/tracking_results/stark_st/*/*/*/*_visgt.txt')

file_path = glob.glob('test/tracking_results/stark_st/baseline_mix/STARKST_epoch=49-v1/auto/TennisBall-8-23-19-10-44_conf_score.txt')
gt_path = glob.glob('test/tracking_results/stark_st/baseline_mix/STARKST_epoch=49-v1/auto/TennisBall-8-23-19-10-44_visgt.txt')
file_path.sort()
gt_path.sort()
# print(file_path)
import numpy as np
import matplotlib.pyplot as plt

for i in range(len(file_path)):
    if 'mix' in file_path[i]:
        print(file_path[i], gt_path[i])
        with open(file_path[i], 'r') as f:
            b = np.loadtxt(f)
        with open(gt_path[i], 'r') as f:
            gt = np.loadtxt(f)
        guide = (b>=0.5).astype(float)
        
        # plt.plot(guide, label="guide", color="forestgreen", linewidth = 5)
        # plt.plot(gt, label = "gt", color = "palevioletred", linewidth = 5, linestyle='--')
        plt.plot(guide, label="guide", color="darkblue", linewidth = 5)
        plt.plot(gt, label = "gt", color = "goldenrod", linewidth = 4, linestyle='--')

        plt.xlabel('Frame', labelpad=10, font = 'serif', size='20')
        plt.ylabel('Exit label', labelpad=10, font = 'serif', size='20')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--')
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        
        break
    # print(b.shape, b)

# +
import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

data = np.array([10,5,8,9,15,22,26,11,15,16,18,7])

#print(moving_average(data,4))

file_path = glob.glob('test/tracking_results/exotst_tracker/baseline_mix_sim/EXOTST_ep0060/auto/TennisBall-8-23-19-10-44_conf_score.txt')
obj_path = glob.glob('test/tracking_results/exotst_tracker/baseline_mix_sim/EXOTST_ep0060/auto/TennisBall-8-23-19-10-44_objconf.txt')
gt_path = glob.glob('test/tracking_results/exotst_tracker/baseline_mix_sim/EXOTST_ep0060/auto/TennisBall-8-23-19-10-44_visgt.txt')
file_path.sort()
gt_path.sort()
# print(file_path)
import numpy as np
import matplotlib.pyplot as plt

for i in range(len(file_path)):
    if 'mix' in file_path[i]:
        print(file_path[i], gt_path[i])
        with open(file_path[i], 'r') as f:
            b = np.loadtxt(f)
        with open(gt_path[i], 'r') as f:
            gt = np.loadtxt(f)
        with open(obj_path[i], 'r') as f:
            obj = np.loadtxt(f)
        guide = (b>=0.5).astype(float)
        
        objg = moving_average(obj, 20)
        objg = (objg>=0.54).astype(float)
        if len(objg) < len(gt):
            num = len(gt)-len(objg)
            objg = np.pad(objg, (0, num), 'edge')
        # plt.plot(guide)
        # plt.plot(objg,label="objg", color="forestgreen", linewidth = 5)
        # plt.plot(gt, label = "gt", color = "palevioletred", linewidth = 3, linestyle='--')
        plt.plot(objg,label="objg", color="darkblue", linewidth = 5)
        plt.plot(gt, label = "gt", color = "goldenrod", linewidth = 4, linestyle='--')
        #plt.plot(guide, label="guide", color="CornflowerBlue", linewidth = 5)
        #plt.plot(gt, label = "gt", color = "pink", linewidth = 5)
        
        plt.xlabel('Frame', labelpad=15, font = 'serif', size='20')
        plt.ylabel('Exit label', labelpad=15, font = 'serif', size='20')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--')
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        # plt.show()
    
        break

    plt.savefig('fig5-2.png', bbox_inches='tight', dpi=300)
    # print(b.shape, b)


# -


