#!/bin/bash
#python tracking/test.py stark_st baseline_trek150 --dataset trek150_test --threads 4
#python tracking/test.py stark_st baseline_robot --dataset robot_test --threads 4
#python tracking/analysis_results.py
# CUDA_VISIBLE_DEVICES=2 python cal_eps.py exot_st baseline_mix --dataset robot_test --main main --threads 0 --num_gpus 1
#CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix --dataset robot_test --name exotst_tracker
#CUDA_VISIBLE_DEVICES=0 python tracking/test.py exotst_tracker baseline_mix_lowdim --dataset robot_test --run_fn run_epsilon \
#--modelname exot_st2 --ckpt_name EXOTST_epoch=%d-v3-momentum.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1

python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py stark_st baseline_got10k_only --dataset trek150_test --run_fn run_seq \
# --modelname stark_st2 --ckpt_name STARKST_ep0050.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_got10k_only --dataset trek150_test --name stark_st \
# --modelname stark_st2 --ckpt_name STARKST_ep0050.pth.tar

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py stark_st baseline_mix --dataset trek150_test --run_fn run_seq \
# --modelname stark_st2 --ckpt_name STARKST_epoch=49-v1.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix --dataset trek150_test --name stark_st \
# --modelname stark_st2 --ckpt_name STARKST_epoch=49-v1.pth.tar

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker baseline_mix_lowdim --dataset trek150_test --run_fn run_epsilon \
# --modelname exot_merge --ckpt_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=27.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix_lowdim --dataset trek150_test --name exotst_tracker \
# --modelname exot_merge --ckpt_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=27.pth.tar

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker baseline_mix_sim --dataset trek150_test --run_fn run_epsilon \
# --modelname exot_merge --ckpt_name baseline_mix_sim_fold_0_5/EXOTST_epoch=25.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix_sim --dataset trek150_test --name exotst_tracker \
# --modelname exot_merge --ckpt_name baseline_mix_sim_fold_0_5/EXOTST_epoch=25.pth.tar
# #CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix --dataset trek150_test --name stark_st
# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker baseline_mix_lowdim --dataset trek150_test --run_fn run_epsilon \
# --modelname exot_st2 --ckpt_name baseline_mix_lowdim/EXOTST_ep0060.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix_lowdim --dataset trek150_test --name exotst_tracker \
# --modelname exot_st2 --ckpt_name baseline_mix_lowdim/EXOTST_ep0060.pth.tar 

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker baseline_mix_sim --dataset trek150_test --run_fn run_epsilon \
# --modelname exot_st2 --ckpt_name baseline_mix_sim/EXOTST_ep0060.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix_sim --dataset trek150_test --name exotst_tracker \
# --modelname exot_st2 --ckpt_name baseline_mix_sim/EXOTST_ep0060.pth.tar 

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker cos_mix_lowdim --dataset robot_test --run_fn run_epsilon \
#  --modelname exot_st2 --ckpt_name cos_mix_lowdim_fold_0_5/baseline_mix_fold_0_5/STARKST_epoch=250/EXOTST_epoch=99.pth.tar \
#  --epsilon 0.00125 --threads 0 --version h --num_gpus 1 
#CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param cos_mix_lowdim --dataset robot_test --name exotst_tracker \
#--modelname exot_st2 --ckpt_name cos_mix_lowdim_fold_0_5/baseline_mix_fold_0_5/STARKST_epoch=250/EXOTST_epoch=99.pth.tar 

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker cos_mix_lowdim --dataset trek150_test --run_fn run_epsilon \
#   --modelname exot_st2 --ckpt_name cos_mix_lowdim_tmp/EXOTST1_ep0050.pth.tar \
#   --epsilon 0.00125 --threads 0 --version h --num_gpus 1 #--debug 1
CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results_exot.py --tracker_param cos_mix_lowdim --dataset trek150_test --name exotst_tracker \
--modelname exot_st2 --ckpt_name cos_mix_lowdim_tmp/EXOTST1_ep0050.pth.tar

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker obj_mix_lowdim --dataset trek150_test --run_fn run_epsilon \
#  --modelname exot_st2 --ckpt_name obj_mix_lowdim_fold_0_5/baseline_mix_fold_0_5/STARKST_epoch=250/EXOTST_epoch=99.pth.tar \
#  --epsilon 0.00125 --threads 0 --version h --num_gpus 1

# CUDA_VISIBLE_DEVICES=1 python tracking/test.py exotst_tracker baseline_mix_hs --dataset trek150_test --run_fn run_epsilon \
# --modelname exot_st2 --ckpt_name baseline_mix_hs/EXOTST_ep0060.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=1 python tracking/analysis_results.py --tracker_param baseline_mix_hs --dataset trek150_test --name exotst_tracker \

# CUDA_VISIBLE_DEVICES=2 python tracking/test.py exotst_tracker baseline_mix_enc --dataset trek150_test --run_fn run_epsilon \
#  --modelname exot_merge --ckpt_name baseline_mix_enc/EXOTST_ep0026.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# CUDA_VISIBLE_DEVICES=2 python tracking/analysis_results.py --tracker_param baseline_mix_enc --dataset trek150_test --name exotst_tracker \
#  --modelname exot_merge --ckpt_name baseline_mix_enc/EXOTST_ep0026.pth.tar
