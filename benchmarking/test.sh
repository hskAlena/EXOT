#!/bin/bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
export CUDA_VISIBLE_DEVICES=0

# python tracking/test.py exotst_tracker baseline_mix_lowdim --dataset robot_test --run_fn run_epsilon \
#                           --modelname exot_merge --ckpt_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=27.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# python tracking/analysis_results.py --tracker_param baseline_mix_lowdim --dataset robot_test --name exotst_tracker \
#                                       --modelname exot_merge --ckpt_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=27.pth.tar

####################################################3
# python tracking/test.py exotst_tracker baseline_mix_sim --dataset robot_test --run_fn run_epsilon \
#                           --modelname exot_st2 --ckpt_name baseline_mix_sim/EXOTST_ep0060.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# python tracking/analysis_results.py --tracker_param baseline_mix_sim --dataset robot_test --name exotst_tracker \
#                                       --modelname exot_st2 --ckpt_name baseline_mix_sim/EXOTST_ep0060.pth.tar 

# python tracking/test.py exotst_tracker baseline_mix_enc --dataset robot_test --run_fn run_epsilon \
#                          --modelname exot_st2 --ckpt_name baseline_mix_enc/EXOTST_ep0040.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 #--debug 1
# python tracking/analysis_results.py --tracker_param baseline_mix_enc --dataset robot_test --name exotst_tracker \
#                                       --modelname exot_st2 --ckpt_name baseline_mix_enc/EXOTST_ep0040.pth.tar 

##########################################################
# python tracking/test.py exotst_tracker baseline_mix_hs --dataset robot_test --run_fn run_epsilon \
#                           --modelname exot_st2 --ckpt_name baseline_mix_hs/EXOTST1_ep0060.pth.tar --epsilon 0.00125 --threads 0 --num_gpus 1 --debug 1
# python tracking/analysis_results.py --tracker_param baseline_mix_hs --dataset robot_test --name exotst_tracker \
#                                       --modelname exot_st2 --ckpt_name baseline_mix_hs/EXOTST1_ep0060.pth.tar 

################################################################
