#!/bin/bash

# make data path
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
export CUDA_VISIBLE_DEVICES=1
# tb1 mix 

# chocolate

# python pl_prac.py --script exot_st1 --config baseline_mix --save_dir . --mode single \
#                   --resume True --resume_name baseline_mix_fold_0_5/EXOTST_epoch=99-v1.pth.tar

# python pl_prac.py --script exot_st2 --config baseline_mix_enc --save_dir . \
#                   --mode single --script_prv exot_st1 --config_prv baseline_mix_lowdim \
                    --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=113.pth.tar

# python pl_prac.py --script exot_st2 --config baseline_mix_sim --save_dir . \
#                   --mode single --script_prv exot_st1 --config_prv baseline_mix_lowdim \
                    --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=113.pth.tar

# python pl_prac.py --script exot_st2 --config baseline_mix_lowdim --save_dir . \
#                   --mode single --script_prv exot_st1 --config_prv baseline_mix_lowdim \
                    --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=113.pth.tar

# python pl_prac.py --script exot_merge --config baseline_mix_enc --save_dir . --mode single \
#                   --resume True --resume_name baseline_mix_enc_fold_0_5/EXOTST_epoch=40.pth.tar

# python pl_prac.py --script exot_merge --config baseline_mix_lowdim --save_dir . --mode single \
#                   --resume True --resume_name baseline_mix_hs_fold_0_5/EXOTST_epoch=35.pth.tar

# python pl_prac.py --script exot_st2 --config baseline_mix --save_dir . --mode single --script_prv exot_st1 \
#                   --config_prv baseline_mix --st1_name baseline_mix_fold_0_5/EXOTST_epoch=0.pth.tar


# ella


# python pl_prac.py --script exot_st2 --config baseline_robot_hscos --save_dir . \
#                   --mode single --script_prv exot_st1 --config_prv baseline_robot \
                    --st1_name baseline_robot/EXOTST_epoch=11.pth.tar \
# python pl_prac.py --script exot_st2 --config cos_mix_lowdim --save_dir . \
#                   --mode single --script_prv stark_st1 --config_prv baseline_mix \
                    --st1_name baseline_mix_fold_0_5/STARKST_epoch=250-v1.pth.tar

CUDA_VISIBLE_DEVICES=1 python pl_prac.py --script exot_st2 --config cos_mix_lowdim --save_dir . \
--mode single --script_prv exot_st1 --config_prv cos_mix_lowdim --st1_name cos_mix_lowdim_fold_0_5/EXOTST_epoch=250-v1.pth.tar


# CUDA_VISIBLE_DEVICES=1 python pl_prac.py --script exot_st1 --config cos_mix_lowdim --save_dir . \
# --mode single --resume True --resume_name cos_mix_lowdim_fold_0_5/EXOTST_epoch=237.pth.tar
#CUDA_VISIBLE_DEVICES=1 python pl_prac.py --script exot_st2 --config cos_mix_lowdim --save_dir . \
#--mode single --script_prv stark_st1 --config_prv baseline_mix_nocls --st1_name baseline_mix_nocls_fold_0_5/STARKST_epoch=250.pth.tar
# --resume True --resume_name EXOTST_epoch=250/EXOTST_epoch=54.pth.tar

# python pl_prac.py --script exot_st2 --config baseline_mix_enc --save_dir . \
#                   --mode single --script_prv exot_st1 --config_prv baseline_mix_lowdim \
#                   --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=113.pth.tar \
#                   --resume True --resume_name baseline_mix_enc_fold_0_5/EXOTST_epoch=39.pth.tar
# python pl_prac.py --script exot_st2 --config baseline_mix_lowdim --save_dir . \
#                   --mode single --script_prv exot_st1 --config_prv baseline_mix_lowdim --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=113.pth.tar \
#                   --resume True --resume_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=37.pth.tar
# python pl_prac.py --script exot_st2 --config baseline_mix_lowdim --save_dir . \
#                   --mode single --script_prv exot_st1 --config_prv baseline_mix_lowdim --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=113.pth.tar \
#                   --resume True --resume_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=37.pth.tar
#python pl_prac.py --script exot_st1 --config baseline_mix_enc --save_dir . --mode single --resume True \
# --resume_name baseline_mix_enc_fold_0_5/EXOTST_epoch=20.pth.tar
#--resume False --resume_name baseline_mix_fold_0_5/EXOTST_epoch=99.pth.tar
#CUDA_VISIBLE_DEVICES=2 python pl_prac.py --script exot_st2 --config baseline_mix_lowdim --save_dir . --mode single \
# --script_prv exot_st1 --config_prv baseline_mix_lowdim --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=14.pth.tar

# lucy

# python pl_prac.py --script exot_st1 --config baseline_mix --save_dir . --mode single --resume True --resume_name baseline_mix_fold_0_5/EXOTST_epoch=99-v1.pth.tar
#CUDA_VISIBLE_DEVICES=1 python pl_prac.py --script exot_merge --config baseline_mix_enc --save_dir . --mode single
#--resume True --resume_name baseline_mix_fold_0_5/EXOTST_epoch=99-v1.pth.tar
# python pl_prac.py --script exot_st2 --config baseline_mix --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_mix --st1_name baseline_mix_fold_0_5/EXOTST_epoch=0.pth.tar
#python pl_prac.py --script exot_merge --config baseline_mix_enc --save_dir . --mode multiple --nproc_per_node 2 
#--resume True --resume_name baseline_mix_enc_fold_0_5/EXOTST_epoch=20.pth.tar
#CUDA_VISIBLE_DEVICES=1 python pl_prac.py --script exot_st2 --config baseline_mix_enc --save_dir . --mode single --script_prv exot_st1 \
#--config_prv baseline_mix_lowdim --st1_name baseline_mix_lowdim_fold_0_5/EXOTST_epoch=113.pth.tar


# python pl_prac.py --script stark_st1 --config baseline_mix --save_dir . --mode single 
# python pl_prac.py --script stark_st2 --config baseline_mix --save_dir . --mode single --script_prv stark_st1 --config_prv baseline_mix
# python tracking/train.py --script exot_st1 --config delta_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config delta_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot
# python tracking/train.py --script exot_st1 --config svdd_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config svdd_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot
