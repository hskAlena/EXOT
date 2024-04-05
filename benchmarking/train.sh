#!/bin/bash

# make data path
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

# tb1 robot 
# python tracking/train.py --script stark_st1 --config baseline_robot --save_dir . --mode single
# python tracking/train.py --script exot_st1 --config baseline_robot --save_dir . --mode multiple --nproc_per_node 2
# python tracking/train.py --script exot_st2 --config baseline_robot --save_dir . --mode single --script_prv stark_st1 --config_prv baseline_robot
# python tracking/train.py --script stark_st1 --config baseline_robot --save_dir . --mode single
# python tracking/train.py --script stark_st2 --config baseline_robot --save_dir . --mode single --script_prv stark_st1 --config_prv baseline_robot
# python tracking/train.py --script exot_st1 --config delta_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config delta_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot
# python tracking/train.py --script exot_st1 --config svdd_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config svdd_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot

# tb1 trek150 
#CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script exot_st1 --config baseline_trek150 --save_dir . --mode multiple --nproc_per_node 2
# python tracking/train.py --script exot_st2 --config baseline_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot
# python tracking/train.py --script stark_st1 --config baseline_robot --save_dir . --mode single
# python tracking/train.py --script stark_st2 --config baseline_robot --save_dir . --mode single --script_prv stark_st1 --config_prv baseline_robot
# python tracking/train.py --script exot_st1 --config delta_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config delta_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot
# python tracking/train.py --script exot_st1 --config svdd_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config svdd_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot

# tb1 mix 
python tracking/train.py --script exot_st1 --config baseline_mix --save_dir . --mode multiple --nproc_per_node 2
# python tracking/train.py --script exot_st2 --config baseline_mix --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_mix
# python tracking/train.py --script stark_st1 --config baseline_robot --save_dir . --mode single
# python tracking/train.py --script stark_st2 --config baseline_robot --save_dir . --mode single --script_prv stark_st1 --config_prv baseline_robot
# python tracking/train.py --script exot_st1 --config delta_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config delta_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot
# python tracking/train.py --script exot_st1 --config svdd_robot --save_dir . --mode single
# python tracking/train.py --script exot_st2 --config svdd_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot
