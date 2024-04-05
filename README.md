# EXOT
The official implementation of the **ICRA2023** paper [**EXOT: Exit-aware Object Tracker for Safe Robotic Manipulation of Moving Object**](https://arxiv.org/abs/2306.05262)

## Install the environment
Use the Anaconda
```
conda create -n exot python=3.8
conda activate exot
bash install_exot.sh
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${STARK_ROOT}
    -- data
        -- TREK-150
            |-- P03
            |-- P05
            |-- P06
            ...
        -- robot-data
            -- data_RGB
            	|-- auto
            	|-- human
   ```
## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train EXOT
Training with multiple GPUs using DDP
```
python tracking/train.py --script exot_st1 --config baseline_robot --save_dir . --mode multiple --nproc_per_node 8  # EXOT Stage1
python tracking/train.py --script exot_st2 --config baseline_robot --save_dir . --mode multiple --nproc_per_node 8 --script_prv exot_st1 --config_prv baseline_robot  # EXOT Stage2
```
(Optionally) Debugging training with a single GPU
```
python tracking/train.py --script exot_st1 --config baseline_robot --save_dir . --mode single  # EXOT Stage1
python tracking/train.py --script exot_st2 --config baseline_robot --save_dir . --mode single --script_prv exot_st1 --config_prv baseline_robot  # EXOT Stage2
```
## Test and evaluate EXOT on benchmarks

- LaSOT
```
python tracking/test.py stark_st baseline --dataset lasot --threads 32
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py stark_st baseline_got10k_only --dataset got10k_test --threads 32
python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_only
```
- TrackingNet
```
python tracking/test.py stark_st baseline --dataset trackingnet --threads 32
python lib/test/utils/transform_trackingnet.py --tracker_name stark_st --cfg_name baseline
```
- VOT2020  
Before evaluating "STARK+AR" on VOT2020, please install some extra packages following [external/AR/README.md](external/AR/README.md)
```
cd external/vot20/<workspace_dir>
export PYTHONPATH=<path to the stark project>:$PYTHONPATH
bash exp.sh
```
- VOT2020-LT
```
cd external/vot20_lt/<workspace_dir>
export PYTHONPATH=<path to the stark project>:$PYTHONPATH
bash exp.sh
```
## Test FLOPs, Params, and Speed
```
# Profiling STARK-S50 model
python tracking/profile_model.py --script stark_s --config baseline
# Profiling STARK-ST50 model
python tracking/profile_model.py --script stark_st2 --config baseline
# Profiling STARK-ST101 model
python tracking/profile_model.py --script stark_st2 --config baseline_R101
# Profiling STARK-Lightning-X-trt
python tracking/profile_model_lightning_X_trt.py
```

## Model Zoo
The trained models, the training logs, and the raw tracking results are provided in the [model zoo](MODEL_ZOO.md)

## Acknowledgments
* Thanks for the great [PyTracking](https://github.com/visionml/pytracking) Library, which helps us to quickly implement our ideas.
* We use the implementation of the DETR from the official repo [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr).  
