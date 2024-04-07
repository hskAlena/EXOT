# EXOT
The official implementation of the **ICRA2023** paper [**EXOT: Exit-aware Object Tracker for Safe Robotic Manipulation of Moving Object**](https://arxiv.org/abs/2306.05262)

## Install the environment
Use the [Anaconda](https://www.anaconda.com/)
```
conda create -n exot python=3.8
conda activate exot
bash install_exot.sh
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${EXOT_ROOT}
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
You can download TREK-150 dataset in [https://github.com/matteo-dunnhofer/TREK-150-toolkit](https://github.com/matteo-dunnhofer/TREK-150-toolkit).

## Model Zoo & RMOT-223 Dataset
The trained models and UR5e-made RMOT-223 dataset are provided in the [google drive](https://drive.google.com/drive/folders/1C75Q1t4bNeECwmxt7YUoPgIwYNyE4muu?usp=sharing).

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
Refer to `benchmarking/train_pl.sh` or `benchmarking/train.sh` for detailed commands.

## Test and evaluate EXOT on benchmarks

```
python tracking/test.py exotst_tracker baseline_mix_lowdim --dataset robot_test 
python tracking/analysis_results.py --tracker_param baseline_mix_lowdim --dataset robot_test --name exotst_tracker
```
For more config options and further details, see `benchmarking/test.sh`. 
- Evaluate using UR5e robot 
```
python tracking/video_demo.py exotst_tracker baseline_mix_lowdim --track_format run_video_robot --modelname exot_merge
```
For more config options and further details, see `tracking/run_video_demo.sh`.

## Making pick and place dataset using UR5e
- For automatic creation of pick and place dataset using a hand camera with UR5e, see `data_preparation` folder.
- For human collection of pick and place dataset, buy a 3D Mouse and a running program created by [Radalytica](https://www.universal-robots.com/plus/products/radalytica/3d-mouse-move/). 


## Acknowledgments
* We use the implementation of the STARK from the official repo [https://github.com/researchmm/Stark](https://github.com/researchmm/Stark).  
