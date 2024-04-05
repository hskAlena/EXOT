from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/hskim/projects/mfmot/STARK/data/got10k_lmdb'
    settings.got10k_path = '/home/hskim/projects/mfmot/STARK/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/hskim/projects/mfmot/STARK/data/lasot_lmdb'
    settings.lasot_path = '/home/hskim/projects/mfmot/STARK/data/lasot'
    settings.network_path = '/home/hskim/projects/mfmot/STARK/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/hskim/projects/mfmot/STARK/data/nfs'
    settings.otb_path = '/home/hskim/projects/mfmot/STARK/data/OTB2015'
    settings.prj_dir = '/home/hskim/projects/mfmot/STARK'
    settings.result_plot_path = '/home/hskim/projects/mfmot/STARK/test/result_plots'
    settings.results_path = '/home/hskim/projects/mfmot/STARK/test/tracking_results'    # Where to store tracking results
    settings.robot_path = '/home/hskim/projects/mfmot/STARK/data/robot-data'
    settings.save_dir = '/home/hskim/projects/mfmot/STARK'
    settings.segmentation_path = '/home/hskim/projects/mfmot/STARK/test/segmentation_results'
    settings.tc128_path = '/home/hskim/projects/mfmot/STARK/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/hskim/projects/mfmot/STARK/data/trackingNet'
    settings.trek150_path = '/home/hskim/projects/mfmot/STARK/data/TREK-150'
    settings.uav_path = '/home/hskim/projects/mfmot/STARK/data/UAV123'
    settings.vot_path = '/home/hskim/projects/mfmot/STARK/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

