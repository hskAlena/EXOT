from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/hskim/projects/EXOT/data/got10k_lmdb'
    settings.got10k_path = '/home/hskim/projects/EXOT/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/hskim/projects/EXOT/data/lasot_lmdb'
    settings.lasot_path = '/home/hskim/projects/EXOT/data/lasot'
    settings.network_path = '/home/hskim/projects/EXOT/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/hskim/projects/EXOT/data/nfs'
    settings.otb_path = '/home/hskim/projects/EXOT/data/OTB2015'
    settings.prj_dir = '/home/hskim/projects/EXOT'
    settings.result_plot_path = '/home/hskim/projects/EXOT/test/result_plots'
    settings.results_path = '/home/hskim/projects/EXOT/test/tracking_results'    # Where to store tracking results
    settings.robot_path = '/home/hskim/projects/EXOT/data/robot-data'
    settings.save_dir = '/home/hskim/projects/EXOT'
    settings.segmentation_path = '/home/hskim/projects/EXOT/test/segmentation_results'
    settings.tc128_path = '/home/hskim/projects/EXOT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/hskim/projects/EXOT/data/trackingNet'
    settings.trek150_path = '/home/hskim/projects/EXOT/data/TREK-150'
    settings.uav_path = '/home/hskim/projects/EXOT/data/UAV123'
    settings.vot_path = '/home/hskim/projects/EXOT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

