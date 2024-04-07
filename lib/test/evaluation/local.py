from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.network_path = '/home/hskim/projects/EXOT/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/home/hskim/projects/EXOT'
    settings.result_plot_path = '/home/hskim/projects/EXOT/test/result_plots'
    settings.results_path = '/home/hskim/projects/EXOT/test/tracking_results'    # Where to store tracking results
    settings.robot_path = '/home/hskim/projects/EXOT/data/robot-data'
    settings.save_dir = '/home/hskim/projects/EXOT'
    settings.segmentation_path = '/home/hskim/projects/EXOT/test/segmentation_results'
    settings.trek150_path = '/home/hskim/projects/EXOT/data/TREK-150'
    return settings

