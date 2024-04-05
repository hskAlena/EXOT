from pytorch_lightning import LightningDataModule
from lib.train.base_functions import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class RobotDataModule(LightningDataModule):
    def __init__(self, data_dir='./', k=1, split_seed=123, num_splits=10, batch_size=256, kfoldness = True, test_size=0.4, num_workers=8, pin_memory=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size  
        self.k = k
        self.num_splits = num_splits
        self.split_seed = split_seed
        self.kfoldness = kfoldness
        self.test_size = test_size

    def fill_state(self, cfg, settings):
        self.cfg = cfg
        self.settings = settings

        transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                        tfm.RandomHorizontalFlip(probability=0.5))

        transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                        tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

        transform_val = tfm.Transform(tfm.ToTensor(),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

        # The tracking pairs processing module
        output_sz = settings.output_sz
        search_area_factor = settings.search_area_factor

        self.data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                        output_sz=output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        transform=transform_train,
                                                        joint_transform=transform_joint,
                                                        settings=settings)

        self.data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                        output_sz=output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        transform=transform_val,
                                                        joint_transform=transform_joint,
                                                        settings=settings)

        # Train sampler and loader
        settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
        settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
        self.sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
        self.train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
        
    # def prepare_data(self):
    #     '''called only once and on 1 GPU'''
    #     # download data
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        cfg = self.cfg
        settings = self.settings
        
        
        if stage == 'fit' or stage is None:
            self.dataset_full = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=self.data_processing_train,
                                            frame_sample_mode=self.sampler_mode, batch_size=self.batch_size, train_cls=self.train_cls)

            if self.kfoldness:
                kfold = KFold(n_splits = self.num_splits, shuffle = True, random_state = self.split_seed)
                all_splits = [k for k in kfold.split(self.dataset_full)]
                train_indexes, val_indexes = all_splits[self.k]
                self.train_subsampler = torch.utils.data.SubsetRandomSampler(train_indexes)
                self.val_subsampler = torch.utils.data.SubsetRandomSampler(val_indexes)
            else:
                self.dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=self.data_processing_val,
                                          frame_sample_mode=self.sampler_mode, batch_size=self.batch_size, train_cls=self.train_cls)

        if stage == 'test' or stage is None:
            self.dataset_train, self.dataset_test = train_test_split(self.dataset_full, test_size=self.test_size)

    def train_dataloader(self):
        '''returns training dataloader'''
        cfg = self.cfg     
        if self.settings.mode=='multiple':
            persistent_workers = True
        else:
            persistent_workers = False

        if self.kfoldness:
            loader_train = LTRLoader('train', self.dataset_full, training=True, sampler=self.train_subsampler, batch_size=self.batch_size, 
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1)
        else:
            loader_train = LTRLoader('train', self.dataset_full, training=True, batch_size=self.batch_size, 
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1)
        return loader_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        cfg = self.cfg
        if self.settings.mode=='multiple':
            persistent_workers = True
        else:
            persistent_workers = False
        
        if self.kfoldness:
            loader_val = LTRLoader('val', self.dataset_full, training=False, sampler=self.val_subsampler, batch_size=self.batch_size,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, 
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
        else:
            loader_val = LTRLoader('val', self.dataset_val, training=False, batch_size=self.batch_size,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, 
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

        return loader_val

    def test_dataloader(self):
        '''returns test dataloader'''
        cfg = self.cfg
        if self.settings.mode=='multiple':
            persistent_workers = True
        else:
            persistent_workers = False
        
        loader_test = LTRLoader('test', self.dataset_test, training=False, batch_size=self.batch_size,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, 
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

        return loader_test