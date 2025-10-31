import yaml
import os

import torch
import shutil


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = str(len(dirs))

    return log_dir_index


def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        setattr(cfg, k, v)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype

    if cfg.mode == 'train':
        cfg.base_dir = cfg.base_dir + '/' + cfg.model_name + '/' + cfg.learning_type + '/' + f'seg_{str(cfg.num_seg)}'
    elif cfg.mode in ['dir', 'mixed', 'finetune', 'mmd', 'dan', 'combined']:
        cfg.base_dir = cfg.base_dir + '/' + cfg.model_name + '/' + cfg.learning_type + '/' + f'seg_{str(cfg.num_seg)}' + '/' + f'epoch_{str(cfg.n_epochs)}' + '/' + f'lr_{str(cfg.lr)}'

    print('base_dir: %s' % cfg.base_dir)
    os.makedirs(cfg.base_dir, exist_ok=True)

    if cfg.save_all:
        index = get_log_dir_index(cfg.base_dir)
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, index)
    else:
        temp_dir = '%s/%s' % (cfg.base_dir, 'temp')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir,'temp')

        
    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    cfg.out_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)

    return cfg


class Config:

    def __init__(self, cfg_id, mode='train'):
        self.id = cfg_id

        if mode == 'train' or mode == 'test':
            cfg_name = f'./cfg/{mode}/%s.yml' % cfg_id
        elif mode in ['dir', 'dir', 'mixed', 'finetune', 'mmd', 'dan', 'combined']:
            cfg_name = f'./cfg/update/%s.yml' % cfg_id

        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        if mode == 'train':
            self.base_dir = 'results/train'
        elif mode in ['dir', 'mixed', 'finetune', 'mmd', 'dan', 'combined']:
            self.base_dir = f'results/update/{mode}'
        elif mode == 'test':
            self.base_dir = 'results/inference'
        
        os.makedirs(self.base_dir, exist_ok=True)

        self.model_name = cfg['model_name']

        # common
        self.seed = cfg['seed']
        self.train_mode = cfg['train_mode']
        self.test_mode = cfg['test_mode']
        self.train_sample = cfg['train_sample']
        self.test_sample = cfg['test_sample']
        self.n_epochs = cfg['n_epochs']
        self.lr = cfg['lr']
        self.val_step = cfg['val_step']
        self.save_step = cfg['save_step']
        self.normalize = cfg['normalize']
        self.save_all = cfg['save_all']
        self.optimizer_type = cfg['optimizer_type']
        self.learning_type = cfg['learning_type']

        # model
        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.dropout = cfg['dropout']
        self.num_seg = cfg['num_seg']

        if mode in ['dir', 'mixed', 'finetune', 'mmd', 'dan', 'combined']:
            self.mmd_weight = cfg['mmd_weight']
            self.dan_weight = cfg['dan_weight']