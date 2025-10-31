import argparse
import torch
from utils.script import *
from utils.logger import create_logger
from tensorboardX import SummaryWriter

from config import Config, update_config

from trainer.trainer import trainer
from trainer.updater import updater


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='cfg')
    parser.add_argument('--mode', default='train', help='train / dir / mixed / finetune / mmd / dan / combined')
    parser.add_argument('--model_name', type=str, default='gru', help='gru / lstm / transformer')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--save_all', type=str2bool, default=True)
    parser.add_argument('--learning_type', type=str, default='grad', help='grad / diff')
    parser.add_argument('--num_seg', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-05)

    parser.add_argument('--mmd_weight', type=float, default=0.1)
    parser.add_argument('--dan_weight', type=float, default=0.1)


    args = parser.parse_args()

    cfg = Config(f'{args.cfg}',  mode=args.mode)
    cfg = update_config(cfg, vars(args))

    set_global_seed(cfg.seed)

    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)

    if args.mode == 'train':
        trainer = trainer(cfg, logger, tb_logger)
        trainer.run()
    elif args.mode in ['dir', 'dir', 'mixed', 'finetune', 'mmd', 'dan', 'combined']:
        updater = updater(cfg, logger, tb_logger)
        updater.run()