import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

from data_loader.data_loader import data_loader, test_data_loader
from utils.util import get_y, get_cra, get_recurrent_cra, denormalize
from utils.generate_prediction import generate_img
from models.rnns import rnns
from models.transformer_net import TransformerNetwork

class trainer:
    def __init__(self, cfg, logger, tb_logger):
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.device = cfg.device
        self.train_mode = cfg.train_mode
        self.test_mode = cfg.test_mode

        self.n_epochs = cfg.n_epochs
        self.lr = cfg.lr
        self.train_sample = cfg.train_sample
        self.test_sample = cfg.test_sample
        self.normalize = cfg.normalize
        self.learning_type = cfg.learning_type
        
        self.val_step = cfg.val_step
        self.save_step = cfg.save_step
        self.save_all = cfg.save_all

        self.model_name = cfg.model_name
        self.optimizer_type = cfg.optimizer_type

        # model parameters
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.dropout = cfg.dropout
        self.num_seg = cfg.num_seg

        self.train_data = None
        self.test_data = None

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.data_stats = np.load(f'./dataset/stats/data_stats.npz', allow_pickle=True)

        self.last_pre_result = None

        self.train_exp_name = ['sim_1', 'sim_3', 'sim_4', 'sim_5', 'sim_6']
        self.test_exp_name = ['exp_1', 'exp_2', 'exp_3', 'exp_7']

        self.train_num_samples = [11, 11, 7, 11, 11]
        self.test_num_samples = [10, 9, 9, 10]

    def load_data(self):
        self.train_data = data_loader(mode=self.train_mode, sample=self.train_sample, normalize=self.normalize)
        self.x_train, self.y_train = self.train_data[0], self.train_data[1][:, 2:].to(self.device)

        self.test_data_list = test_data_loader(mode=self.test_mode, sample=self.test_sample, normalize=self.normalize)

    def build_model(self):
        if self.model_name == 'transformer':
            self.model = TransformerNetwork(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout, num_seg=self.num_seg, learning_type=self.learning_type).to(self.device)
        else:
            self.model = rnns(self.input_dim, self.hidden_dim, self.model_name, self.dropout, self.num_seg, self.learning_type).to(self.device)

        self.logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(self.model.parameters())) / 1000000.0))

        # Initialize optimizer, criterion
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == 'RMSProp':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError('Invalid optimizer type')


        self.criterion = torch.nn.MSELoss()
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)

    def run_train_step(self):
        self.model.train()

        self.total_loss = 0

        self.t1 = time.time()

        self.optimizer.zero_grad()

        delta_g = self.model(self.x_train, self.device)
        y_hat = get_y(delta_g, self.x_train, self.learning_type, self.device)
        loss = self.criterion(y_hat, self.y_train)

        loss.backward()
        self.optimizer.step()
        self.total_loss = loss.item()

        del y_hat, loss


    def after_train_step(self):
        # self.scheduler.step()

        stat_str = f'Epoch {self.epoch}, Loss: {self.total_loss:.5f}, lr: {self.optimizer.param_groups[0]["lr"]:.5f}, time: {time.time()-self.t1:.5f}'

        self.logger.info(stat_str)
        

    def run_test_step(self):
        self.logger.info(f'-'*100)
        self.logger.info(f'Epoch {self.epoch} Test Start')
        self.model.eval()
        
        self.metrics_dict = {}
        for idx, test_data in enumerate(self.test_data_list):
            metrics, self.prediction_dict = get_recurrent_cra(test_data, self.model, self.model_name, self.data_stats, self.normalize, self.learning_type, self.logger, self.device)

            mean_list= []
            sub_mean_list = []
            metrics_keys = list(metrics.keys())
            for key, value in metrics.items():
                sub_mean_metrics = np.array([np.sum(value[i])/(value.shape[0]-i) for i in range(value.shape[0])])
                mean_metrics = np.mean(sub_mean_metrics)
                mean_list.append(mean_metrics)
                sub_mean_list.append(sub_mean_metrics)

            self.metrics_dict[self.test_exp_name[idx]] = metrics

            self.logger.info(f'\n>> test sample {self.test_exp_name[idx]}')
            for i in range(len(metrics_keys)):
                self.logger.info(f'>> {metrics_keys[i]}: {mean_list[i]:.5f}')
                self.logger.info(f'>> sub {metrics_keys[i]}: {list(np.round(sub_mean_list[i], 5))}')
        self.logger.info(f'-'*100)

    def save(self):
        torch.save(self.model.state_dict(), f'{self.cfg.model_path}/{self.epoch}_model.pth')

        np.savez(f'{self.cfg.out_dir}/{self.epoch}_metrics_dict.npz', **self.metrics_dict)
        np.savez(f'{self.cfg.out_dir}/{self.epoch}_prediction_dict.npz', **self.prediction_dict)

        with open(f'{self.cfg.result_dir}/{self.epoch}_metrics.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['test_exp_name', 'metrics_name'] + [f'sample_{i+1}' for i in range(10)] + ['mean'])
            for key, sub_dict in self.metrics_dict.items():
                for sub_key, value in sub_dict.items():
                    sub_mean_value = np.array([np.sum(value[i])/(value.shape[0]-i) for i in range(value.shape[0])])
                    if len(sub_mean_value) == 10:

                        writer.writerow([key, sub_key] + list(np.round(sub_mean_value, 5)) + [f'{np.mean(sub_mean_value):.5f}'])
                    elif len(sub_mean_value) == 9:
                        writer.writerow([key, sub_key] + list(np.round(sub_mean_value, 5)) + [''] + [f'{np.mean(sub_mean_value):.5f}'])

        self.logger.info('>>> Save model and metrics')


    def run(self):
        self.load_data()
        self.build_model()

        self.t_0 = time.time()
        for self.epoch in range(1, self.n_epochs+1):
            self.run_train_step()
            self.after_train_step()

            if self.epoch % self.save_step == 0 or self.epoch == self.n_epochs:
                self.run_test_step()
                self.save()
        
        self.logger.info(f'>>> Total Training time: {time.time()-self.t_0:.5f} seconds')