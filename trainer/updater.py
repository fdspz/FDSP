import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from termcolor import cprint

from data_loader.data_loader import data_loader, data_loader4update
from utils.util import get_y, get_cra, get_recurrent_cra_update
from utils.generate_prediction import generate_img, generate_img4update
from models.rnns import rnns
from models.transformer_net import TransformerNetwork

import csv
from trainer.compute_loss import compute_loss


class updater:
    def __init__(self, cfg, logger, tb_logger):
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.mode = cfg.mode

        self.seed = cfg.seed
        self.device = cfg.device
        self.train_mode = cfg.train_mode
        self.test_mode = cfg.test_mode

        self.n_epochs = cfg.n_epochs
        self.lr = cfg.lr

        self.train_sample = cfg.train_sample
        self.test_sample = cfg.test_sample
        self.normalize = cfg.normalize
        
        self.val_step = cfg.val_step
        self.save_step = cfg.save_step
        self.save_all = cfg.save_all

        self.model_name = cfg.model_name
        self.optimizer_type = cfg.optimizer_type
        self.learning_type = cfg.learning_type

        # model parameters
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.dropout = cfg.dropout
        self.num_seg = cfg.num_seg

        self.mmd_weight = cfg.mmd_weight
        self.dan_weight = cfg.dan_weight

        self.train_data = None
        self.test_data = None

        self.model = None
        self.optimizer = None
        self.mse_criterion = None
        self.label_criterion = None

        self.data_stats = np.load(f'./dataset/stats/data_stats.npz', allow_pickle=True)

        self.last_pre_result = None

        self.exp_names = ['exp_1', 'exp_2', 'exp_3', 'exp_7']
        self.num_samples = [10, 9, 9, 10]

    def load_data(self):
        self.x_train, self.y_train, self.test_data = data_loader4update(self.exp_names[self.exp_id], self.update_time, self.normalize, self.mode)
        self.y_train = self.y_train[:, 2:].to(self.device)
        self.test_data = self.test_data.to(self.device)

    def build_model(self):
        # Initialize model
        print('>> update strategy: ', self.mode)
        print('>> model name: ', self.model_name)

        if self.model_name == 'transformer':
            self.model = TransformerNetwork(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout, num_seg=self.num_seg, learning_type=self.learning_type).to(self.device)
        else:
            self.model = rnns(self.input_dim, self.hidden_dim, self.model_name, self.dropout, self.num_seg, self.learning_type).to(self.device)

        self.logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(self.model.parameters())) / 1000000.0))

        # load trained model
        if self.mode not in ['dir', 'mixed']:
            model_dir = f'results/train/{self.model_name}/{self.learning_type}/seg_{self.num_seg}/{self.seed}/models'
            self.model.load_state_dict(torch.load(f'{model_dir}/100_model.pth'))

        if self.mode == 'finetune':
            # fix part parameters of the model for fine-tuning
            for name, param in self.model.named_parameters():
                if 'mlp_i' in name or 'mixer' in name:
                    param.requires_grad = False

        # Initialize optimizer, criterion
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == 'RMSProp':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError('Invalid optimizer type')


        self.mse_criterion = torch.nn.MSELoss()
        self.label_criterion = torch.nn.CrossEntropyLoss()

    def run_train_step(self):
        self.model.train()

        self.total_loss = 0

        self.t1 = time.time()

        self.optimizer.zero_grad()

        self.domain_label = torch.zeros(len(self.x_train), dtype=torch.long).to(self.device)
        self.domain_label[51:] = 1
        loss, self.loss_items = compute_loss(self.model, self.model_name, self.mode, self.x_train, self.y_train, self.domain_label, self.mse_criterion, self.label_criterion, self.learning_type, self.mmd_weight, self.dan_weight, self.device)

        loss.backward()
        self.optimizer.step()
        self.total_loss = loss.item()

    def after_train_step(self):
        stat_str = f'Epoch {self.epoch}, Loss: {self.total_loss:.5f}, lr: {self.optimizer.param_groups[0]["lr"]:.5f}, time: {time.time()-self.t1:.5f}, Pred Loss: {self.loss_items[0]:.5f}, Label Loss: {self.loss_items[2]:.5f}, MMD Loss: {self.loss_items[3]:.5f}'

        self.logger.info(stat_str)

    def run_test_step(self):
        self.logger.info(f'-'*100)
        self.logger.info(f'Epoch {self.epoch} Test Start')
        self.model.eval()
        
        self.metrics, self.prediction_dict = get_recurrent_cra_update(self.test_data, self.model, self.model_name, self.update_time, self.data_stats, self.normalize, self.learning_type, self.logger, self.device)

        self.sub_mean_list = []
        metrics_keys = list(self.metrics.keys())
        for key, value in self.metrics.items():
            sub_mean_metrics = np.sum(value)/value.shape[0]
            self.sub_mean_list.append(sub_mean_metrics)


        for i in range(len(metrics_keys)):
            self.logger.info(f'>> sub {metrics_keys[i]}: {np.round(self.sub_mean_list[i], 5)}')
        self.logger.info(f'-'*100)

    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.model_path}/ut_{self.update_time}.pth')

    def save_results(self):
        np.savez(f'{self.out_dir}/metrics_dict.npz', **self.metrics_dict)

        with open(f'{self.result_dir}/metrics.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['test_exp_name', 'metrics_name'] + [f'sample_{i+1}' for i in range(1, self.num_samples[self.exp_id])] + ['mean'])
            for key, sub_data in self.mean_metrics_dict.items():
                mean_value = np.mean(sub_data, axis=0)
                for idx, sub_key in enumerate(['cra', 'mae', 'rmse', 'mape']):
                    if len(sub_data) == 9:
                        writer.writerow([key, sub_key] + list(np.round(sub_data[:, idx], 5)) + [f'{mean_value[idx]:.5f}'])
                    elif len(sub_data) == 8:
                        writer.writerow([key, sub_key] + list(np.round(sub_data[:, idx], 5)) + [''] + [f'{mean_value[idx]:.5f}'])

        self.logger.info('>>> Save model and metrics')


    def save_prediction(self):
        np.savez(f'{self.prediction_path}/prediction_dict.npz', **self.all_prediction_dict)

    def update(self):
        self.load_data()

        for self.epoch in range(1, self.total_epochs + 1):
            self.run_train_step()
            self.after_train_step()
                
        self.run_test_step()

    def run(self):
        self.metrics_dict = {}
        self.mean_metrics_dict = {}

        for self.exp_id in range(len(self.exp_names)):
            self.start_time = time.time()
            self.build_model()

            self.metrics_dict[self.exp_names[self.exp_id]] = {}
            self.mean_metrics_dict[self.exp_names[self.exp_id]] = []
            
            total_y_hat = []
            total_y = []
            for self.update_time in range(1, self.num_samples[self.exp_id]):
                if self.mode in ['dir', 'mixed']:
                    self.build_model()

                if self.mode not in ['dir', 'mixed']:
                    self.total_epochs = self.update_time * self.n_epochs
                else:
                    self.total_epochs = 200

                self.logger.info(f'>>> Start updating model for exp {self.exp_names[self.exp_id]} with {self.update_time} samples')

                    
                self.model_path = f'{self.cfg.model_path}/{self.exp_names[self.exp_id]}'
                self.prediction_path = f'{self.cfg.out_dir}/prediction_results/{self.exp_names[self.exp_id]}'
                os.makedirs(self.model_path, exist_ok=True)
                os.makedirs(self.prediction_path, exist_ok=True)

                self.out_dir = self.cfg.out_dir
                self.result_dir = self.cfg.result_dir
                
                self.update()
                self.logger.info(f'>>> Finish updating model for exp {self.exp_names[self.exp_id]} with {self.update_time} samples')
                cprint('-'*100, 'red')

                self.metrics_dict[self.exp_names[self.exp_id]][self.update_time] = self.metrics
                self.mean_metrics_dict[self.exp_names[self.exp_id]].append(np.array(self.sub_mean_list))

                self.y_hat_padding = np.zeros((self.test_data.shape[0], 14)).astype(np.float32)
                self.y_padding = np.zeros((self.test_data.shape[0], 14)).astype(np.float32)

                self.y_hat_padding[self.update_time + 1:] = self.prediction_dict['y_hat']
                self.y_padding[self.update_time + 1:] = self.prediction_dict['y']
                total_y_hat.append(self.y_hat_padding)
                total_y.append(self.y_padding)
                
                self.save_model()

            self.all_prediction_dict = {
                'y_hat': np.array(total_y_hat),
                'y': np.array(total_y),
            }
            self.save_prediction()
            
            self.mean_metrics_dict[self.exp_names[self.exp_id]] = np.array(self.mean_metrics_dict[self.exp_names[self.exp_id]])

            self.logger.info(f'>>> Total Training time: {time.time()-self.start_time:.5f} seconds')

        self.save_results()
        self.logger.info('>>> Finish updating all models')


            
        
        