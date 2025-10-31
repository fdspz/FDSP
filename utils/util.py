import torch
import torch.nn as nn
import numpy as np
from termcolor import cprint


def get_recurrent_cra(data, model, model_name, data_stats, normalize, learning_type, log, device):
    data_length = data.shape[0]
    # print(f"data_length: {data_length}")
    time_seq = data[:, :2].to(device)

    # cras = np.zeros((data_length - 1, data_length - 1))

    metrics_list = ['cra', 'mae', 'rmse', 'mape']
    metrics = {metric: np.zeros((data_length - 1, data_length - 1)) for metric in metrics_list}

    total_y_hat = []
    total_y = []

    for i in range(1, data_length):
        x = data[:i].unsqueeze(0).to(device)
        y = data[i:, 2:].to(device)

        y_hat_list = []
        for j in range(i, data_length):
            y_hat_padding = np.zeros((data_length, 14)).astype(np.float32)
            y_padding = np.zeros((data_length, 14)).astype(np.float32)


            delta_g = model(x, device)
            y_hat = get_y(delta_g, x, learning_type, device)
            
            y_hat_list.append(y_hat)

            # update input
            y_hat = torch.cat([time_seq[j].unsqueeze(0), y_hat], dim=1).unsqueeze(0)
            x = torch.cat([x, y_hat], dim=1)
        
        y_hat = torch.cat(y_hat_list, dim=0)


        y, y_hat = denormalize(y, y_hat, data_stats, normalize, model_name)

        y_padding[i:] = y
        y_hat_padding[i:] = y_hat

        total_y_hat.append(y_hat_padding)
        total_y.append(y_padding)

        sub_cra = get_cra(y, y_hat)

        # cras[i - 1, :sub_cra.shape[0]] = sub_cra
        mae, rmse, mape = compute_mae_rmse_mape(y, y_hat)
        metrics['cra'][i - 1, :sub_cra.shape[0]] = sub_cra
        metrics['mae'][i - 1, :sub_cra.shape[0]] = mae
        metrics['rmse'][i - 1, :sub_cra.shape[0]] = rmse
        metrics['mape'][i - 1, :sub_cra.shape[0]] = mape

    total_y_hat = np.array(total_y_hat)
    total_y = np.array(total_y)


    prediction_dict = {
        'y': total_y,
        'y_hat': total_y_hat,
    }

    return metrics, prediction_dict


def get_recurrent_cra_update(data, model, model_name, update_time, data_stats, normalize, learning_type, log, device):
    data_length = data.shape[0]
    time_seq = data[:, :2].to(device)

    metrics_list = ['cra', 'mae', 'rmse', 'mape']
    metrics = {metric: np.zeros(data_length - update_time) for metric in metrics_list}

    x = data[:update_time + 1].unsqueeze(0).to(device)
    y = data[update_time + 1:, 2:].to(device)

    

    y_hat_list = []
    for j in range(1, data_length - update_time):

        delta_g = model(x, device)
        y_hat = get_y(delta_g, x, learning_type, device)
    
        y_hat_list.append(y_hat)

        # update input
        y_hat = torch.cat([time_seq[update_time + j].unsqueeze(0), y_hat], dim=1).unsqueeze(0)
        x = torch.cat([x, y_hat], dim=1)
    
    y_hat = torch.cat(y_hat_list, dim=0)

    y, y_hat = denormalize(y, y_hat, data_stats, normalize, model_name)

    prediction_dict = {
        'y': y,
        'y_hat': y_hat,
    }

    sub_cra = get_cra(y, y_hat)

    mae, rmse, mape = compute_mae_rmse_mape(y, y_hat)
    metrics['cra'] = sub_cra
    metrics['mae'] = mae
    metrics['rmse'] = rmse
    metrics['mape'] = mape

    return metrics, prediction_dict



def get_cra(y, y_hat):
    """
        Return
        cra: [sample1]
        ...
        cra: [sample1, sample2, ...]
    """

    cra_list = []
    for i in range(y.shape[0]):
        cra = 1 - np.abs(y[i] - y_hat[i]) / (y[i] + 1e-8)
        cra = np.mean(cra)
        cra_list.append(cra)
    
    cra = np.array(cra_list)

    return cra
    

def compute_mae_rmse_mape(y, y_hat):

    # MAE, RMSE, MAPE
    mae_list = []
    rmse_list = []
    mape_list = []
    for i in range(y.shape[0]):
        mae = np.mean(np.abs(y[i] - y_hat[i]))
        rmse = np.sqrt(np.mean((y[i] - y_hat[i]) ** 2))
        mape = np.mean(np.abs(y[i] - y_hat[i]) / (y[i] + 1e-8))

        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
    mae = np.array(mae_list)
    rmse = np.array(rmse_list)
    mape = np.array(mape_list)

    # mae = np.mean(np.abs(y - y_hat))
    # rmse = np.sqrt(np.mean((y - y_hat) ** 2))
    # mape = np.mean(np.abs(y - y_hat) / y)

    return mae, rmse, mape

def get_y(delta_g, input_x, learning_type, device, particle_delta_g=None):
    """
        Function for getting y_hat by increment
        y_hat = x_t + delta_g * delta_t
    """
    last_x = []
    for i in range(len(input_x)):
        last_x.append(input_x[i][-1])
    last_x = torch.stack(last_x, dim=0).to(device)

    delta_t = last_x[:, 1] - last_x[:, 0]

    if learning_type == 'grad':
        y_hat = last_x[:, 2:] - delta_g * delta_t.unsqueeze(1)  # learning the gradient
    elif learning_type == 'diff':
        y_hat = last_x[:, 2:] - delta_g                       # learning the difference
    else:
        raise ValueError("learning_type should be 'grad' or 'diff'")

    if particle_delta_g is not None:
        if learning_type == 'grad':
            particle_y_hat = last_x[:, 2:].unsqueeze(0) - particle_delta_g * delta_t.unsqueeze(1).unsqueeze(0)  # learning the gradient
        elif learning_type == 'diff':
            particle_y_hat = last_x[:, 2:].unsqueeze(0) - particle_delta_g                                    # learning the difference
        else:
            raise ValueError("learning_type should be 'grad' or 'diff'")
        return y_hat, particle_y_hat

    return y_hat


def denormalize(y, y_hat, data_stats, normalize, model_name, particle_y_hat=None):

    def denorm(x, stats):
        if normalize == "minmax":
            return x * (stats['max'] - stats['min']) + stats['min']
        elif normalize == "std":
            return x * stats['std'] + stats['mean']
        else:
            return x
    y = y.cpu().detach().numpy()
    y_hat = y_hat.cpu().detach().numpy()
    y = denorm(y, data_stats)
    y_hat = denorm(y_hat, data_stats)
    
    return y, y_hat