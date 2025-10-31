import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch


def data_loader(mode='sim', sample=[1, 3, 4, 5, 6], normalize="minmax"):
    data_stats = np.load(f'./dataset/stats/data_stats.npz', allow_pickle=True)

    x_list = []
    y_list = []
    for i in sample:
        subdata_dict = np.load(f'./dataset/{mode}_{i}.npz', allow_pickle=True)
        tc, lenx_l, lenx_r, leny_u, leny_d = subdata_dict['lc'], subdata_dict['lenx_l'], subdata_dict['lenx_r'], subdata_dict['leny_u'], subdata_dict['leny_d']
        tc, lenx_l, lenx_r, leny_u, leny_d = np.array(tc, dtype=np.float32), np.array(lenx_l, dtype=np.float32), np.array(lenx_r, dtype=np.float32), np.array(leny_u, dtype=np.float32), np.array(leny_d, dtype=np.float32)
        tc = tc / 100000.0
        
        tp = np.roll(tc, shift=-1, axis=0)

        data = np.concatenate((tc[None, ...], tp[None, ...], lenx_l, lenx_r, leny_u, leny_d), axis=0)
        data = data.transpose(1, 0)

        if normalize == "minmax":
            data[:, 2:] = (data[:, 2:] - data_stats['min']) / (data_stats['max'] - data_stats['min'])
        elif normalize == "std":
            data[:, 2:] = (data[:, 2:] - data_stats['mean']) / data_stats['std']
        else:
            pass
            
        # convert into x and y for training and test
        for i in range(1, len(data)):
            x, y = data[:i], data[i]
            x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
            x_list.append(x)
            y_list.append(y)

    y = torch.stack(y_list)

    return x_list, y


def test_data_loader(mode='sim', sample=[1, 3, 4, 5, 6], normalize="minmax"):
    data_stats = np.load(f'./dataset/stats/data_stats.npz', allow_pickle=True)

    data_list = []
    for i in sample:
        subdata_dict = np.load(f'./dataset/{mode}_{i}.npz', allow_pickle=True)
        tc, lenx_l, lenx_r, leny_u, leny_d = subdata_dict['lc'], subdata_dict['lenx_l'], subdata_dict['lenx_r'], subdata_dict['leny_u'], subdata_dict['leny_d']
        tc, lenx_l, lenx_r, leny_u, leny_d = np.array(tc, dtype=np.float32), np.array(lenx_l, dtype=np.float32), np.array(lenx_r, dtype=np.float32), np.array(leny_u, dtype=np.float32), np.array(leny_d, dtype=np.float32)
        tc = tc / 100000.0
        
        tp = np.roll(tc, shift=-1, axis=0)

        data = np.concatenate((tc[None, ...], tp[None, ...], lenx_l, lenx_r, leny_u, leny_d), axis=0)
        data = data.transpose(1, 0)

        if normalize == "minmax":
            data[:, 2:] = (data[:, 2:] - data_stats['min']) / (data_stats['max'] - data_stats['min'])
        elif normalize == "std":
            data[:, 2:] = (data[:, 2:] - data_stats['mean']) / data_stats['std']
        else:
            pass

        data = torch.tensor(data, dtype=torch.float32)
        data_list.append(data)
        
    return data_list

def data_loader4update(exp_name, update_time, normalize="minmax", update_mode='finetune'):

    source_x_list, source_y = data_loader(mode='sim', sample=[1, 3, 4, 5, 6], normalize=normalize)

    sample_id = int(exp_name.split('_')[-1])
    mode = exp_name.split('_')[0]
    update_x_list, update_y = data_loader(mode=mode, sample=[sample_id], normalize=normalize)

    # extract data for update
    train_x_list = update_x_list[:update_time]
    train_y = update_y[:update_time]

    if update_mode not in ['finetune', 'dir']:
        train_x_list = source_x_list + train_x_list
        train_y = torch.cat((source_y, train_y), dim=0)

    # data for test
    test_data_list = test_data_loader(mode='exp', sample=[sample_id], normalize=normalize)
    test_data = test_data_list[0]

    return train_x_list, train_y, test_data


# python -m data_loader.data_loader
if __name__ == '__main__':
    # ---------------------------------------------------------------------
    # test data_loader
    # train_mode = 'sim'
    # test_mode = 'exp'
    # train_sample = [1, 3, 4, 5, 6]
    # test_sample = [1, 2, 3, 7]
    # train_data = data_loader(mode=train_mode, sample=train_sample)
    # test_data = data_loader(mode=test_mode, sample=test_sample)
    
    # x_train, y_train = train_data
    # x_test, y_test = test_data
    # print(f'Number of samples: {len(x_train)}')
    # print(f'Number of samples: {len(x_test)}')

    # for i in range(len(x_train)):
    #     print(f'x_train[{i}].shape: {x_train[i].shape}, y_train[{i}].shape: {y_train[i].shape}')
    # for i in range(len(x_test)):
    #     print(f'x_test[{i}].shape: {x_test[i].shape}, y_test[{i}].shape: {y_test[i].shape}')

    # ---------------------------------------------------------------------
    # test data_loader4update
    exp_name = 'exp_1'
    update_time = 1
    train_x_list, train_y, test_data = data_loader4update(exp_name, update_time, normalize="minmax", update_mode='dir')
    print(f'Number of updated samples: {len(train_x_list)}, {len(train_y)}')
    print(f'Number of test samples: {test_data.shape}')

    # ---------------------------------------------------------------------
    # Recurrent load data
    # test_mode = 'exp'
    # # test_sample = [1, 2, 7]
    # test_sample = [1]
    # for item in test_sample:
    #     x_list, y = data_loader(mode=test_mode, sample=[item])

    #     for x in x_list:
    #         print(f'x.shape: {x.shape}')

    # # ---------------------------------------------------------------------
    # Test test_data_loader
    # test_mode = 'exp'
    # test_sample = [1]
    # test_data_list = test_data_loader(mode=test_mode, sample=test_sample)
    # for data in test_data_list:
    #     print(f'data.shape: {data.shape}')