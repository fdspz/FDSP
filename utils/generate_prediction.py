import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_key_points(y):
    x_key_points = [50, 65, 85, 100]
    y_key_points = [230, 276, 320]
    points = np.zeros([y.shape[0], y.shape[1], 2])
    # construct the coordinates of the key points

    for i in range(len(y)):
        for j in range(len(y[i])):
            if j < 4:
                points[i, j, 0] = int(y[i, j] * 10)
                points[i, j, 1] = x_key_points[j]
            elif 4 <= j < 8:
                points[i, j, 0] = 550 - int(y[i, j] * 10)
                points[i, j, 1] = x_key_points[j-4]
            elif 8 <= j < 11:
                points[i, j, 0] = y_key_points[j - 8]
                points[i, j, 1] = int(y[i, j] * 10)
            else:
                points[i, j, 0] = y_key_points[j - 11]
                points[i, j, 1] = 150 - int(y[i, j] * 10)
    return points


def generate_img(exp_name, exp_id, num_samples, y_hat_list, y_list):
    num_samples = np.cumsum(num_samples)
    num_samples = np.insert(num_samples, 0, 0)

    y_hat_list = y_hat_list[num_samples[exp_id]:num_samples[exp_id+1]]
    y_list = y_list[num_samples[exp_id]:num_samples[exp_id+1]]
    
    pre_points = generate_key_points(y_hat_list)
    gt_points = generate_key_points(y_list)

    imgs = np.load(f'dataset/raw_images/{exp_name[exp_id]}_images_with_points.npz', allow_pickle=True)

    lc = list(imgs.keys())
    lc.sort(key=int)
    lc = lc[1:]

    rows = 3
    cols = 4
    if len(lc) > 12:
        rows = (len(lc) + 3) // 4
        cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(20, 1.5 * rows))

    for i, ax in enumerate(axs.flat):
        if i < gt_points.shape[0]:
            img = imgs[lc[i]].copy()
            
            pre_points_i = pre_points[i]
            for point in pre_points_i:
                cx, cy = int(point[0]), int(point[1])
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)  # Blue X for marking
            
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f'LC = {lc[i]}')
        ax.axis('off')
    
    return fig



def generate_img4update(y_hat_list, y_list, save_dir, save, mode='train'):
    exp_name = ['exp_1', 'exp_2', 'exp_7']
    num_samples = [10, 9, 10]

    sample_list = np.cumsum(num_samples)
    sample_list = np.insert(sample_list, 0, 0)

    if mode == 'train':
        sample_length = len(y_list)
        breakpoints = [0] * 3
        remaining_length = sample_length

        for i in range(3):
            if remaining_length > num_samples[i]:
                breakpoints[i] = num_samples[i]
                remaining_length -= num_samples[i]
            else:
                breakpoints[i] = remaining_length
                break
        
        breakpoints_list = np.cumsum(breakpoints)
        breakpoints_list = np.insert(breakpoints_list, 0, 0)

        for index in range(len(breakpoints)):
            if breakpoints[index] > 0:
                pre_points = generate_key_points(y_hat_list[breakpoints_list[index]:breakpoints_list[index+1]])
                gt_points = generate_key_points(y_list[breakpoints_list[index]:breakpoints_list[index+1]])

                imgs = np.load(f'dataset/raw_images/{exp_name[index]}_images_with_points.npz', allow_pickle=True)

                lc = list(imgs.keys())
                lc.sort(key=int)
                lc = lc[1:breakpoints[index]+1]

                rows = 3
                cols = 4
                fig, axs = plt.subplots(rows, cols, figsize=(20, 1.5 * rows))

                for i, ax in enumerate(axs.flat):
                    if i < len(lc):
                        img = imgs[lc[i]].copy()
                        gt_points_i = gt_points[i]
                        for point in gt_points_i:
                            cx, cy = int(point[0]), int(point[1])
                            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)  # Blue X for gt
                        
                        pre_points_i = pre_points[i]
                        for point in pre_points_i:
                            cx, cy = int(point[0]), int(point[1])
                            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)  # Red X for marking
                        
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        ax.set_title(f'LC = {lc[i]}')
                    ax.axis('off')

                if save:
                    plt.savefig(f'{save_dir}/train_{exp_name[index]}.png')
                    plt.close()

    elif mode == 'test':
        sample_length = len(y_list)
        start_point = np.sum(num_samples) - sample_length
        breakpoints_list = [[sample_list[i], sample_list[i+1]] for i in range(len(sample_list)-1)]
        
        for i, (start, end) in enumerate(breakpoints_list):
            if start_point > start and start_point < end:
                breakpoints_list[i][0] = start_point
                break
            else:
                breakpoints_list[i][0] = -1
                breakpoints_list[i][1] = -1
        
        count = 0
        for index, (start, end) in enumerate(breakpoints_list):
            if start != -1:
                if count == 0:
                    initial_point = start
                    count += 1
                pre_points = generate_key_points(y_hat_list[start - initial_point: end - initial_point])
                gt_points = generate_key_points(y_list[start - initial_point: end - initial_point])

                imgs = np.load(f'dataset/raw_images/{exp_name[index]}_images_with_points.npz', allow_pickle=True)

                lc = list(imgs.keys())
                lc.sort(key=int)

                lc = lc[start - sample_list[index] + 1:]

                rows = 3
                cols = 4
                fig, axs = plt.subplots(rows, cols, figsize=(20, 1.5 * rows))

                for i, ax in enumerate(axs.flat):
                    if i < len(lc):
                        img = imgs[lc[i]].copy()

                        gt_points_i = gt_points[i]
                        for point in gt_points_i:
                            cx, cy = int(point[0]), int(point[1])
                            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)  # Green X for gt
                        
                        pre_points_i = pre_points[i]
                        for point in pre_points_i:
                            cx, cy = int(point[0]), int(point[1])
                            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)  # Red X for marking
                        
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        ax.set_title(f'LC = {lc[i]}')
                    ax.axis('off')
                
                if save:
                    plt.savefig(f'{save_dir}/test_{exp_name[index]}.png')
                    plt.close()


# python -m utils.generate_prediction
if __name__ == '__main__':
    data_length = 10
    y_hat_list = np.random.rand(data_length, 14)
    y_list = np.random.rand(data_length, 14)
    save_dir = 'results/update/2021-07-15-16-00-00'
    generate_img4update(y_hat_list, y_list, save_dir, save=False, mode='test')
