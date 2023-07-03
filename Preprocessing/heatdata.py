from Preprocessing.MECO_data_split import split_into_time_series
from Preprocessing.heatmap import draw_heatmap

import numpy as np
import matplotlib.pyplot as plt
import cv2
import io


def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.close(fig)
    return img


def get_gauss_data(data, wh=100):
    X, y, demo = split_into_time_series(data, truncate=False, test_size=0)
    X_x = [np.array(t['Fix_X'].values) for t in X]
    X_y = [np.array(t['Fix_Y'].values) for t in X]
    X_dur = [np.array(t['Fix_Duration'].values) for t in X]

    X_img = np.zeros((len(X), 600, 1800))
    for i in range(len(X)):
        X_img[i, :, :] = get_img_from_fig(draw_heatmap(X_x[i], X_y[i], X_dur[i], gaussian_wh=wh))

    return X_img, y, demo


def get_scatter_data(data, ):
    X, y, demo = split_into_time_series(data, truncate=False, test_size=0)
    X_x = [np.array(t['Fix_X'].values) for t in X]
    X_y = [np.array(t['Fix_Y'].values) for t in X]
    X_dur = [np.array(t['Fix_Duration'].values) for t in X]

    X_img = np.zeros((len(X), 600, 1800))
    for i in range(len(X)):
        mtx = np.zeros((600, 1800))

        for j in range(len(X_x[i])):
            if X_x[i][j] < 1800 and X_y[i][j] < 600:
                mtx[X_y[i][j]][X_x[i][j]] = X_dur[i][j]

        fig, ax = plt.subplots(figsize=(6, 18), dpi=100, facecolor='b', edgecolor='k')
        ax.set_axis_off()

        plot_list = []
        for rows, cols in zip(np.where(mtx != 0)[0], np.where(mtx != 0)[1]):
            plot_list.append([cols, rows, mtx[rows, cols]])
        plot_list = np.array(plot_list)

        plt.scatter(plot_list[:, 0], plot_list[:, 1], c=plot_list[:, 2], s=plot_list[:, 2], cmap='gray')

        plt.xlim(0, mtx.shape[1])
        plt.ylim(0, mtx.shape[0])
        plt.gca().invert_yaxis()

        X_img[i, :, :] = get_img_from_fig(fig).T
    return X_img, y, demo
