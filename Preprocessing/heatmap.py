import numpy
from matplotlib import pyplot as plt


def draw_display(disp_size, dpi=100.0):
    # construct screen (black background)
    screen = numpy.zeros((disp_size[1], disp_size[0], 3), dtype='float32')
    # determine the figure size in inches
    fig_size = (disp_size[0] / dpi, disp_size[1] / dpi)
    # create a figure
    fig = plt.figure(figsize=fig_size, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, disp_size[0], 0, disp_size[1]])
    ax.imshow(screen)

    return fig, ax


def gaussian(x, sx, y=None, sy=None):
    # square Gaussian if only x values are passed
    if y is None:
        y = x
    if sy is None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx ** 2)) + ((float(j) - yo) ** 2 / (2 * sy ** 2))))
    return M


def draw_heatmap(X, Y, dur, disp_size=None, alpha=1, save_to=None, gaussian_wh=200, gaussian_sd=6):
    if disp_size is None:
        disp_size = (1800, 600)
    fig, ax = draw_display(disp_size)

    # Gaussian
    g = gaussian(gaussian_wh, gaussian_wh / gaussian_sd)
    # matrix of zeroes
    st = gaussian_wh / 2
    hm_size = (disp_size[1] + int(2 * st), disp_size[0] + int(2 * st))
    heatmap = numpy.zeros(hm_size, dtype=float)

    # create heatmap
    for i in range(0, len(X)):
        # get x and y coordinates
        x = st + X[i] - int(gaussian_wh / 2)
        y = st + Y[i] - int(gaussian_wh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < disp_size[0]) or (not 0 < y < disp_size[1]):
            h_adj = [0, gaussian_wh]
            v_adj = [0, gaussian_wh]
            if x < 0:
                h_adj[0] = abs(x)
                x = 0
            elif x > disp_size[0]:
                h_adj[1] = gaussian_wh - int(x - disp_size[0])
            if y < 0:
                v_adj[0] = abs(y)
                y = 0
            elif y > disp_size[1]:
                v_adj[1] = gaussian_wh - int(y - disp_size[1])
            # add adjusted Gaussian to the current heatmap
            cur_hm = heatmap[int(y):int(y + v_adj[1]), int(x):int(x + h_adj[1])]
            cur_g = g[int(v_adj[0]):int(v_adj[1]), int(h_adj[0]):int(h_adj[1])]
            if cur_hm.shape == cur_g.shape:
                cur_hm += cur_g * dur[i]
        else:
            # add Gaussian to the current heatmap
            heatmap[int(y):int(y + gaussian_wh), int(x):int(x + gaussian_wh)] += g * dur[i]
    # resize heatmap
    heatmap = heatmap[int(st):disp_size[1] + int(st), int(st):disp_size[0] + int(st)]
    # remove zeros
    lb = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lb] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='gray', alpha=alpha)

    # ax.invert_yaxis()

    if save_to is not None:
        fig.savefig(save_to)

    return fig
