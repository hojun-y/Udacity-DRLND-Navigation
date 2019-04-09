import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters


def save_line_plot(data, title, path, sigma=1):
    plt.title(title)
    plt.plot(data, '0.6', linewidth=0.5)
    smoothed = filters.gaussian_filter1d(data, sigma)
    plt.plot(smoothed, 'r', linewidth=0.9)
    plt.grid(True, which='both')
    plt.axhline(color="k", linewidth=0.8)
    plt.axvline(color="k", linewidth=0.8)
    plt.show()
    plt.savefig(path)
    plt.clf()
