import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse), mean_x, mean_y, ell_radius_x * 2 * scale_x, ell_radius_y * 2 * scale_y

def calculate_perimeter(width, height):
    perimeter = np.pi * (3*(width+height) - np.sqrt((3*width + height)*(width + 3*height)))
    return perimeter

def plot_ellipse(t, data, tick):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()
    labels = ['Pre', 'Post', 'Follow']

    for i, (title, dependency) in enumerate(data.items()):
        x, y = np.array(dependency)
        ax = axs[i % 2]

        if (i // 2) == 2:
            edgecolor = 'red'
            dotcolor = 'orange'
        elif (i // 2) == 1:
            edgecolor = 'purple'
            dotcolor = 'skyblue'
        else:
            edgecolor = 'teal'
            dotcolor = 'lightgreen'

        ax.scatter(x, y, s=0.5, c=dotcolor, alpha=0.5)
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)

        ellipse, center_x, center_y, width, height = confidence_ellipse(
            x, y, ax, n_std=1.645, edgecolor=edgecolor, label=labels[i//2], zorder=0)

        perimeter = calculate_perimeter(width, height)

        ax.set_xlim(tick[0])
        ax.set_ylim(tick[1])
        ax.set_title(title.split('_')[0])
        ax.legend()

        print(title)
        print('center coordinate:', (center_x, center_y))
        print('(width, height):', (width, height))
        print('perimeter:', perimeter)
        print('===' * 20)

    plt.suptitle(t, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

def organize_and_plot_ellipse(base_dir, categories, max_subjects=10):
    import tqdm
    sub_dir = []  # list to hold file paths per subject

    temp = []
    for cat in categories:
        temp.append(glob.glob(os.path.join(base_dir, cat, '*.csv')))

    for i in temp:
        for j in i:
            if len(sub_dir) < max_subjects:
                sub_dir.append([])
            sub_dir[int(os.path.basename(j).split('-')[-2]) - 1].append(j)

    for sub in sub_dir:
        tmp = {}
        minmax = [[0, 0], [0, 0]]  # Used to adjust plot ticks

        for i, c in enumerate(sub):
            vars = ['open_' + categories[i].split('_')[-1], 'close_' + categories[i].split('_')[-1]]
            df = pd.read_csv(c)
            if len(df.index) > 3000:
                df.drop(df.index[0], axis=0, inplace=True)

            tmp[vars[0]] = df[['open_X[mm]', 'open_Y[mm]']].T.to_numpy().tolist()
            tmp[vars[1]] = df[['close_X[mm]', 'close_Y[mm]']].T.to_numpy().tolist()

            for j in range(2):
                for k in range(2):
                    if minmax[j][0] > np.min(tmp[vars[k]][j]):
                        minmax[j][0] = np.min(tmp[vars[k]][j])
                    if minmax[j][1] < np.max(tmp[vars[k]][j]):
                        minmax[j][1] = np.max(tmp[vars[k]][j])
                minmax[j][0] = int(np.floor(minmax[j][0] / 5) * 5)
                minmax[j][1] = int(np.ceil(minmax[j][1] / 5) * 5)

        plot_ellipse(os.path.basename(sub[0])[:7], tmp, minmax)

