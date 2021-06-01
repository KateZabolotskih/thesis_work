import matplotlib.patches as patches
from EMM import EMM
import numpy as np
import matplotlib.pyplot as plt

color = 'inferno'

def add_rects(ax):
    rect1 = patches.Rectangle((420, 320),
                              60,
                              30,
                              linewidth=1,
                              edgecolor='r',
                              facecolor='none',
                              label='C')
    ax.add_patch(rect1)

    rect2 = patches.Rectangle((380, 250),
                              100,
                              10,
                              linewidth=1,
                              edgecolor='g',
                              facecolor='none',
                              label='A')
    ax.add_patch(rect2)

    rect3 = patches.Rectangle((380, 310),
                              40,
                              10,
                              linewidth=1,
                              edgecolor='b',
                              facecolor='none',
                              label='M')
    ax.add_patch(rect3)

    rect4 = patches.Rectangle((300, 270),
                              20,
                              10,
                              linewidth=1,
                              edgecolor='y',
                              facecolor='none',
                              label='B')
    ax.add_patch(rect4)

    rect5 = patches.Rectangle((320, 270),
                              30,
                              10,
                              linewidth=1,
                              edgecolor='w',
                              facecolor='none',
                              label='T')
    ax.add_patch(rect5)

def get_local_max(data):
    local_max = []
    x, y = data.shape
    rad = 10

    for i in range(x):
        for j in range(y):
            l = i - rad if i - rad > 0 else 0
            r = i + rad if i + rad < x else x
            u = j - rad if j - rad > 0 else 0
            d = j + rad if j + rad < y else y
            slice = data[l:r, u:d]

            if data[i][j] == np.max(slice) and data[i][j] > 50:
                local_max.append([j, i, data[i][j]])

    return np.array(local_max).T

def draw(file_name, image_name):
    emm = EMM(file_name)
    loc = get_local_max(emm.Z)
    loc[0] = [emm.X[int(i)] for i in loc[0]]
    loc[1] = [emm.Y[int(i)] for i in loc[1]]
    m_x, m_y = np.meshgrid(emm.X, emm.Y)
    fig, ax = plt.subplots()
    add_rects(ax)
    c = ax.contourf(m_x, m_y, emm.Z, 30, cmap=color)
    ax.scatter(loc[0],
               loc[1],
               color='r',
               marker="X",
               label="Local maximums")

    ax.set_title(file_name)
    ax.set_xlabel("Emission (nm)")
    ax.set_ylabel("Excitation (nm)")
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    ax.legend()
    plt.show()

    fig.savefig(image_name)