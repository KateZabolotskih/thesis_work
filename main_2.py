import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import data
from sklearn.decomposition import PCA
import pandas as pd

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

def get_local_max(data, koeff, X, Y):
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

            if data[i][j] == np.max(slice) and data[i][j] > koeff:
                local_max.append([X[j], Y[i], data[i][j]])

    return np.array(local_max).T

def parse_line(line):
    line = line.replace('\n', '').split('\t')
    data = []
    for number in line:
        if number == '':
            data.append(0)
        elif not number.isdigit() and not "," in number:
            continue
        else:
            data.append(float(number.replace(',', '.')))

    return data

def del_ryley(Z, X, Y):
    for i in range(len(Y)):
        for j in range (len(X)):
            if X[j] <= 1.026 * Y[i]:
                Z[i][j] = -0.001


        continue
    return Z

def del_roman(Z, X, Y):
    for i in range(len(Y)):
        for j in range(len(X)):
            if 1.9 * Y[i] <= X[j]:
                Z[i][j] = -0.001
        continue
    return Z

# reading sample out of the file
def read_data(filename, line=None):
    with open(filename) as file:
        lines = file.readlines()
        y = parse_line(lines[3])
        x = []
        # making matrix for the input data
        matrix = []
        for i in range(4, len(lines)):
            numbers = parse_line(lines[i])
            x.append(numbers[0])
            if len(numbers) < 52:
                numbers.extend([0 for _ in range(77 - len(numbers))])
            zs = [numbers[j] for j in range(1, len(numbers))]
            matrix.append(zs)
        z = np.array(matrix).T
        return x, y, z

# getting samples out of files list
def get_data(files_list, dir):
    z_samples = []
    for filename in files_list:
        X, Y, z_matrix = read_data(dir + filename + data.file_type)
        # smoothing the input samples and making common samples list
        z_samples.append(np.ravel(del_roman(del_ryley(z_matrix, X, Y), X, Y)))
        # for getting raw samples
        #z_samples.append(np.ravel(z_matrix))
    return X, Y, z_samples

# PCA decomposition of the samples represented with res_shape shape
def get_pca_res(samples, res_shape, region):
    pca = PCA()
    # making long list of the input data
    pca_data = pca.fit_transform(np.array(samples).T.tolist())
    # drawing histogram
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['pc_' + str(x) for x in range(1, len(per_var) + 1)]
    # plot.bar(x=range(1, 11), height=per_var[0:10], tick_label=labels[0:10])
    plot.bar(labels, per_var)
    plot.title(region)
    # getting principal components
    pca_df = pd.DataFrame(pca_data, columns=labels)
    # xy_grids = np.meshgrid(x, y)
    comp1_grid = np.reshape(pca_df.pc_1.to_numpy(), res_shape)
    comp2_grid = np.reshape(pca_df.pc_2.to_numpy(), res_shape)
    return comp1_grid, comp2_grid

# use to draw 3 dimensional graph
def draw_3d_plot(x_mesh, y_mesh, z_mesh, name):
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='inferno')
    ax.set_xlabel('Emission, nm.')
    ax.set_ylabel('Excitation, nm.')
    ax.set_zlabel('Intencity')
    ax.set_xlabel(name)


# use to draw contour graph
def draw_contour(x_mesh, y_mesh, z_mesh, name):
    fig = plot.figure()
    ax = fig.add_subplot()
    cs = ax.contourf(x_mesh, y_mesh, z_mesh, cmap='inferno', levels=60)
    add_rects(ax)
    ax.set_xlabel('Emission, nm.')
    ax.set_ylabel('Excitation, nm.')
    ax.set_xlabel(name)
    plot.colorbar(cs)

if __name__ == '__main__':
    path = data.path_Kivu
    files = data.files_Kivu

    # listing all .txt files in the directory
    files_list = []
    for elem in files:
        files_list.append(elem)
    # files_list = filter(lambda str: str.endswith('.txt'), os.listdir('Permafrost'))
    # getting data from them
    x, y, z_samples = get_data(files_list, path)
    # getting x and y grids for graphics
    xy_grids = np.meshgrid(x, y)
    # getting 1'st and 2'nd principal components
    comp1_grid, comp2_grid = get_pca_res(z_samples, xy_grids[0].shape, path)
    # making graphs
    print(get_local_max(comp1_grid, 200, x, y))
    draw_3d_plot(xy_grids[0], xy_grids[1], comp1_grid, "1 comp")
    draw_contour(xy_grids[0], xy_grids[1], comp1_grid, "1 comp")
    print('get_local_max(comp2_grid)')
    print(get_local_max(comp2_grid, 50, x, y))
    draw_3d_plot(xy_grids[0], xy_grids[1], comp2_grid, "2 comp")
    draw_contour(xy_grids[0], xy_grids[1], comp2_grid, "2 comp")
    plot.show()