import draw_original
import data
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def get_pca_res(samples, res_shape):
    pca = PCA()
    # making long list of the input data
    pca_data = pca.fit_transform(np.array(samples).T.tolist())
    # drawing histogram
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['pc_' + str(x) for x in range(1, len(per_var) + 1)]
    # plot.bar(x=range(1, 11), height=per_var[0:10], tick_label=labels[0:10])
    # getting principal components
    pca_df = pd.DataFrame(pca_data, columns=labels)
    # xy_grids = np.meshgrid(x, y)
    comp1_grid = np.reshape(pca_df.pc_1.to_numpy(), res_shape)
    comp2_grid = np.reshape(pca_df.pc_2.to_numpy(), res_shape)
    return comp1_grid, comp2_grid


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for elem in data.files_Kivu:
        draw_original.draw(data.path_Kivu + elem + data.file_type, 'elem')


