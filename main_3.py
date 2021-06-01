import numpy as np
import matplotlib.pyplot as plt
from EMM import EMM
from scipy.stats import pearsonr
import data


def get_AA_sums(file):
    emm = EMM(file)
    AA = {
        'C': {'x1': 420, 'x2': 480, 'y1': 320, 'y2': 350},
        'A': {'x1': 380, 'x2': 480, 'y1': 250, 'y2': 260},
        'M': {'x1': 380, 'x2': 420, 'y1': 310, 'y2': 320},
        'B': {'x1': 300, 'x2': 320, 'y1': 270, 'y2': 280},
        'T': {'x1': 320, 'x2': 350, 'y1': 270, 'y2': 280}
    }
    sums_AA = {
        'C': 0,
        'A': 0,
        'M': 0,
        'B': 0,
        'T': 0
    }

    for i in range(len(emm.Y)):
        for j in range(len(emm.X)):
            for aa in AA.keys():
                if AA[aa]['x1'] <= emm.X[j] <= AA[aa]['x2'] and AA[aa]['y1'] <= emm.Y[i] <= AA[aa]['y2']:
                    sums_AA[aa] += emm.Z[i][j]

    return sums_AA


# def get_correlation_coeff(X, Y):
#     return pearsonr(X, Y)[0]


# def get_m1(M, C, A):
#     return M / (C + A)


# def get_m2(M, B, T):
#     return M / (B + T)


# def get_K(C, A, B, T):
#     return (C + A) / (B + T)


# def boxplot(datas, labels, image_name):
#     fig, ax = plt.subplots()
#     ax.boxplot(datas, vert=False, labels=labels)
#
#     fig.savefig(image_name)

if __name__ == '__main__':
    K_Kivu, M_Kivu = [], []
    K_North, M_North = [], []
    K_Baikal, M_Baikal = [], []

    for file_Africa in data.files_Kivu:
        sums = get_AA_sums(data.path_Kivu + file_Africa + data.file_type)
        C, A, M, B, T = sums['C'], sums['A'], sums['M'], sums['B'], sums['T']
        K = (C + A) / (B + T)
        K_Kivu.append(K)
        M_Kivu.append(M)

    for file_North in data.files_North:
        sums = get_AA_sums(data.path_North + file_North + data.file_type)
        C, A, M, B, T = sums['C'], sums['A'], sums['M'], sums['B'], sums['T']
        K = (C + A) / (B + T)
        K_North.append(K)
        M_North.append(M)

    for file_Baikal in data.files_Baikal:
        sums = get_AA_sums(data.path_Baikal + file_Baikal + data.file_type)
        C, A, M, B, T = sums['C'], sums['A'], sums['M'], sums['B'], sums['T']
        K = (C + A) / (B + T)
        K_Baikal.append(K)
        M_Baikal.append(M)

    fig1, axes1 = plt.subplots(1, len(data.areas), figsize=(12, 5), squeeze=False)

    ax = axes1[np.unravel_index(0, shape=axes1.shape)]
    ax.set_title(data.areas[0], fontsize=16, fontweight="bold")
    ax.scatter(M_Kivu, K_Kivu, color='g')
    ax.set_ylabel("K")
    ax.set_xlabel("M")

    ax = axes1[np.unravel_index(1, shape=axes1.shape)]
    ax.set_title(data.areas[1], fontsize=16, fontweight="bold")
    ax.scatter(M_North, K_North, color='r')
    ax.set_ylabel("K")
    ax.set_xlabel("M")

    ax = axes1[np.unravel_index(2, shape=axes1.shape)]
    ax.set_title(data.areas[2], fontsize=16, fontweight="bold")
    ax.scatter(M_Baikal, K_Baikal, color='b')
    ax.set_ylabel("K")
    ax.set_xlabel("M")

    fig2, ax = plt.subplots()
    ax.set_title("Common picture", fontsize=16, fontweight="bold")
    ax.set_xlabel("M")
    ax.set_ylabel("K")
    ax.scatter(M_Kivu, K_Kivu, color='g')
    ax.scatter(M_North, K_North)
    ax.scatter(M_Baikal, K_Baikal)
    ax.legend(data.areas)
    plt.show()

    fig1.savefig("pictures/" + "fields")
    fig2.savefig("pictures/" + "common_picture")