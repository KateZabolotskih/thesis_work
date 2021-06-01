import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plot

class EMM(object):

    def __init__(self, file_name):
        self.X, self.Y, self.Z = self.read_data(file_name)

        # grids = np.meshgrid(self.X, self.Y)
        # fig, ax = plot.subplots()
        # c = ax.contourf(grids[0], grids[1], self.Z, 30, cmap='inferno')
        # ax.set_xlabel("Emission (nm)")
        # ax.set_ylabel("Excitation (nm)")
        # fig.colorbar(c, ax=ax)
        # fig.tight_layout()
        # ax.legend()
        # plot.show()

        self.clean_noise()
        self.Z = (median_filter(self.Z, footprint=np.ones((10, 6)), mode='constant'))

        # grids = np.meshgrid(self.X, self.Y)
        # fig, ax = plot.subplots()
        # c = ax.contourf(grids[0], grids[1], self.Z, 30, cmap='inferno')
        # ax.set_xlabel("Emission (nm)")
        # ax.set_ylabel("Excitation (nm)")
        # fig.colorbar(c, ax=ax)
        # fig.tight_layout()
        # ax.legend()
        # plot.show()

    def read_data(self, file_name):
        with open(file_name, 'r') as f:
            for _ in range(3):
                f.readline()

            y = self.parse_line(f.readline())
            x = []
            z = []

            for line in f:
                numbers = self.parse_line(line)
                x.append(numbers[0])

                if len(numbers) < 52:
                    numbers.extend([0 for _ in range(77 - len(numbers))])

                zs = [numbers[i] for i in range(1, len(numbers))]
                z.append(zs)

            # xx, yy = np.meshgrid(x, y)
            z = np.array(z).T

            return (x, y, z)

    def parse_line(self, line):
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

    def clean_noise(self):
        for i in range(len(self.Y)):
            for j in range(len(self.X)):
                if (1.9 * self.Y[i] <= self.X[j]):
                    self.Z[i][j] = -0.001
                if (self.X[j] <= 1.028 * self.Y[i]):
                    self.Z[i][j] = -0.001
            continue
