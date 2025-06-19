import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import time

def gs_iteration_modified(size_real, weight, interval, phi, e, w0):

    def GS_algorithm(phase, weight):
        size_ = [(part - 1) / 2 for part in size_part]
        # creating a meshgrid with the center being (0, 0)
        X, Y = np.meshgrid(np.arange(-size_[0], size_[0] + 1), np.arange(-size_[1], size_[1] + 1))
        # print (X.shape, Y.shape) => (640, 640)
        A0 = np.exp(-((X.T) ** 2) / (1000 ** 2) - (Y.T ** 2) / (1000 ** 2)) * np.exp(1j * phase)
        B0 = fftshift(fft2(A0, (size_part[0], size_part[1])))
        at, _ = Multibeam(np.sqrt(weight))
        D = (at) * np.exp(1j * np.angle(B0))
        E = ifft2(ifftshift(D))
        Output = np.angle(E)
        return Output

    def Multibeam(weight):
        row, column = weight.shape
        single_r = (interval - 1) / 2
        single_x, single_y = np.meshgrid(np.arange(-single_r, single_r + 1), np.arange(-single_r, single_r + 1))
        singlepattern = np.exp(-2 * (single_x ** 2 + single_y ** 2) / w0 ** 2)
        Multi = np.tile(singlepattern, (row, column))
        for i in range(row):
            for ii in range(column):
                Multi[i * interval: (i + 1) * interval, ii * interval: (ii + 1) * interval] *= weight[i, ii]
        Multi_x, Multi_y = Multi.shape
        Multipattern = np.zeros((int(size_part[0]), int(size_part[1])))

        position = np.array([[np.floor(size_part[0] / 2) - np.floor(Multi_x / 2), np.floor(size_part[0] / 2) + np.floor(Multi_x / 2)], [np.floor(size_part[1] / 2) - np.floor(Multi_y / 2), np.floor(size_part[1] / 2) + np.floor(Multi_y / 2)]])
        Multipattern[int(position[0, 0]):int(position[0, 1]), int(position[1, 0]):int(position[1, 1])] = Multi
        if e > 0:
            print("check")
            Multipattern[int(position[0, 0]) - singlepattern.shape[0] : int(position[0, 0]), int(Multipattern.shape[0] / 2) - int(singlepattern.shape[0] / 2) : int(Multipattern.shape[0] / 2) - int(singlepattern.shape[0] / 2) + singlepattern.shape[0] ] = singlepattern * e
            # print out the numerical value of the multipattern
            print(Multipattern)
        return Multipattern, position

    if size_real[0] > 500:
        ratio = 2
    else:
        ratio = 4

    size_part = [int(1 * size_real[0] * ratio), int(1 * size_real[0] * ratio)]
    padnum = [(sp - sr) / 2 for sp, sr in zip(size_part, size_real)]

    real_rect = np.array([[padnum[0] + 1, padnum[0] + size_real[0]], [padnum[1] + 1, padnum[1] + size_real[1]]])
    real_rect = np.array([[padnum[0], padnum[0] + size_real[0]], [padnum[1], padnum[1] + size_real[1]]])

    # real_rect = np.array([[241, 400], [276, 365]])

    phase = GS_algorithm(phi, weight)
    Phase_f = phase[int(real_rect[0, 0]) :int(real_rect[0, 1]), int(real_rect[1, 0]):int(real_rect[1, 1])]
    Phase_n = np.mod(Phase_f, 2 * np.pi)
    Image_SLM = Phase_n.T

    return Image_SLM, phase