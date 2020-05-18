import numpy as np


class Density:

    def __init__(self, img, sample_size):

        # image shape is used for reconstruct vector to a 2d image
        self.img = img
        self.img_shape = img.shape
        self.rec_img = np.zeros(self.img_shape)
        self.sample_size = sample_size
        # cdf
        cdf = np.cumsum(self.img.reshape(1, -1))
        self.cdf = cdf / np.max(cdf)

    def get_idx(self):

        # search the corresponding idx(pixel position) of random uniform distributed samples in img_cdf
        # so that we could put value 1 into that position
        samples = np.random.uniform(0, 1, self.sample_size)
        idx = np.searchsorted(self.cdf, samples)
        return idx

    def get_position(self, idx):
        # x is the remainder of idx divided by the width of img(shape[1])
        # y is the quotient of idx divided by the width of img(shape[1])
        # x describes which column, y describes which row
        x_pos = idx % self.img_shape[1]
        x_pos = x_pos.reshape(-1, 1)
        y_pos = idx // self.img_shape[1]
        y_pos = y_pos.reshape(-1, 1)
        coord = np.concatenate((x_pos, y_pos), axis=1)
        return coord

    def generate_coordinate(self, height, width):

        x, y = np.mgrid[0:height:1, 0:width:1]
        x_coord = y.reshape(-1, 1)
        y_coord = x.reshape(-1, 1)
        coord = np.concatenate((x_coord, y_coord), axis=1)
        return coord




