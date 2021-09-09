    import numpy as np
import time
import matplotlib.pyplot as plt
from drawnow import drawnow
import cv2 as cv
import torch
from tqdm import tqdm

import config as cfg

import os
import sys



class CloudVisualizer:

    def __init__(self, t_sleep=0.5, point_size=3, ortho=True):
        self.t_sleep = t_sleep
        self.point_size = point_size
        self.ortho = ortho

        self.pcd_src_2d, self.pcd_tgt_2d, self.pcd_est = None, None, None

    def reset(self, pcd_src, pcd_tgt, pcd_est):
        self.pcd_src_2d = self.project(pcd_src[:, :3])
        self.pcd_tgt_2d = self.project(pcd_tgt[:, :3])
        self.pcd_est = pcd_est[:, :3]
        drawnow(self.plot)

    def update(self, new_est):
        self.pcd_est = new_est
        drawnow(self.plot)
        time.sleep(self.t_sleep)

    def plot(self):
        pcd_est_2d = self.project(self.pcd_est)

        # get scale and center
        yx_min, yx_max = np.vstack([self.pcd_src_2d, self.pcd_tgt_2d]).min(axis=0),\
                         np.vstack([self.pcd_src_2d, self.pcd_tgt_2d]).max(axis=0)
        dimensions = yx_max - yx_min
        center = yx_min + dimensions/2

        # get appropriate x/y axes limits
        dimensions = np.array([dimensions.max()*1.1]*2)
        yx_min = center - dimensions/2
        yx_max = center + dimensions/2

        cmap = plt.get_cmap('tab20')
        magenta, gray, cyan = cmap.colors[12], cmap.colors[14], cmap.colors[18]

        plt.scatter(self.pcd_src_2d[:, 0], self.pcd_src_2d[:, 1], c=np.asarray(magenta)[None, :],
                    s=self.point_size, alpha=0.5)
        plt.scatter(self.pcd_tgt_2d[:, 0], self.pcd_tgt_2d[:, 1], c=np.asarray(gray)[None, :],
                    s=self.point_size, alpha=0.5)
        plt.scatter(pcd_est_2d[:, 0], pcd_est_2d[:, 1], c=np.asarray(cyan)[None, :],
                    s=self.point_size, alpha=0.7)
        plt.xlim([yx_min[0], yx_max[0]])
        plt.ylim([yx_min[1], yx_max[1]])
        plt.xticks([])
        plt.yticks([])

    def project(self, points):
        Xs, Ys, Zs = points[:, 0], points[:, 1], points[:, 2] + 3.0  # push back a little
        if self.ortho:
            xs = Xs
            ys = Ys
        else:
            xs = np.divide(Xs, Zs)
            ys = np.divide(Ys, Zs)
        points_2d = np.hstack([xs[:, None], ys[:, None]])
        return points_2d

    def capture(self, path):
        plt.savefig(path, dpi='figure')
