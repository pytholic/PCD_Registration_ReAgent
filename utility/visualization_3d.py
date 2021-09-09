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

import plotly
import plotly.graph_objs as go

class CloudVisualizer:

    def __init__(self, t_sleep=0.5, point_size=3, ortho=True):
        self.t_sleep = t_sleep
        self.point_size = point_size
        self.ortho = ortho

        self.pcd_src_3d, self.pcd_tgt_3d, self.pcd_est = None, None, None

    def reset(self, pcd_src, pcd_tgt, pcd_est):
        self.pcd_src_3d = pcd_src
        self.pcd_tgt_3d = pcd_tgt
        self.pcd_est = pcd_est
        drawnow(self.plot)

    def update(self, new_est):
        self.pcd_est = new_est
        drawnow(self.plot)
        time.sleep(self.t_sleep)

    def plot(self):

        # get scale and center
        zyx_min, zyx_max = np.vstack([self.pcd_src_3d, self.pcd_tgt_3d]).min(axis=0),\
                         np.vstack([self.pcd_src_3d, self.pcd_tgt_3d]).max(axis=0)
        dimensions = zyx_max - zyx_min
        center = zyx_min + dimensions/3

        # get appropriate x/y axes limits
        dimensions = np.array([dimensions.max()*1.1]*3)
        zyx_min = center - dimensions/3
        zyx_max = center + dimensions/3

        cmap = plt.get_cmap('tab20')
        magenta, gray, cyan = cmap.colors[12], cmap.colors[14], cmap.colors[18]

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(self.pcd_src_3d[:, 0], self.pcd_src_3d[:, 1], self.pcd_src_3d[:, 2], c=np.asarray(magenta)[None, :],
                    s=self.point_size, alpha=0.5)
        ax.scatter3D(self.pcd_tgt_3d[:, 0], self.pcd_tgt_3d[:, 1], self.pcd_tgt_3d[:, 2], c=np.asarray(gray)[None, :],
                    s=self.point_size, alpha=0.5)
        ax.scatter3D(self.pcd_est[:, 0], self.pcd_est[:, 1], self.pcd_est[:, 2], c=np.asarray(cyan)[None, :],
                    s=self.point_size, alpha=0.7)
        ax.set_xlim(zyx_min[0], zyx_max[0])
        ax.set_ylim(zyx_min[1], zyx_max[1])
        ax.set_zlim(-10,10)


#     def project(self, points):
#         Xs, Ys, Zs = points[:, 0], points[:, 1], points[:, 2] + 3.0  # push back a little
#         if self.ortho:
#             xs = Xs
#             ys = Ys
#         else:
#             xs = np.divide(Xs, Zs)
#             ys = np.divide(Ys, Zs)
#         points_2d = np.hstack([xs[:, None], ys[:, None]])
#         return points_2d

    def capture(self, path):
        plt.savefig(path, dpi='figure')
