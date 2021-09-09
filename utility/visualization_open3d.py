import numpy as np
import time
import matplotlib.pyplot as plt
from drawnow import drawnow
import cv2 as cv
import torch
from tqdm import tqdm
import copy
import open3d as o3d

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
        source_temp = copy.deepcopy(self.pcd_src_3d)
        target_temp = copy.deepcopy(self.pcd_tgt_3d)
        est_temp = copy.deepcopy(self.pcd_est)
        
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(source_temp)
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(target_temp)
        pcd_est = o3d.geometry.PointCloud()
        pcd_est.points = o3d.utility.Vector3dVector(est_temp)

        pcd_src.paint_uniform_color([0, 0, 1])
        pcd_tgt.paint_uniform_color([1, 0, 0])
        pcd_est.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([pcd_src, pcd_tgt, pcd_est])

    def capture(self, path):
        plt.savefig(path, dpi='figure')
