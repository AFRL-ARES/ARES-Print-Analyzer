#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/ares_print_analyzer/results_visualization/polar_histogram.py
# Project: ARES-Print-Analyzer
# Created Date: Wednesday, November 5th 2025, 1:24:07 pm
# Author(s): Arthur W. N. Sloan
# -----
# MIT License
# 
# Copyright (c) 2025 AFRL-ARES
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
###

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
from scipy.spatial.distance import pdist
import cv2 as cv

matplotlib.use('Agg')
def plot_polar_representation(contour:np.ndarray,
                              point_id:int=0,
                              angle_bins:int=12,
                              dist_bins:int=5) -> np.ndarray:
    
    points = np.squeeze(contour)
    points *= np.array([-1,1]) # flip left and right to put coordinates in image space rather than numpy index space for nicer plotting
    all_distances = pdist(points)
    # Reusing the logic from the get histogram part here
    log_distances = np.log(np.maximum(1e-5, all_distances))

    maxLogDistance = np.max(log_distances)
    minLogDistance = np.max([0.0, np.min(log_distances)]) # Ensure min is not negative

    radialBound = maxLogDistance + (maxLogDistance - minLogDistance) * 0.01
    intervalSize = radialBound / float(dist_bins)
    angleSize = np.pi / float(angle_bins)/2

    # 3. Use NumPy broadcasting to calculate all pairwise differences at once.
    # This is the core of the vectorization for the main histogram calculation.
    # points[:, np.newaxis, :] -> shape (N, 1, 2)
    # points[np.newaxis, :, :] -> shape (1, N, 2)
    # The difference is an (N, N, 2) array of all (dx, dy) pairs.
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]

    # 4. Calculate angles and distances for all pairs simultan`eously.
    # These operations now act on entire (N, N) matrices.
    angles = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])
    angles[angles<0] += 2 * np.pi # convert angles from [-pi, pi] range to [0, 2*pi]
    distances = np.sqrt(diffs[:, :, 0]**2 + diffs[:, :, 1]**2)
    log_distances =  np.log(np.maximum(1e-5, distances))


    r_ticks = np.exp(intervalSize*np.arange(1,6))
    theta_tics = np.linspace(0,2*np.pi,12,endpoint=False)

    # Get the data corresponding to the points we want
    r = distances[point_id,:]
    r = np.roll(r,-point_id)
    theta = angles[point_id,:]
    theta = np.roll(theta,-point_id)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(4,4))
    ax.plot(theta,r,ls='none',marker='o',markersize=1,color='tab:red')
    ax.set_rticks(r_ticks)  
    ax.set_yticklabels([])
    ax.set_xticks(theta_tics)
    ax.set_thetalim(0, 2*np.pi)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Turn the plot into some openCV compatible image data by reading the rgba buffer

    fig.set_dpi(300)
    fig.tight_layout()
    fig.canvas.draw()

    b = fig.axes[0].get_window_extent()
    img = np.array(fig.canvas.buffer_rgba())
    # img = img[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    plt.close('all')

    #return the image
    return img


def plot_histogram_colormap(hist:np.ndarray,
                            point_id:int=0,
                            angle_bins:int=12,
                            dist_bins:int=5) -> tuple[np.ndarray,np.ndarray]:
    
    # plots the histogram data of one point in the contour
    # provides both a 2d heatmatp histogram and a 1d histogram

    h = hist[point_id,:]
    h2 = h.reshape(dist_bins,angle_bins)

    # 2d heatmap histogram
    fig, ax = plt.subplots(figsize=(4,4))
    a = ax.pcolor(h2,
              edgecolors='k',linewidth=1.5,
              cmap='viridis')
    ax.set_ylabel(r'$log(r)$',fontsize=12,fontweight='bold')
    ax.set_xlabel(r'$\theta$',fontsize=12,fontweight='bold')
    ax.tick_params(axis='both',
                   which='both',
                   top=False,
                   bottom=False,
                   left=False,
                   right=False,
                   labeltop=False,
                   labelleft=False,
                   labelbottom=False)
    ax.set_aspect(2.4)
    cbar = fig.colorbar(a,ax=ax,fraction=0.046, pad=0.04)
    cbar.set_ticks([])

    # Turn the plot into some openCV compatible image data by reading the rgba buffer
    fig.set_dpi(300)
    fig.canvas.draw()
    # b = fig.axes[0].get_window_extent()
    img1 = np.array(fig.canvas.buffer_rgba())
    # img1 = img1[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
    img1 = cv.cvtColor(img1, cv.COLOR_RGBA2BGR)


    # 1d histogram
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(np.arange(0,60),h)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel(r'Bin #',fontsize=12,fontweight='bold')
    # Turn the plot into some openCV compatible image data by reading the rgba buffer
    fig.set_dpi(300)
    fig.tight_layout()

    fig.canvas.draw()
    # b = fig.axes[0].get_window_extent()
    img2 = np.array(fig.canvas.buffer_rgba())
    # img2 = img2[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
    img2 = cv.cvtColor(img2, cv.COLOR_RGBA2BGR)
    plt.close('all')

    return img1, img2

def plot_summary_histogram(hist1:np.ndarray,
                         hist2:np.ndarray,
                         row_inds:np.ndarray,
                         col_inds:np.ndarray,
                         angle_bins:int=12,
                         dist_bins:int=5):
    # this function takes the histogram data, and the score assignment data 
    #get the sizes of the histograms so we can pair up the row/columun values propperly
    size1 = hist1.shape[0]
    size2 = hist2.shape[0]
    worker_size= row_inds[-1]+1

    if size1 == worker_size:
        worker = hist1
        task = hist2
    elif size2 == worker_size:
        worker = hist2
        task=hist1

    # Sorts and prunes both histograms such that each indiex corresponds to the most similar point.
    worker = worker[row_inds,:]
    task = task[col_inds,:]

    sum_ = worker+task
    diff = worker-task

    disp_hist = diff**2/(sum_ + 1e-16)
    mean_diff = np.mean(disp_hist,axis=0)

    props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
    text_str = 'Difference Score: {:.2f}'.format(np.sum(mean_diff)/2)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(np.arange(0,60),mean_diff)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
    ax.set_xlabel(r'Bin #',fontsize=12,fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Turn the plot into some openCV compatible image data by reading the rgba buffer

    fig.set_dpi(300)
    fig.tight_layout()

    fig.canvas.draw()
    # b = fig.axes[0].get_window_extent()
    img = np.array(fig.canvas.buffer_rgba())
    # img = img[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
    img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
    plt.close('all')
    return img
