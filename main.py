import glob
import math
import numpy as np
import cv2 as cv
import sklearn
import pandas
import matplotlib.pyplot as plt
from ipywidgets import interact
import fingerprint_functions as ff
import torch

# 指定图片所在文件夹路径
folder_path = 'DB1_B'

# 获取所有图片路径
image_paths = glob.glob(folder_path + '/*.tif')

for image_path in image_paths:
    # 读取图片
    fingerprint = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    size = fingerprint.shape
    # print(size[0],size[1])
    cv.imshow('Fingerprint', fingerprint)
    # gblured = cv.GaussianBlur(fingerprint, (5, 5), 1.3)
    # cv.imshow('gblured', gblured)

    # sobel计算梯度
    gx = cv.Sobel(fingerprint, cv.CV_32F, 1, 0)
    gy = cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
    # cv.imshow('Gx', gx)
    # cv.imshow('Gy', gy)

    # 计算梯度大小
    gx2, gy2 = gx ** 2, gy ** 2
    gm = np.sqrt(gx2 + gy2)
    # cv.imshow('Gradient magnitude', gm)
    sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize=False)
    # cv.imshow('Integral of the gradient magnitude', sum_gm)

    # 分割
    threshold = sum_gm.max() * 0.2
    mask = cv.threshold(sum_gm, threshold, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
    cv.imshow('Mask', mask)
    cv.imshow('Segment', cv.merge((mask, fingerprint, fingerprint)))

    # 计算梯度方向，脊的方向即为梯度的垂直方向，以W窗内的平均梯度方向为准
    W = (23, 23)  # 窗
    gxx = cv.boxFilter(gx2, -1, W, normalize=False)
    gyy = cv.boxFilter(gy2, -1, W, normalize=False)
    gxy = cv.boxFilter(gx * gy, -1, W, normalize=False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy
    orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2
    sum_gxx_gyy = gxx + gyy
    strengths = np.divide(cv.sqrt((gxx_gyy ** 2 + gxy2 ** 2)), sum_gxx_gyy, out=np.zeros_like(gxx),
                          where=sum_gxx_gyy != 0)
    cv.imshow('Orientation image', ff.draw_orientations(fingerprint, orientations, strengths, mask, 1, 16))

    # 计算中心
    mm = cv.moments(mask)
    mask_x = round(mm['m10'] / mm['m00'])
    mask_y = round(mm['m01'] / mm['m00'])
    # print(mask_x, mask_y)
    # cv.circle(mask, (mask_x, mask_y), 3, (0, 255, 0), 1, cv.LINE_AA)
    # cv.imshow('mask', mask)
    # cv.waitKey(0)

    # 中心附近采样
    region_width = size[0] // 8
    region_height = size[1] // 8
    region = fingerprint[(mask_y - region_height):(mask_y + region_height),
             (mask_x - region_width):(mask_x + region_width)]
    # cv.rectangle(fingerprint, (mask_x - region_width, mask_y - region_height),
    #              (mask_x + region_width, mask_y + region_height), (0, 255, 0), 3)
    # cv.imshow('ROI', fingerprint)
    # print(region_height, region_width)
    cv.imshow('Region', region)

    # 估计中心附近频率
    smoothed = cv.blur(region, (region_height // 8, region_width // 8), -1)  # 滤波
    xs = np.sum(smoothed, 1)
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
    distances = local_maxima[1:] - local_maxima[:-1]
    ridge_period = np.average(distances)  # 脊线平均距离
    # print(ridge_period)

    # Gabor滤波
    or_count = 8
    gabor_bank = [ff.gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)]  # 构建滤波核
    # for i in range(0, or_count):
    #     cv.imshow('Gabor_Bank', gabor_bank[i])
    #     cv.waitKey(0)
    nf = 255 - fingerprint
    image_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
    # for i in range(0, or_count):
    #     cv.imshow('Filters', image_filtered[i])
    #     cv.waitKey(0)
    y_coords, x_coords = np.indices(fingerprint.shape)
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    filtered = image_filtered[orientation_idx, y_coords, x_coords]
    enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
    cv.imshow('Enhanced', enhanced)

    # 检测细节点位置
    _, ridge_lines = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow('Ridge_Lines', ridge_lines)
    skeleton = cv.ximgproc.thinning(ridge_lines, thinningType=cv.ximgproc.THINNING_GUOHALL)
    cv.imshow('Ske', skeleton)
    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([ff.compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)
    skeleton01 = np.where(skeleton != 0, 1, 0).astype(np.uint8)
    cn_values = cv.filter2D(skeleton01, -1, ff.cn_filter, borderType=cv.BORDER_CONSTANT)
    cn = cv.LUT(cn_values, cn_lut)
    cn[skeleton == 0] = 0
    minutiae = [(x, y, cn[y, x] == 1) for y, x in zip(*np.where(np.isin(cn, [1, 3])))]
    # cv.imshow('Minutiae_Raw', ff.draw_minutiae(fingerprint, minutiae))
    # cv.imshow('Minutiae_Ske', ff.draw_minutiae(skeleton, minutiae))
    mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,
                    1:-1]
    standard_distance = np.maximum(region_height // 4, region_width // 4)
    # print(standard_distance)
    filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]] > standard_distance, minutiae))  # 滤去里mask边界过近的点
    cv.imshow('fMinutiae_Raw', ff.draw_minutiae(fingerprint, filtered_minutiae))
    cv.imshow('fMinutiae_Ske', ff.draw_minutiae(skeleton, filtered_minutiae))

    # 给出特征点处的方向
    nd_lut = [[ff.compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]
    valid_minutiae = []
    for x, y, term in filtered_minutiae:
        d = None
        if term:  # termination: simply follow and compute the direction
            d = ff.follow_ridge_and_compute_angle(x, y, cn, cn_values, nd_lut)
        else:  # bifurcation: follow each of the three branches
            dirs = nd_lut[cn_values[y, x]][8]  # 8 means: no previous direction
            if len(dirs) == 3:  # only if there are exactly three branches
                angles = [ff.follow_ridge_and_compute_angle(x + ff.xy_steps[d][0], y + ff.xy_steps[d][1], cn, cn_values,
                                                            nd_lut, d) for d in dirs]
                if all(a is not None for a in angles):
                    a1, a2 = min(((angles[i], angles[(i + 1) % 3]) for i in range(3)),
                                 key=lambda t: ff.angle_abs_difference(t[0], t[1]))
                    d = ff.angle_mean(a1, a2)
        if d is not None:
            valid_minutiae.append((x, y, term, d))

    cv.imshow('Minutiae_Valid', ff.draw_minutiae(skeleton, valid_minutiae))

    # 去除旋转的影响
    xyd = np.array([(ff.x_r, ff.y_r, d) for ff.x_r, ff.y_r, _, d in valid_minutiae])
    d_cos, d_sin = np.cos(xyd[:, 2]).reshape((-1, 1, 1)), np.sin(xyd[:, 2]).reshape((-1, 1, 1))
    rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])
    xy = xyd[:, :2]
    cell_coords = np.transpose(rot @ ff.ref_cell_coords.T + xy[:, :, np.newaxis], [0, 2, 1])
    dists = np.sum((cell_coords[:, :, np.newaxis, :] - xy) ** 2, -1)
    cs = ff.Gs(dists)
    diag_indices = np.arange(cs.shape[0])
    cs[diag_indices, :, diag_indices] = 0
    local_structures = ff.Psi(np.sum(cs, -1))

    cv.waitKey(0)
    # print(f"""Fingerprint image: {fingerprint.shape[1]}x{fingerprint.shape[0]} pixels
    # Minutiae: {len(valid_minutiae)}
    # Local structures: {local_structures.shape}""")
    # f1, m1, ls1 = fingerprint, valid_minutiae, local_structures
    # ofn = 'samples/sample_1_2'
    # f2, (m2, ls2) = cv.imread(f'{ofn}.png', cv.IMREAD_GRAYSCALE), np.load(f'{ofn}.npz', allow_pickle=True).values()
    # dists = np.sqrt(np.sum((ls1[:, np.newaxis, :] - ls2) ** 2, -1))
    # dists /= (np.sqrt(np.sum(ls1 ** 2, 1))[:, np.newaxis] + np.sqrt(np.sum(ls2 ** 2, 1)))
    # r
    # # Select the num_p pairs with the smallest distances (LSS technique)
    # num_p = 5  # For simplicity: a fixed number of pairs
    # pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
    # score = 1 - np.mean(dists[pairs[0], pairs[1]])  # See eq. (23) in MCC paper
    # print(f'Comparison score: {score:.2f}')
# 释放窗口
cv.destroyAllWindows()




