import math
import numpy as np
import cv2 as cv


# Utility function to draw orientations over an image
def draw_orientations(fingerprint, orientations, strengths, mask, scale=3, step=8, border=0):
    if strengths is None:
        strengths = np.ones_like(orientations)
    h, w = fingerprint.shape
    sf = cv.resize(fingerprint, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)
    res = cv.cvtColor(sf, cv.COLOR_GRAY2BGR)
    d = (scale // 2) + 1
    sd = (step + 1) // 2
    c = np.round(np.cos(orientations) * strengths * d * sd).astype(int)
    s = np.round(-np.sin(orientations) * strengths * d * sd).astype(int)  # minus for the direction of the y axis
    thickness = 1 + scale // 5
    for y in range(border, h - border, step):
        for x in range(border, w - border, step):
            if mask is None or mask[y, x] != 0:
                ox, oy = c[y, x], s[y, x]
                cv.line(res, (d + x * scale - ox, d + y * scale - oy), (d + x * scale + ox, d + y * scale + oy),
                        (0, 255, 0), thickness, cv.LINE_AA)
    return res


# Utility function to draw a set of minutiae over an image
def draw_minutiae(fingerprint, minutiae, termination_color=(255, 0, 0), bifurcation_color=(0, 0, 255)):
    res = cv.cvtColor(fingerprint, cv.COLOR_GRAY2BGR)

    for x, y, t, *d in minutiae:
        color = termination_color if t else bifurcation_color
        if len(d) == 0:
            cv.drawMarker(res, (x, y), color, cv.MARKER_CROSS, 8)
        else:
            d = d[0]
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv.circle(res, (x, y), 3, color, 1, cv.LINE_AA)
            cv.line(res, (x, y), (x + ox, y - oy), color, 1, cv.LINE_AA)
    return res


# Utility function to generate gabor filter kernels

_sigma_conv = (3.0 / 2.0) / ((6 * math.log(10)) ** 0.5)


# sigma is adjusted according to the ridge period, so that the filter does not contain more than three effective peaks
def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period


def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))
    if p % 2 == 0:
        p += 1
    return (p, p)


def gabor_kernel(period, orientation):
    f = cv.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi / 2 - orientation, period, gamma=1, psi=0)
    f /= f.sum()
    f -= f.mean()
    return f


# Utility functions for minutiae
def angle_abs_difference(a, b):
    return math.pi - abs(abs(a - b) - math.pi)


def angle_mean(a, b):
    return math.atan2((math.sin(a) + math.sin(b)) / 2, ((math.cos(a) + math.cos(b)) / 2))


# Utility functions for MCC
def draw_minutiae_and_cylinder(fingerprint, origin_cell_coords, minutiae, values, i, show_cylinder=True):
    def _compute_actual_cylinder_coordinates(x, y, t, d):
        c, s = math.cos(d), math.sin(d)
        rot = np.array([[c, s], [-s, c]])
        return (rot @ origin_cell_coords.T + np.array([x, y])[:, np.newaxis]).T

    res = draw_minutiae(fingerprint, minutiae)
    if show_cylinder:
        for v, (cx, cy) in zip(values[i], _compute_actual_cylinder_coordinates(*minutiae[i])):
            cv.circle(res, (int(round(cx)), int(round(cy))), 3, (0, int(round(v * 255)), 0), 1, cv.LINE_AA)
    return res


def draw_match_pairs(f1, m1, v1, f2, m2, v2, cells_coords, pairs, i, show_cylinders=True):
    # nd = _current_parameters.ND
    h1, w1 = f1.shape
    h2, w2 = f2.shape
    p1, p2 = pairs
    res = np.full((max(h1, h2), w1 + w2, 3), 255, np.uint8)
    res[:h1, :w1] = draw_minutiae_and_cylinder(f1, cells_coords, m1, v1, p1[i], show_cylinders)
    res[:h2, w1:w1 + w2] = draw_minutiae_and_cylinder(f2, cells_coords, m2, v2, p2[i], show_cylinders)
    for k, (i1, i2) in enumerate(zip(p1, p2)):
        (x1, y1, *_), (x2, y2, *_) = m1[i1], m2[i2]
        cv.line(res, (int(x1), int(y1)), (w1 + int(x2), int(y2)), (0, 0, 255) if k != i else (0, 255, 255), 1,
                cv.LINE_AA)
    return res


cn_filter = np.array([[1, 2, 4],
                      [128, 0, 8],
                      [64, 32, 16]
                      ])


def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))


def compute_next_ridge_following_directions(previous_direction, values):
    next_positions = np.argwhere(values != 0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        # There is a previous direction: return all the next directions, sorted according to the distance from it,
        #                                except the direction, if any, that corresponds to the previous position
        next_positions.sort(key=lambda d: 4 - abs(abs(d - previous_direction) - 4))
        if next_positions[-1] == (
                previous_direction + 4) % 8:  # the direction of the previous position is the opposite one
            next_positions = next_positions[:-1]  # removes it
    return next_positions


r2 = 2 ** 0.5
xy_steps = [(-1, -1, r2), (0, -1, 1), (1, -1, r2), (1, 0, 1), (1, 1, r2), (0, 1, 1), (-1, 1, r2), (-1, 0, 1)]


def follow_ridge_and_compute_angle(x, y, cn, cn_values, nd_lut, d=8):
    px, py = x, y
    length = 0.0
    while length < 20:  # max length followed
        next_directions = nd_lut[cn_values[py, px]][d]
        if len(next_directions) == 0:
            break
        # Need to check ALL possible next directions
        if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
            break  # another minutia found: we stop here
        # Only the first direction has to be followed
        d = next_directions[0]
        ox, oy, l = xy_steps[d]
        px += ox;
        py += oy;
        length += l
    # check if the minimum length for a valid direction has been reached
    return math.atan2(-py + y, px - x) if length >= 10 else None


mcc_radius = 70
mcc_size = 16

g_r = 2 * mcc_radius / mcc_size
x_r = np.arange(mcc_size) * g_r - (mcc_size / 2) * g_r + g_r / 2.
y_r = x_r[..., np.newaxis]
iy, ix = np.nonzero(x_r ** 2 + y_r ** 2 <= mcc_radius ** 2)
ref_cell_coords = np.column_stack((x_r[ix], x_r[iy]))

mcc_sigma_s = 7.0
mcc_tau_psi = 400.0
mcc_mu_psi = 1e-2


def Gs(t_sqr):
    """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
    return np.exp(-0.5 * t_sqr / (mcc_sigma_s ** 2)) / (math.tau ** 0.5 * mcc_sigma_s)


def Psi(v):
    """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
    return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))
