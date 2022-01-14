import cv2
import torch
import numpy as np
import numpy.linalg as LA
from scipy.signal import savgol_filter
from kornia.geometry.conversions import rotation_matrix_to_quaternion, QuaternionCoeffOrder, normalize_homography, normalize_quaternion
from gyro import (
    QuaternionProduct,
    QuaternionReciprocal,
)
from gyro.gyro_function import GetIntrinsics


def homography_to_quaternion(homography, w, h):
    n_homo = normalize_homography(
        torch.from_numpy(homography), (h, w), (h, w)).numpy()
    intrinsic = np.array([
        [w / 1.27, 0.0, 0.5 * (w - 1)],
        [0.0, w / 1.27, 0.5 * (h - 1)],
        [0.0, 0.0, 1.0]
    ])
    tmp = []
    for h in n_homo:
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(h, intrinsic)
        Rs = np.stack(Rs)
        i = np.argmin(np.abs(Rs[:, 2, 2] - 1))
        tmp.append(Rs[i])
    tmp = rotation_matrix_to_quaternion(torch.from_numpy(
        np.stack(tmp)), order=QuaternionCoeffOrder.WXYZ)[:, [1, 2, 3, 0]].numpy()
    return tmp


def process_frames(frames, w, h):
    n_frames = len(frames)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames - 1, 3), np.float32)
    homography = []
    for i in range(n_frames - 2):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           #  maxCorners=1000,
                                           #  qualityLevel=0.2,
                                           #  minDistance=10,
                                           #  blockSize=5)
                                           maxCorners=400,
                                           qualityLevel=0.3,
                                           minDistance=30,
                                           blockSize=9)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        prev_pts = cv2.cornerSubPix(
            prev_gray, prev_pts, (5, 5), (-1, 1), criteria)

        curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        # retval, inliers = cv2.estimateAffine2D(prev_pts, curr_pts)
        retval = cv2.findHomography(prev_pts, curr_pts)[0]
        homography.append(retval)

        # Extract traslation
        dx = retval[0][2]
        dy = retval[1][2]
        # Extract rotation angle
        da = np.arctan2(retval[1, 0], retval[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: {:03d}/{:3d} -  Tracked points : {:3d}".format(i,
              n_frames, len(prev_pts)), end="\r", flush=True)

    # Compute trajectory using cumulative sum of transformations
    print("transforms: ", len(transforms))
    trajectory = np.cumsum(transforms, axis=0)
    homography = np.stack(homography)
    diff_quats = homography_to_quaternion(homography, w, h)
    diff_quats = normalize_quaternion(torch.from_numpy(diff_quats)).numpy()
    quats = np.zeros((diff_quats.shape[0] + 1, 4))
    quats[0, :] = np.array([0, 0, 0, 1])
    for i in range(1, diff_quats.shape[0] + 1):
        quats[i] = QuaternionProduct(diff_quats[i - 1], quats[i - 1])
        quats[i] /= LA.norm(quats[i])

    for i in range(diff_quats.shape[0] - 1, 20, -1):
        quats[i] = QuaternionProduct(
            quats[i], QuaternionReciprocal(quats[i - 20]))
        quats[i] /= LA.norm(quats[i])
    np.savetxt("quaternion.txt", quats)
    return trajectory, transforms, homography, quats


def movingAverage(curve, window_size, future_frames=2, mode="savgol"):
    if mode == "avg":
        f = np.ones(window_size) / window_size
        curve_pad = np.lib.pad(
            curve, (window_size - (future_frames + 1), future_frames), 'edge')
        return np.convolve(curve_pad, f, mode='valid')
    elif mode == "savgol":
        return savgol_filter(curve, window_size, 3)


def smooth(trajectory, window_size=31):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(
            trajectory[:, i], window_size)

    return smoothed_trajectory


def smooth_transform(transforms, trajectory):
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    return transforms + difference
