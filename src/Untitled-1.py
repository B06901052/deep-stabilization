# %%
import cv2
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
from IPython.display import Video

import torch
from torchvision import transforms
from torchvision.io import read_video, read_video_timestamps

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from kornia.contrib import ImageStitcher
from kornia.geometry.transform import warp_perspective, get_perspective_transform

import utils
def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

# %%
fname = "../deep-stabilization/dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4"

# %%
video_frames, audio_frames, meta = read_video(fname, end_pts=100, pts_unit="sec")
print(meta)
print("video size: ", video_frames.shape)
print("audio size: ", audio_frames.shape)

# %%
# utils.show_frames(video_frames[:100:10], 2, 5, (30,16))

# %%
img1 = video_frames[0:1].permute(0,3,1,2).float() / 255
img2 = video_frames[100:101].permute(0,3,1,2).float() / 255

print(img1.shape)

feature1 = transforms.CenterCrop((270*3,480*3))(img1)
feature2 = transforms.CenterCrop((270*3,480*3))(img2)

feature1 = torch.cat(transforms.FiveCrop(256)(feature1))
feature2 = torch.cat(transforms.FiveCrop(256)(feature2))

print(feature1.shape)

# K.color.rgb_to_grayscale(img1).shape
utils.show_frame(feature1[3].permute(1,2,0))

# %%
matcher2 = KF.LocalFeatureMatcher(
    KF.SIFTFeature(2000, device="cuda"),
    KF.DescriptorMatcher('smnn', 0.9)
    )

input_dict = {"image0": K.color.rgb_to_grayscale(feature1).cuda(), # LofTR works on grayscale images only 
              "image1": K.color.rgb_to_grayscale(feature2).cuda()}

with torch.no_grad():
    correspondences = matcher2(input_dict)
    del input_dict["image0"], input_dict["image1"]
    
for k,v in correspondences.items():
    print (k)

print(len(correspondences["keypoints0"]))

# %%
# for x in range(5):
#     idx = torch.topk(correspondences["confidence"][correspondences["batch_indexes"]==x], 100).indices
#     print((correspondences["keypoints0"][correspondences["batch_indexes"]==x][idx] - correspondences["keypoints1"][correspondences["batch_indexes"]==x][idx]).mean(dim=0))
# print("\n\n\n")
# for x in range(5):
#     idx = torch.topk(correspondences["confidence"][correspondences["batch_indexes"]==x], 150).indices
#     print((correspondences["keypoints0"][correspondences["batch_indexes"]==x][idx] - correspondences["keypoints1"][correspondences["batch_indexes"]==x][idx]).mean(dim=0))
# print("\n\n\n")
tmp = []
for x in range(5):
    tmp.append((correspondences["keypoints0"][correspondences["batch_indexes"]==x] - correspondences["keypoints1"][correspondences["batch_indexes"]==x]).median(dim=0)[0])
    print(tmp[-1])

# %%
src = torch.Tensor([
    [135*1+128, 240*1+128],# 左上
    [135*1+128, 240*7-128],# 右上
    [135*7-128, 240*1+128],# 左下
    [135*7-128, 240*7-128] # 右下
]).cuda()

dst = torch.vstack(tmp[:4]) + src

# %%
img1[0].permute(1,2,0).shape

# %%
res = cv2.warpAffine(img1[0].permute(1,2,0).numpy(), H[:2], (1080, 1920))

# %%
utils.show_frame(torch.from_numpy(res))

# %%
H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)

# %%
b

# %%
print(src)
print(dst)
b = get_perspective_transform(src.unsqueeze(0), dst.unsqueeze(0))

out = warp_perspective(img1.cuda(), b, (1080,1920)).cpu()
outt = torch.where(out == 0.0, img2, out)
utils.show_frame(outt[0].permute(1,2,0))

# %%
utils.show_frame(img1[0].permute(1,2,0))

# %%
utils.show_frame(img2[0].permute(1,2,0))

# %%
out = warp_perspective(img1.cuda(), torch.from_numpy(H).cuda().unsqueeze(0).float(), (1080,1920)).cpu()
outtt = torch.where(out == 0.0, img2, out)
utils.show_frame(outtt[0].permute(1,2,0))

# %%
for k,v in correspondences.items():
    print (k)

# %%
th = torch.quantile(correspondences["confidence"], 0.0)
idx = correspondences["confidence"] > th
print(idx.sum())

# %%
mkpts0 = correspondences['keypoints0'][idx].cpu().numpy()
mkpts1 = correspondences['keypoints1'][idx].cpu().numpy()
H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

# %%
H

# %%
draw_LAF_matches(
    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
    torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={'inlier_color': (0.2, 1, 0.2),
               'tentative_color': None, 
               'feature_color': (0.2, 0.5, 1), 'vertical': False})

# %%
from kornia.geometry.transform import get_perspective_transform, warp_perspective
idx = torch.topk(correspondences["confidence"], 12).indices
# idx = torch.randperm(20)
src = correspondences["keypoints0"][idx[:4]].unsqueeze(0)
dst = correspondences["keypoints1"][idx[:4]].unsqueeze(0)
a = get_perspective_transform(src, dst)
src = correspondences["keypoints0"][idx[2:6]].unsqueeze(0)
dst = correspondences["keypoints1"][idx[2:6]].unsqueeze(0)
b = get_perspective_transform(src, dst)

out = warp_perspective(img1.cuda(), (a+b)/2, (1080//4,1920//4)).cpu()
outt = torch.where(out < 0.0, img2, out)
utils.show_frame(outt[0].permute(1,2,0))

# %%
# Import numpy and OpenCV
import numpy as np
import cv2# Read input video

fname = "../deep-stabilization/dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4"
cap = cv2.VideoCapture(fname)
 
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# Define the codec for output video
 
# Set up output video
fps = 30
print(w, h)

# Read first frame
_, prev = cap.read()
 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# prev_gray = (prev_gray&192)|((prev_gray&32)<<1)

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)
log = []
for i in range(n_frames-2):
    log.append([])
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=400,
                                     qualityLevel=0.3,
                                     minDistance=20,
                                     blockSize=9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    prev_pts = cv2.cornerSubPix( prev_gray, prev_pts, (5,5), (-1,1), criteria )
    # Read next frame
    success, curr = cap.read()
    if not success:
        break
 
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # curr_gray = (curr_gray&192)|((curr_gray&32)<<1)
 
    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
 
    # Sanity check
    assert prev_pts.shape == curr_pts.shape
 
    # Filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
 
    #Find transformation matrix
    retval, inliers = cv2.estimateAffine2D(prev_pts, curr_pts)
 
    # Extract traslation
    dx = retval[0][2]
    dy = retval[1][2]
 
    # Extract rotation angle
    da = np.arctan2(retval[1,0], retval[0,0])
    log[-1].append(len(inliers))
    log[-1].append(np.arctan2(retval[0,1], retval[1,1]))
 
    # Store transformation
    transforms[i] = [dx,dy,da]
 
    # Move to next frame
    prev_gray = curr_gray
 
    print("Frame: {:03d}/{:3d} -  Tracked points : {:3d}".format(i, n_frames, len(prev_pts)), end="\r", flush=True)
  
# Compute trajectory using cumulative sum of transformations
print("transforms: ", len(transforms))
trajectory = np.cumsum(transforms, axis=0)

# %%
np.array(log)

# %%
print(np.arctan2(retval[1,0], retval[0,0]))
print(np.arctan2(retval[0,1], retval[1,1]))
print(retval)

# %%
from scipy.signal import savgol_filter
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return savgol_filter(curve, window_size, 3)
    # return curve_smoothed

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def smooth(trajectory, SMOOTHING_RADIUS=60):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
    return smoothed_trajectory

# %%
# Calculate difference in smoothed_trajectory and trajectory
smoothed_trajectory = smooth(trajectory)
difference = smoothed_trajectory - trajectory
# median = np.median(np.abs(difference))
# new_trajectory = trajectory.copy()
# for i, d in enumerate(difference):
#     if d[0]>median:
#         new_trajectory[i] = smoothed_trajectory[i]
    
# smoothed_trajectory = smooth(new_trajectory)
# difference = smoothed_trajectory - trajectory
# # Calculate newer transformation array
transforms_smooth = transforms + difference


# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frames=[]
# Write n_frames-1 transformed frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../video_out.mp4', fourcc, fps, (w, h))
for i in range(n_frames-2):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]
 
    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy
 
    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame.astype(np.float64)/255, m, (w,h))

    # Fix border artifacts
    # frame_stabilized = fixBorder(frame_stabilized)

    # Write the frame to the file
    frame_out = cv2.hconcat([frame.astype(np.float64)/255, frame_stabilized])

    # If the image is too big, resize it.
    if frame_out.shape[1] > 1920:
        frame_out = cv2.resize(frame_out, (frame_out.shape[1]//2, frame_out.shape[0]));
 
    frames.append(frame_out)
    out.write((frame_out*255).astype(np.uint8))

out.release()

# %%
from torchvision.io import read_video, read_video_timestamps

# %%
from IPython.display import Video
Video("../video_out.mp4", width=960, height=540)

# %%
Video("../stable_video.avi", width=960, height=540)


