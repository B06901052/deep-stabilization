import cv2
import numpy as np
import numpy.linalg as LA
from queue import SimpleQueue
from math import ceil
from time import time
from importlib import reload
import matplotlib.pyplot as plt
from IPython.display import Video
from warnings import simplefilter
from scipy.linalg import sqrtm

import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.io import read_video, read_video_timestamps, write_video
from torchvision.utils import save_image

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from kornia.contrib import ImageStitcher
from kornia.geometry.transform import warp_perspective, get_perspective_transform
from kornia.geometry.homography import find_homography_dlt_iterated
from kornia.geometry.transform import HomographyWarper

import utils
simplefilter("ignore", UserWarning)

# read video
fname = "./deep-stabilization/dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4"

video_frames, _, meta = read_video(fname, end_pts=5, pts_unit="sec")
print(video_frames.shape)
print(meta)


def GetIntrinsics(focal_length, offset, width, height):
    intrinsics = [
        [float(focal_length), 0.0, 0.5*(width-1)+offset[0]*width], 
        [0.0, float(focal_length), 0.5*(height-1)+offset[1]*height], 
        [0.0, 0.0, 1.0]
        ]
    return np.array(intrinsics)


K_metrix = GetIntrinsics(1920/1.27,(0,0),1920,1080)

# select frame
img1 = video_frames.permute(0,3,1,2).float() / 255
feature1 = transforms.Resize((1080//4,1920//4))(img1)

# find match point
# loftr = KF.LocalFeatureMatcher(
#     KF.SIFTFeature(1000, device="cuda"),
#     KF.DescriptorMatcher('smnn', 0.8)
#     )
loftr = KF.LoFTR('outdoor').cuda()

t_all = 0
mkpts0 = []
mkpts1 = []
batch_idx = []
window = 8
for x in range(ceil(len(feature1)/window)):
    f1 = feature1[window*x:window*(x+1)]
    f1 = K.color.rgb_to_grayscale(f1).cuda()
    input_dict = {"image0": f1, # LofTR works on grayscale images only 
                  "image1": f1[window//2:window//2+1].expand(f1.shape)}
    
    with torch.no_grad():
        t = time()
        # correspondences = matcher(input_dict)
        with autocast():
            correspondences = loftr(input_dict)
        t_all += time()-t
        del f1, input_dict
        
        th = torch.quantile(correspondences["confidence"], 0.0)
        idx = correspondences["confidence"] >= th
        print("keypoints count: ", idx.sum().item())
        mkpts0.append(correspondences['keypoints0'][idx].cpu())
        mkpts1.append(correspondences['keypoints1'][idx].cpu())
        batch_idx.append((correspondences['batch_indexes'][idx]+window*x).cpu())

mkpts0 = torch.cat(mkpts0, dim=0)
mkpts1 = torch.cat(mkpts1, dim=0)
batch_idx = torch.cat(batch_idx, dim=0)

print(mkpts0.shape)
print(batch_idx.shape)
print(batch_idx)

print("Get matching points: {:5.3f} sec".format(t_all)) 
print("number of keypoints: {:5.2f}".format(len(correspondences["keypoints0"])/window))

q = SimpleQueue()
H = np.eye(3)
count = 0
Hs = []
for x in range(len(feature1)):
    t=time()
    if (x-window//2) % window:
        try:
            # H1, _ = cv2.findHomography(mkpts0[batch_idx==x], mkpts1[batch_idx==x], cv2.USAC_MAGSAC, 0.5, confidence=0.999, maxIters=100)
            H1 = find_homography_dlt_iterated(mkpts0[batch_idx==x], mkpts1[batch_idx==x])
            Hs.append(H1)
        except:
            Hs.append(torch.eye(3))
    else:
        Hs.append(torch.eye(3))
    #     a = cv2.decomposeHomographyMat(H1,K_metrix)# num, Rs, Ts, Ns
    #     for x in a:
    #         print(x)
    #         print("\n\n")
    #     input()
    # except cv2.error as e:
    #     H1 = np.eye(3)
    # try:
    #     H = H1 @ H
    # except:
    #     pass
    # q.put_nowait(H1)
    # t_all += time()-t
    # count += 1
    # if count >= 16:
    #     try:
    #         H = H @ LA.inv(q.get())
    #     except:
    #         pass
    #     Hs.append(sqrtm(sqrtm(sqrtm(sqrtm(H)))).real)

frames = []
for i, (img, H) in enumerate(zip(img1, Hs)):
    try:
        warper = HomographyWarper(1080, 1920, mode='bilinear', padding_mode='zeros', normalized_coordinates=True, align_corners=False)
        frames.append(warper(img, H))
    except:
        frames.append(img)

print(torch.cat(frames).shape)
frames = (torch.cat(frames).view(-1,3,1080,1920).permute(0,2,3,1) * 255).type(torch.uint8)
print(frames.shape)
write_video("test.mp4", frames, fps=meta["video_fps"])

# plt.imsave("res.png", res)
# plt.imsave("img1.png", img1.permute(1,2,0).numpy())
# plt.imsave("img2.png", img2.permute(1,2,0).numpy())