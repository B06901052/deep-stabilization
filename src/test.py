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
from torchvision import transforms
from torchvision.io import read_video, read_video_timestamps, write_video
from torchvision.utils import save_image

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from kornia.contrib import ImageStitcher
from kornia.geometry.transform import warp_perspective, get_perspective_transform

simplefilter("ignore", UserWarning)

# read video
fname = "./deep-stabilization/dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4"

video_frames, _, meta = read_video(fname, end_pts=15, pts_unit="sec")
print(video_frames.shape)
print(meta)

# select frame
img1 = video_frames[:-1].permute(0,3,1,2).float() / 255
img2 = video_frames[1:].permute(0,3,1,2).float() / 255
feature1 = transforms.Resize((1080//8,1920//8))(img1)
feature2 = transforms.Resize((1080//8,1920//8))(img2)

# find match point
# matcher = KF.LocalFeatureMatcher(
#     KF.SIFTFeature(100, device="cuda"),
#     KF.DescriptorMatcher('smnn', 0.8)
#     )
loftr = KF.LoFTR('outdoor').cuda()

t_all = 0
mkpts0 = []
mkpts1 = []
batch_idx = []
for x in range(ceil(len(feature1)/32)):
    f1 = feature1[32*x:32*(x+1)].cuda()
    f2 = feature2[32*x:32*(x+1)].cuda()
    input_dict = {"image0": K.color.rgb_to_grayscale(f1).cuda(), # LofTR works on grayscale images only 
                  "image1": K.color.rgb_to_grayscale(f2).cuda()}

    with torch.no_grad():
        t = time()
        # correspondences = matcher(input_dict)
        correspondences = loftr(input_dict)
        t_all += time()-t
        del f1, f2, input_dict
        
        th = torch.quantile(correspondences["confidence"], 0.8)
        idx = correspondences["confidence"] >= th
        print("keypoints count: ", idx.sum().item())
        mkpts0.append(correspondences['keypoints0'][idx].cpu().numpy())
        mkpts1.append(correspondences['keypoints1'][idx].cpu().numpy())
        batch_idx.append((correspondences['batch_indexes'][idx]+32*x).cpu().numpy())

mkpts0 = np.vstack(mkpts0)
mkpts1 = np.vstack(mkpts1)
batch_idx = np.hstack(batch_idx)

print(mkpts0.shape)
print(batch_idx.shape)
print(np.max(batch_idx))

print("Get matching points: {:5.3f} sec".format(t_all)) 
print("number of keypoints: {:5.2f}".format(len(correspondences["keypoints0"])/32))

q = SimpleQueue()
H = np.eye(3)
count = 0
Hs = []
for x in range(len(feature1)):
    t=time()
    try:
        H1, _ = cv2.findHomography(mkpts0[batch_idx==x], mkpts1[batch_idx==x], cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    except cv2.error as e:
        H1 = np.eye(3)
    try:
        H = H1 @ H
    except:
        pass
    q.put_nowait(H1)
    t_all += time()-t
    count += 1
    if count >= 16:
        try:
            H = H @ LA.inv(q.get())
        except:
            pass
        Hs.append(sqrtm(sqrtm(sqrtm(sqrtm(H)))).real)

frames = []
for img, H in zip(img1, Hs):
    frames.append(cv2.warpAffine(img.permute(1,2,0).numpy(), H[:2], (1920, 1080)))

frames = (torch.from_numpy(np.stack(frames)) * 255).type(torch.uint8)
print(frames.shape)
write_video("test.mp4", frames, fps=meta["video_fps"])

# plt.imsave("res.png", res)
# plt.imsave("img1.png", img1.permute(1,2,0).numpy())
# plt.imsave("img2.png", img2.permute(1,2,0).numpy())