"""
refenrence from: https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
"""
import cv2
import numpy as np
from scipy.signal import savgol_filter

fname = "./deep-stabilization/dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4"
# fname = "./deep-stabilization/dvs/test/stabilzation/s_114_outdoor_running_trail_daytime_stab.mp4"
cap = cv2.VideoCapture(fname)
 
# Get metadata
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# Read first frame
_, prev = cap.read()
 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# prev_gray = (prev_gray&192)|((prev_gray&32)<<1)

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)
for i in range(n_frames-2):
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                    #  maxCorners=1000,
                                    #  qualityLevel=0.2,
                                    #  minDistance=10,
                                    #  blockSize=5)
                                     maxCorners=400,
                                     qualityLevel=0.3,
                                     minDistance=30,
                                     blockSize=9)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prev_pts = cv2.cornerSubPix( prev_gray, prev_pts, (5,5), (-1,1), criteria )
    # Read next frame
    success, curr = cap.read()
    if not success:
        break
 
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
 
    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
 
    # Filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
 
    #Find transformation matrix
    retval, inliers = cv2.estimateAffine2D(prev_pts, curr_pts)
    # retval = cv2.findHomography(prev_pts, curr_pts)[0]

    # Extract traslation and rotation angle
    dx = retval[0][2]
    dy = retval[1][2]
    da = np.arctan2(retval[1,0], retval[0,0])
 
    # Store transformation
    transforms[i] = [dx,dy,da]
 
    # Move to next frame
    prev_gray = curr_gray
 
    print("Frame: {:03d}/{:3d} -  Tracked points : {:3d}".format(i, n_frames, len(prev_pts)), end="\r", flush=True)
  
# Compute trajectory using cumulative sum of transformations
print("transforms: ", len(transforms))
trajectory = np.cumsum(transforms, axis=0)


def movingAverage(curve, window_size):
    assert window_size%2, "window_size should be odd"
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (window_size//2, window_size//2), 'edge')
    # Apply convolution
    return np.convolve(curve_pad, f, mode='valid')
    return savgol_filter(curve, window_size, 3)

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

def smooth(trajectory, SMOOTHING_RADIUS=30):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], SMOOTHING_RADIUS)
 
    return smoothed_trajectory

# Calculate difference in smoothed_trajectory and trajectory
difference = smooth(trajectory) - trajectory
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
    # frame_stabilized = cv2.warpPerspective(frame.astype(np.float64)/255, m, (w,h))

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