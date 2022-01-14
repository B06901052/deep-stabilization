import torch
import matplotlib.pyplot as plt


def show_frame(frame: torch.tensor):
    fig = plt.figure(figsize = (16,12)) # create a 5 x 5 figure 
    ax = fig.add_subplot(111)
    ax.imshow(frame.numpy(), interpolation='bicubic')
    
def show_frames(frames: torch.tensor, nrows=10, ncols=10, figsize=(20,16)):
    fig, ax = plt.subplots(nrows, ncols, figsize = figsize)
    for i, frame in enumerate(frames):
        ax[i//ncols][i%ncols].imshow(frame.numpy(), interpolation='bicubic')
        ax[i//ncols][i%ncols].axis('off')
    
    plt.tight_layout()