import cv2
from inference.interact.interactive_utils import *
import torch
import numpy as np
import os
from PIL import Image

from model.network import XMem
from inference.inference_core import InferenceCore

import matplotlib.pyplot as plt

from progressbar import progressbar

torch.set_grad_enabled(False)

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def images_to_video(image_folder, video_name, fps):
    images = [img for img in sorted(os.listdir(image_folder),key=natural_sort_key) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def segment(first_frame_path, video_path, fps):

    network = XMem(config, 'saves\XMem.pth').eval().to(device)
    torch.cuda.empty_cache()

    video_name = video_path
    mask_name = first_frame_path

    mask = np.array(Image.open(mask_name))
    print(np.unique(mask))
    num_objects = len(np.unique(mask)) - 1
    # mask.show()

    processor = InferenceCore(network, config=config)
    processor.set_all_labels(range(1, num_objects + 1))  # consecutive labels
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(video_name)


    # # Desired width and the aspect ratio 9:11
    # desired_width = 480
    # aspect_ratio = 16 / 9

    # # Calculate the desired height maintaining the 9:11 aspect ratio
    # desired_height = int(desired_width * aspect_ratio)


    ## you could play with the second arg here, it may somehow be related to the fps?? not sure
    # cap.set(cv2.CAP_PROP_FPS, 1)
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = 2000
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # You can change these two numbers
    frames_to_propagate = total_frames
    ## change this would skip frames
    visualize_every = 3

    current_frame_index = 0

    with torch.cuda.amp.autocast(enabled=True):
        while (cap.isOpened()):
            # load frame-by-frame
            _, frame = cap.read()
            if frame is None or current_frame_index > frames_to_propagate:
                break
            
            # convert numpy array to pytorch tensor format
            frame_torch, _ = image_to_torch(frame, device=device)
            if current_frame_index == 0:
                # initialize with the mask
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(device)
                # the background mask is not fed into the model
                prediction = processor.step(frame_torch, mask_torch[1:])
            
            else:
                # propagate only
                ## skip frames not needed 
                if (current_frame_index % visualize_every) != 0:
                    current_frame_index += 1
                    #print("skip frame {}".format(current_frame_index))
                    continue
                prediction = processor.step(frame_torch)

            # argmax, convert to numpy
            prediction = torch_prob_to_numpy_mask(prediction)

            if current_frame_index % visualize_every == 0:
                black_background_frame = np.zeros_like(frame)
                visualization = overlay_davis(black_background_frame, prediction)
                Image.fromarray(visualization).save('./saving_frame/{0:06d}.jpg'.format(current_frame_index))
            
            current_frame_index += 1
            # if(current_frame_index % 100 == 0):
            #     print('Number of Images saved: {}'.format(current_frame_index))

    images_to_video('./saving_frame','new'+video_name,fps)

    