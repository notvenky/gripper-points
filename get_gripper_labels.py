import sys
sys.path.append("../")

import pickle
import cv2
import yaml
import imageio

import torch

from p3po.points_class import PointsClass
from pathlib import Path
from tqdm import tqdm

read_from_pickle = False
pickle_path = "/path/to/pickle.pkl"
pickle_image_key = "pixels"

video_paths = ['/home/venky/co-tracker/videos/home_occl.mov',
               '/home/venky/co-tracker/videos/lab.mov',
               '/home/venky/co-tracker/videos/lab2.mov',
               '/home/venky/co-tracker/videos/lab3.mov',
               '/home/venky/co-tracker/videos/lab4.mov']

write_videos = True
subsample = 1

with open("p3po/cfgs/suite/p3po.yaml") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if read_from_pickle:
    examples = pickle.load(open(pickle_path, "rb"))
    num_demos = len(examples['observations'])
else:
    num_demos = len(video_paths)

if write_videos:
    Path(f"{cfg['root_dir']}/processed_data/videos").mkdir(parents=True, exist_ok=True)
    video_write_path = f"{cfg['root_dir']}/processed_data/videos"

points_class = PointsClass(**cfg)
episode_list = []

mark_every = 8
for i in tqdm(range(num_demos), desc="Processing Demos"):
    # Read the frames from the pickle or video, these frames must be in RGB so if reading from a pickle make sure to convert if necessary
    if read_from_pickle:
        frames = examples['observations'][i][pickle_image_key][0::subsample]
    else:
        frames = []
        video = cv2.VideoCapture(video_paths[i])
        subsample_counter = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if subsample_counter % subsample == 0:
                # CV2 reads in BGR format, so we need to convert to RGB
                resized_frame = cv2.resize(frame, (256, 256))
                frames.append(resized_frame[:, :, ::-1])
            subsample_counter += 1
        video.release()

    points_class.add_to_image_list(frames[0])
    points_class.find_semantic_similar_points()
    points_class.track_points(is_first_step=True)
    points_class.track_points(one_frame=(mark_every == 1))

    points_list = []
    points = points_class.get_points()
    points_list.append(points[0])

    if write_videos:
        video_list = []
        image = points_class.plot_image()
        video_list.append(image[0])

    for idx, image in enumerate(tqdm(frames[1:], desc=f"Processing Frames for Demo {i}", leave=False)):
        points_class.add_to_image_list(image)

        if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
            to_add = mark_every - (idx + 1) % mark_every
            if to_add < mark_every:
                for j in range(to_add):
                    points_class.add_to_image_list(image)
            else:
                to_add = 0

            points_class.track_points(one_frame=(mark_every == 1))

            points = points_class.get_points(last_n_frames=mark_every)
            for j in range(mark_every - to_add):
                points_list.append(points[j])

            if write_videos:
                images = points_class.plot_image(last_n_frames=mark_every)
                for j in range(mark_every - to_add):
                    video_list.append(images[j])

    if write_videos:
        imageio.mimsave(f"{video_write_path}/{cfg['task_name']}_{i}.mp4", video_list, fps=30)
    
    episode_list.append(torch.stack(points_list))
    points_class.reset_episode()

final_graph = {}
final_graph['episode_list'] = episode_list
final_graph['subsample'] = subsample
final_graph['pixel_key'] = pickle_image_key
final_graph['pickle_path'] = pickle_path
final_graph['video_paths'] = video_paths
final_graph['cfg'] = cfg

Path(f"{cfg['root_dir']}/processed_data/points").mkdir(parents=True, exist_ok=True)
pickle.dump(final_graph, open(f"{cfg['root_dir']}/processed_data/points/{cfg['task_name']}.pkl", "wb"))
print(f"Saved the processed data to {cfg['root_dir']}/processed_data/points/{cfg['task_name']}.pkl")