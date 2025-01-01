import os
import sys
sys.path.append("../")

import multiprocessing
import pickle
import cv2
import yaml
import imageio
import torch
from pathlib import Path
from tqdm import tqdm

# Disable FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_videos_on_gpu(
    process_id,
    actual_gpu_id,
    video_paths,
    read_from_pickle,
    pickle_path,
    pickle_image_key,
    write_videos,
    subsample,
    cfg
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(actual_gpu_id)
    cfg["device"] = "cuda:0"
    from p3po.points_class import PointsClass

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

    for i in range(num_demos):
        print(f"| {process_id:^4} | {actual_gpu_id:^3} | Demo {i+1}/{num_demos} | Initializing |")

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

        with tqdm(
            total=len(frames) - 1, 
            desc=f"Proc {process_id}, GPU {actual_gpu_id}, Demo {i+1}/{num_demos}",
            position=process_id,
            leave=True
        ) as pbar:
            for idx, image in enumerate(frames[1:]):
                points_class.add_to_image_list(image)

                if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
                    to_add = mark_every - (idx + 1) % mark_every
                    if to_add < mark_every:
                        for _ in range(to_add):
                            points_class.add_to_image_list(image)
                    else:
                        to_add = 0

                    points_class.track_points(one_frame=(mark_every == 1))
                    pts = points_class.get_points(last_n_frames=mark_every)

                    for j in range(mark_every - to_add):
                        points_list.append(pts[j])

                    if write_videos:
                        imgs = points_class.plot_image(last_n_frames=mark_every)
                        for j in range(mark_every - to_add):
                            video_list.append(imgs[j])
                pbar.update(1)

        if write_videos:
            out_name = f"{video_write_path}/{cfg['task_name']}_proc{process_id}_{i}.mp4"
            imageio.mimsave(out_name, video_list, fps=30)

        episode_list.append(torch.stack(points_list))
        points_class.reset_episode()
        print(f"| {process_id:^4} | {actual_gpu_id:^3} | Demo {i+1}/{num_demos} | Done         |")

    final_graph = {
        "episode_list": episode_list,
        "subsample": subsample,
        "pixel_key": pickle_image_key,
        "pickle_path": pickle_path,
        "video_paths": video_paths,
        "cfg": cfg
    }

    Path(f"{cfg['root_dir']}/processed_data/points").mkdir(parents=True, exist_ok=True)
    out_path = f"{cfg['root_dir']}/processed_data/points/{cfg['task_name']}_proc{process_id}.pkl"
    pickle.dump(final_graph, open(out_path, "wb"))
    print(f"| {process_id:^4} | {actual_gpu_id:^3} | Saved to {out_path} |")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    read_from_pickle = False
    pickle_path = "/path/to/pickle.pkl"
    pickle_image_key = "pixels"

    video_paths = [
        '/home/venky/co-tracker/videos/home_occl.mov',
        '/home/venky/co-tracker/videos/lab.mov',
        '/home/venky/co-tracker/videos/lab2.mov',
        '/home/venky/co-tracker/videos/lab3.mov',
        '/home/venky/co-tracker/videos/lab4.mov'
    ]

    write_videos = True
    subsample = 1

    with open("p3po/cfgs/suite/p3po.yaml") as stream:
        cfg = yaml.safe_load(stream)

    mid_idx = len(video_paths) // 2
    video_paths_0 = video_paths[:mid_idx]
    video_paths_1 = video_paths[mid_idx:]

    print("| Proc | GPU | Info                              |")
    print("|------|-----|------------------------------------|")

    p0 = multiprocessing.Process(
        target=process_videos_on_gpu,
        args=(0, 0, video_paths_0, read_from_pickle, pickle_path, pickle_image_key, write_videos, subsample, cfg)
    )
    p1 = multiprocessing.Process(
        target=process_videos_on_gpu,
        args=(1, 1, video_paths_1, read_from_pickle, pickle_path, pickle_image_key, write_videos, subsample, cfg)
    )

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("| Done | All | All processes finished            |")
