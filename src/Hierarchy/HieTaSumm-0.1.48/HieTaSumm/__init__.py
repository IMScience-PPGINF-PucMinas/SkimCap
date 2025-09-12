from .Summary import Summary
import os
import json
import cv2 as cv
import numpy as np
from datetime import timedelta
from pathlib import Path
import logging


class HieTaSumm:
    def __init__(self, **kwargs):
        self.f_score_epochs = []
        self.cusa_epochs = []
        self.cuse_epochs = []
        self.cov_epochs = []
        PACKAGEDIR = Path(__file__).parent.absolute()
        my_file = PACKAGEDIR / 'options.json'

        with open(my_file, 'r') as json_file:
            default_data = json.load(json_file)

        # Retrieve parameters from kwargs or default configuration
        dataset_videos = kwargs.get('video_path')
        dataset_features = kwargs.get('features_path')  # new parameter for precomputed features
        dataset_frames = kwargs.get('summary_path')
        if not dataset_frames:
            dataset_frames = default_data['summary_path']
        percent = int(kwargs.get('percent', default_data['percent']))
        alpha = int(kwargs.get('alpha', default_data['alpha']))
        rate = int(kwargs.get('rate', default_data['rate']))
        time = int(kwargs.get('time', default_data['time']))
        hierarchy = kwargs.get('hierarchy', default_data['hierarchy'])
        selected_model = kwargs.get('selected_model', default_data['selected_model'])
        is_binary = kwargs.get('is_binary', default_data['is_binary'])
        gen_summary_method = kwargs.get('gen_summary_method', default_data['gen_summary_method'])
        
        # Define output directories for precomputed features processing.
        # (Adjust these paths as needed.)
        out_skim_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalskimTESTE_100'
        out_index_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalindexesTESTE_100'
        
        # Ensure the summary directory exists (create intermediate directories if needed)
        if not os.path.isdir(dataset_frames):
            os.makedirs(dataset_frames, exist_ok=True)
        
        # If pre-computed features are provided, use them for summarization.
        if dataset_features and os.path.exists(dataset_features):
            # Process only files ending with '_resnet.npy'
            video_list = [f for f in os.listdir(dataset_features) if f.endswith('_resnet.npy')]
            video_list.sort()  # guarantee order
            for video in video_list:
                if video != '.ipynb_checkpoints':
                    # Build full path for the feature file.
                    full_video_path = os.path.join(dataset_features, video)
                    # Derive base name (e.g. "Pp5DCsgaALg_resnet")
                    base_name = os.path.splitext(video)[0]
                    # Define expected output paths.
                    output_resnet_path = os.path.join(out_skim_dir, base_name + ".npy")
                    output_txt_path = os.path.join(out_index_dir, base_name + ".txt")
                    base_bn_name = base_name.replace('_resnet', '_bn')
                    output_bn_path = os.path.join(out_skim_dir, base_bn_name + ".npy")
                    
                    # If all outputs exist, skip processing this file.
                    if os.path.exists(output_resnet_path) and os.path.exists(output_txt_path) and os.path.exists(output_bn_path):
                        #print(f"Skipping {video} as it is already fully processed.")
                        continue
                    
                    # Otherwise, process the file.
                    fscore, cusa, cuse, cov = self.hierarchical_summarization(
                        dataset_features, full_video_path, rate, time, percent, alpha,
                        gen_summary_method, is_binary, hierarchy, selected_model)
                    self.f_score_epochs.append(fscore)
                    self.cusa_epochs.append(cusa)
                    self.cuse_epochs.append(cuse)
                    self.cov_epochs.append(cov)
        # Otherwise, if a video path is provided, extract frames first.
        elif dataset_videos and os.path.exists(dataset_videos):
            video_list = [f for f in os.listdir(dataset_videos) if f.endswith('_resnet.npy')]
            video_list.sort()  # guarantee order
            for video in video_list:
                print("------------------------")
                print("{}/{}/".format(dataset_videos, video))
                self.frame_extractor("{}/{}".format(dataset_videos, video), rate, dataset_frames)
            
            if os.path.exists(dataset_frames):
                video_list = [f for f in os.listdir(dataset_videos) if f.endswith('_resnet.npy')]
                video_list.sort()  # guarantee order
                for video in video_list:
                    if video != '.ipynb_checkpoints':
                        fscore, cusa, cuse, cov = self.hierarchical_summarization(
                            dataset_frames, video, rate, time, percent, alpha,
                            gen_summary_method, is_binary, hierarchy, selected_model)
                        self.f_score_epochs.append(fscore)
                        self.cusa_epochs.append(cusa)
                        self.cuse_epochs.append(cuse)
                        self.cov_epochs.append(cov)
        else:
            print("No valid video_path or features_path provided.")
        
        print("----------------------")
        print(f'The avg fscore of all videos was: {np.mean(self.f_score_epochs)}')
        print(f'The avg CUSa of all videos was: {np.mean(self.cusa_epochs)}')
        print(f'The avg CUSe of all videos was: {np.mean(self.cuse_epochs)}')
        print(f'The avg Cov of all videos was: {np.mean(self.cov_epochs)}')


    def frame_extractor(self, video_file, rate, frames):
        SAVING_FRAMES_PER_SECOND = 1 / rate
        filename, _ = os.path.splitext(video_file)
        video = os.path.join(frames, filename.split('/')[-1])
        frames_folder = os.path.join(video, "frames")
        if not os.path.isdir(video):
            os.mkdir(video)
        if not os.path.isdir(frames_folder):
            os.mkdir(frames_folder)
            cap = cv.VideoCapture(video_file)
            fps = cap.get(cv.CAP_PROP_FPS)
            saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
            saving_frames_durations = self.get_saving_frames_durations(cap, saving_frames_per_second)
            count = 0
            frame_number = 1
            while True:
                is_read, frame = cap.read()
                if not is_read:
                    break
                frame_duration = count / fps
                try:
                    closest_duration = saving_frames_durations[0]
                except IndexError:
                    break
                if frame_duration >= closest_duration:
                    number = str(frame_number).zfill(6)
                    frame_number += 1
                    cv.imwrite(os.path.join(frames_folder, f"{number}.jpg"), frame)
                    try:
                        saving_frames_durations.pop(0)
                    except IndexError:
                        pass
                count += 1

    def get_saving_frames_durations(self, cap, saving_fps):
        s = []
        clip_duration = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / cap.get(cv.CAP_PROP_FPS))
        for i in np.arange(0, clip_duration, saving_fps):
            s.append(i)
        return s

    def hierarchical_summarization(self, dataset_frames, video, rate, time, percent, alpha, gen_summary_method, is_binary, hierarchy, selected_model):
        summ = Summary(dataset_frames, video, rate, time, hierarchy, selected_model, is_binary, percent, alpha, gen_summary_method)
        return summ.fscore, summ.mean_cusa, summ.mean_cuse, summ.cov_value

if __name__ == "__main__":
    HieTaSumm()
    # Example call using video path:
    # HieTaSumm(
    #     video_path='/content/videos',
    #     percent=15,
    #     rate=30,
    #     time=1,
    #     alpha=25,
    #     selected_model='resnet50',
    #     summary_path='/content/skim-8-vgg',
    #     gen_summary_method={"method": "group_central_frames"},
    #     hierarchy="watershed_hierarchy_by_area"
    # )
    # Or, if you have precomputed video features:
    # HieTaSumm(
    #     features_path='/path/to/features',
    #     percent=15,
    #     rate=30,
    #     time=1,
    #     alpha=25,
    #     selected_model='resnet50',
    #     summary_path='/content/skim-8-vgg',
    #     gen_summary_method={"method": "group_central_frames"},
    #     hierarchy="watershed_hierarchy_by_area"
    # )
