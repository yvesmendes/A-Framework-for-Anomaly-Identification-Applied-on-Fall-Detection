#!/usr/bin/env python

import argparse
import json
import shutil

import torch

# from .io import IO
from processor.io import IO
import tools.utils as utils
import os

from scipy.special import expit, softmax,logit
import tensorflow as tf

DIRPATH = os.getcwd()
OPEN_POSE_PATH = DIRPATH[:-5]
print(DIRPATH)
class Demo(IO):

    def start(self,path_video):

        openpose = "/home/yves/projetos/openpose/build/examples/openpose/openpose.bin"

        video_name = path_video.split('/')[-1].split('.')[0]

        output_snippets_dir = './data/openpose_estimation/snippets/{}'.format(video_name)
        output_sequence_dir = './data/openpose_estimation/data'
        output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_name)
        output_result_dir = self.arg.output_dir
        output_result_path = '{}/{}.mp4'.format(output_result_dir, video_name)
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
    
        # pose estimation
        openpose_args = dict(
            video=path_video,
            write_json=output_snippets_dir,
            display=0,
            render_pose=0,
            model_pose='COCO')
        command_line = openpose + ' '
        command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])

        if not os.path.exists(output_snippets_dir):
            shutil.rmtree(output_snippets_dir, ignore_errors=True)
            os.makedirs(output_snippets_dir)

        os.system(command_line)

        video = utils.video.get_video_frames(path_video)
        height, width, _ = video[0].shape
        
        video_info = utils.openpose.json_pack(
            output_snippets_dir, video_name, width, height)

        if not os.path.exists(output_sequence_dir):
            os.makedirs(output_sequence_dir)
        with open(output_sequence_path, 'w') as outfile:
            json.dump(video_info, outfile)
        if len(video_info['data']) == 0:
            print('Can not find pose estimation results.')
            return

        #On the processor.demo_fall_det.py
        pose, _ = utils.video.video_info_parsing(video_info)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()

        self.model.eval()
        output, feature = self.model.extract_feature(data)

        
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)


        #we try different values of the output and feature values
        # f3 = output.sum(dim=3).sum(dim=2).sum(dim=1)
        # f3 = torch.flatten(output[0])[0:500]
        f3 = feature.reshape(-1)
        f3 = f3.detach().cpu().numpy()

        top_5 = []

        return top_5, [], f3

    @staticmethod
    def get_parser(add_help=False):

        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        parser.add_argument('--video',
            # default='./resource/media/skateboarding.mp4',
            default='./resource/media/kkkk.mp4',
            help='Path to video')
        parser.add_argument('--openpose',
            default='3dparty/openpose/build',
            help='Path to openpose')
        parser.add_argument('--output_dir',
            default='./data/demo_result',
            help='Path to save results')
        parser.add_argument('--height',
            default=1080,
            type=int,
            help='Path to save results')
        parser.set_defaults(config="./config/st_gcn/kinetics-skeleton/demo.yaml")
        parser.set_defaults(print_log=False)

        return parser
