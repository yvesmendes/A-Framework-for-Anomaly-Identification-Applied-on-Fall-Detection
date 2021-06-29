#!\\usr\\bin\\env python
import os
import sys
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

# from .io import IO
from processor.io import IO
import tools
import tools.utils as utils
import os

DIRPATH = os.getcwd()
OPEN_POSE_PATH = DIRPATH[:-5]

class Demo(IO):

    def start(self,path_video):

        # openpose = '{}\\examples\\openpose\\openpose.bin'.format(self.arg.openpose)
        openpose = OPEN_POSE_PATH+'openpose\\bin\\OpenPoseDemo.exe'
        # video_name = path_video.split('\\')[len(path_video.split('\\'))-1]
        video_name = path_video.split('\\')[-1].split('.')[0]
        # print(video_name)
        # sys.exit()
        output_snippets_dir = '.\\data\\openpose_estimation\\snippets\\{}'.format(video_name)
        output_sequence_dir = '.\\data\\openpose_estimation\\data'
        output_sequence_path = '{}\\{}.json'.format(output_sequence_dir, video_name)
        output_result_dir = self.arg.output_dir
        output_result_path = '{}\\{}.mp4'.format(output_result_dir, video_name)
        label_name_path = '.\\resource\\kinetics_skeleton\\label_name.txt'
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
        print(command_line)
        # sys.exit()
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
        
        argClasses = output.sum(dim=3).sum(dim=2).sum(dim=1).argsort(dim=0)[-5:]
        vetor_saida = output.sum(dim=3).sum(dim=2).sum(dim=1).argsort(dim=0)[-5:]
        path_resultado = "features_ao_vivo\\"
        f3 = output.sum(dim=3).sum(dim=2).sum(dim=1)
        np.save(path_resultado+"\\"+video_name,f3.detach().cpu().numpy())
        # if not(os.path.exists(path_resultado+video_name+"\\")):
            # os.mkdir(path_resultado+video_name+"\\")
            # np.save(path_resultado+video_name+"\\"+"f3",f3.detach().cpu().numpy())

        print("argClasses")
        print(argClasses)
        # print(label_name[label])
        #'side kick'
        labels_violencia = ['drop kicking', 'high kick', 'punching person (boxing)', 'sword fighting', 'wrestling']
        top_5 = []
        # violencia_top5 = False
        violencia_top5 = "0"
        for t in argClasses[-5:]:
            top_5.append(label_name[t])
            print(label_name[t])
            if label_name[t] in labels_violencia:
                # violencia_top5 = True
                violencia_top5 = "1"
                break
        stgcn_label_predict = label_name[label]
        # print("top_5")
        # print(top_5)
        # print(violencia_top5=="1")
        
        DIR_LIMPAR = "data//openpose_estimation//"
        PASTAS_LIMPAR = ["data","snippets"]

        for pasta in PASTAS_LIMPAR:
            files = os.listdir(DIR_LIMPAR+pasta)
            files.sort()
            for file in files:
                full_path = DIR_LIMPAR+pasta+"/"+file
                if os.path.isfile(full_path):
                    os.remove(full_path)
                else:
                    shutil.rmtree(full_path)

        

        return (path_resultado+"\\"+video_name+".npy")

    @staticmethod
    def get_parser(add_help=False):

        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        parser.add_argument('--video',
            # default='.\\resource\\media\\skateboarding.mp4',
            default='.\\resource\\media\\kkkk.mp4',
            help='Path to video')
        parser.add_argument('--openpose',
            default='3dparty\\openpose\\build',
            help='Path to openpose')
        parser.add_argument('--output_dir',
            default='.\\data\\demo_result',
            help='Path to save results')
        parser.add_argument('--height',
            default=1080,
            type=int,
            help='Path to save results')
        parser.set_defaults(config=".\\config\\st_gcn\\kinetics-skeleton\\demo.yaml")
        parser.set_defaults(print_log=False)

        return parser
