#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2
import keyboard
import copy
import time

def move_camera(camera, delta_R, delta_T):
    """
    Move the camera by updating its rotation matrix R and translation vector T.

    Args:
    - camera (Camera): The camera object to be moved.
    - delta_R : The additional rotation matrix.
    - delta_T : The additional translation vector.
    """

    new_R = np.dot(camera.R, delta_R)
    #new_T = camera.T + delta_T
    #new_T = delta_T + np.dot(delta_R, camera.T)

    camera.R = new_R
    #camera.T = new_T
    camera.trans += np.dot(camera.R, delta_T)


    camera.world_view_transform = torch.tensor(getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx, fovY=camera.FoVy).transpose(0,1).cuda()
    camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]



def myrender(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    cv2.namedWindow("Image")
    print('g: save')
    print('1: switch filter')
    print('+: threshold up')
    print('-: threshold down')
    with torch.no_grad():
        idx = 0
        model_path = dataset.model_path
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        view = copy.deepcopy(scene.getTrainCameras()[0])
        view.trans = np.dot(view.T, view.R)
        view.T = np.array([0.0, 0.0, 0.0])
        
        use_filter = False
        threshold = 1.0


        while True:
            time.sleep(1.0/60)
            delta_T = np.array([0.0, 0.0, 0.0])
            delta_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

            dis = 0.5
            tx = np.array([dis, 0.0, 0.0])
            ty = np.array([0.0, dis, 0.0])
            tz = np.array([0.0, 0.0, dis])

            angle = np.radians(5)
            rx = np.array([[1, 0, 0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
            ry = np.array([[np.cos(angle), 0, np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]])
            rz = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])

            """
            op = input()
            if op == 'a':
                delta_T += tx
            elif op == 'd':
                delta_T -= tx
            elif op == 'w':
                delta_T -= tz
            elif op == 's':
                delta_T += tz
            elif op == 'q':
                delta_T -= ty
            elif op == 'e':
                delta_T += ty
            if op == 'j':
                delta_R = np.dot(ry.T, delta_R)
            elif op == 'l':
                delta_R = np.dot(ry, delta_R)
            elif op == 'i':
                delta_R = np.dot(rx, delta_R)
            elif op == 'k':
                delta_R = np.dot(rx.T, delta_R)
            """
            if keyboard.is_pressed(' '):
                view = copy.deepcopy(scene.getTrainCameras()[0])

            if keyboard.is_pressed('1'):
                use_filter = not use_filter
                print('use_filter : ', use_filter)
            if keyboard.is_pressed('+'):
                threshold += 0.01
                print('threshold  : {:.2f}'.format(threshold))
            if keyboard.is_pressed('-'):
                threshold -= 0.01
                print('threshold  : {:.2f}'.format(threshold))

            if keyboard.is_pressed('a'):
                delta_T -= tx
            if keyboard.is_pressed('d'):
                delta_T += tx
            if keyboard.is_pressed('w'):
                delta_T += tz
            if keyboard.is_pressed('s'):
                delta_T -= tz
            if keyboard.is_pressed('q'):
                delta_T += ty
            if keyboard.is_pressed('e'):
                delta_T -= ty

            if keyboard.is_pressed('j'):
                delta_R = np.dot(ry.T, delta_R)
            if keyboard.is_pressed('l'):
                delta_R = np.dot(ry, delta_R)
            if keyboard.is_pressed('i'):
                delta_R = np.dot(rx, delta_R)
            if keyboard.is_pressed('k'):
                delta_R = np.dot(rx.T, delta_R)
            if keyboard.is_pressed('o'):
                delta_R = np.dot(rz, delta_R)
            if keyboard.is_pressed('u'):
                delta_R = np.dot(rz.T, delta_R)

            if keyboard.is_pressed('g'):
                torchvision.utils.save_image(rendering, os.path.join(model_path, 'save_' + '{0:05d}'.format(idx) + '.png'))

            if cv2.waitKey(1) == 27:
                break

            move_camera(view, delta_R, delta_T)
            idx += 1
            result = render(view, gaussians, pipeline, background)

            rendering = result["render"]

            if use_filter:
                thickness = result["tmpinfo"][1, :, :].unsqueeze(0).repeat(3, 1, 1)
                rendering[thickness > threshold] = 0


            image = rendering.cpu().permute(1, 2, 0).numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image", image)
            #torchvision.utils.save_image(rendering, os.path.join(model_path, "renderd.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--depth_min", default=0, type=float)
    parser.add_argument("--depth_max", default=20, type=float)
    parser.add_argument("--mode", default="cut", type=str)

    parser.add_argument("--skip_groundtruth", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    #render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.depth_min, args.depth_max, args.mode, args.skip_groundtruth)
    myrender(model.extract(args), args.iteration, pipeline.extract(args))