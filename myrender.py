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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, depth_min, depth_max, mode, skip_groundtruth):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    sum_alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "sum_alpha")
    cnt_gs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "cnt_gs")
    myeval_path = os.path.join(model_path, name, "ours_{}".format(iteration), "myeval")

    path1 = os.path.join(model_path, name, "ours_{}".format(iteration), "path1")    # mean of alpha
    path2 = os.path.join(model_path, name, "ours_{}".format(iteration), "path2")    # var of alpha


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    #makedirs(sum_alpha_path, exist_ok=True)
    #makedirs(cnt_gs_path, exist_ok=True)
    #makedirs(myeval_path, exist_ok=True)

    makedirs(path1, exist_ok=True)
    makedirs(path2, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        result = render(view, gaussians, pipeline, background)
        rendering = result["render"]

        tmpinfo = result["tmpinfo"]
        depth = tmpinfo[0, :, :].unsqueeze(0).repeat(3, 1, 1)
        #sum_alpha = tmpinfo[1, :, :].unsqueeze(0).repeat(3, 1, 1)
        #cnt_gs = tmpinfo[2, :, :].unsqueeze(0).repeat(3, 1, 1)
        #myeval = tmpinfo[1, :, :].unsqueeze(0).repeat(3, 1, 1)

        v1 = tmpinfo[1, :, :].unsqueeze(0).repeat(3, 1, 1)
        v2 = tmpinfo[2, :, :].unsqueeze(0).repeat(3, 1, 1)
        
        
        if mode == "cut":
            depth = (depth - depth_min) / (depth_max - depth_min)
        elif mode == "rec":
            depth = 1.0/(depth+1)
        elif mode == "exp":
            depth = torch.exp(-depth)

        #print("?????????????????????", torch.max(v1))
        #print(torch.max(v2))

        v1 = torch.div(v1, torch.max(v1))
        #print(torch.max(v1), "================")
        #exit()
        #v1 = torch.div(v1, 3)
        #v2 = torch.div(v1, torch.max(v2))
        #v2 = torch.div(v2, 0.3)
        #exit()

        #print(sum_alpha)
        #print(myeval)
        #print(torch.max(myeval))
        #print(cnt_gs)
        #print(myeval / cnt_gs)
        #sum_alpha = torch.div(sum_alpha, torch.max(sum_alpha))
        #cnt_gs = torch.div(cnt_gs, torch.max(cnt_gs))
        #myeval = torch.div(myeval, torch.max(myeval))
        #myeval = torch.div(myeval, 0.5)
        #myeval = 0.5/(myeval + 1)
        #myeval = torch.exp(-myeval)
        #print(sum_alpha)
        #exit()

        if not skip_groundtruth :
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(sum_alpha, os.path.join(sum_alpha_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(cnt_gs, os.path.join(cnt_gs_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(myeval, os.path.join(myeval_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(v1, os.path.join(path1, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(v2, os.path.join(path2, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, depth_min : float, depth_max : float, mode : str, skip_groundtruth : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, depth_min, depth_max, mode, skip_groundtruth)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, depth_min, depth_max, mode, skip_groundtruth)

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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.depth_min, args.depth_max, args.mode, args.skip_groundtruth)