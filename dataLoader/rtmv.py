import pdb

import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
import glob
from torchvision import transforms as T

from .ray_utils import *

class RTMVDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.img_wh = (int(1600 / downsample), int(1600 / downsample))
        self.define_transforms()

        # self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.scene_bbox = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_intrinsics()
        self.read_meta()
        # self.define_proj_mat()

        self.white_bg = True
        # self.near_far = [2.0, 6.0] # TODO?
        self.near_far = [0. , 1.0]

        # self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        # self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, '00000.json'), 'r') as f:
            meta = json.load(f)['camera_data']

        self.shift = np.array(meta['scene_center_3d_box'])
        self.scale = (np.array(meta['scene_max_3d_box']) -
                      np.array(meta['scene_min_3d_box'])).max() / 2 * 1.05  # enlarge a little



        fx = meta['intrinsics']['fx'] * self.downsample
        fy = meta['intrinsics']['fy'] * self.downsample
        cx = meta['intrinsics']['cx'] * self.downsample
        cy = meta['intrinsics']['cy'] * self.downsample
        w = int(meta['width']*self.downsample)
        h = int(meta['height']*self.downsample)
        K = np.float32([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])
        self.intrinsics = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, [fx, fy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.img_wh = (w, h)

    def read_meta(self):

        if self.split == 'train': start_idx, end_idx = 0, 100
        elif self.split == 'trainval': start_idx, end_idx = 0, 105
        elif self.split == 'test': start_idx, end_idx = 105, 150
        else: start_idx, end_idx = 0, 150

        img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))[start_idx:end_idx]
        poses = sorted(glob.glob(os.path.join(self.root_dir, '*.json')))[start_idx:end_idx]

        assert(len(img_paths)==len(poses))


        # self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(img_paths), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#
            pose = poses[i]
            with open(pose, 'r') as f:
                p = json.load(f)['camera_data']
            c2w = np.array(p['cam2world']).T[:3]
            c2w = torch.FloatTensor(c2w)
            c2w[:, 1:3] *= -1
            if 'bricks' in self.root_dir:
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # bound in [-0.5, 0.5]
            self.poses += [c2w]
            del p


            # frame = self.meta['frames'][i]
            # pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            # c2w = torch.FloatTensor(pose)
            # self.poses += [c2w]

            image_path = img_paths[i]
            img = Image.open(image_path)
            del image_path

            # image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            # self.image_paths += [image_path]
            # img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]
            del img

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            del rays_o
            del rays_d

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

        #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1],
                                                                  3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

        del img_paths
        del poses
        del self.directions
        del c2w
        pdb.set_trace()

    def define_transforms(self):
        self.transform = T.ToTensor()