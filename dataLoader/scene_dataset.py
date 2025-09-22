import pdb

import sys
import numpy as np
import torch,cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from tqdm import tqdm
import os
from PIL import Image
import glob
from torchvision import transforms as T




# pdb.set_trace()
sys.path.append('/work/Users/lisicheng/Code/TensoRF/')
from dataLoader.ray_utils import *
# from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from models.tensoRF import TensorVMSplit
from compressai.ops import compute_padding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SceneDataset(Dataset):
    def __init__(self):
        self.model_basedir = "./log/falling_google/"
        self.data_basedir = "../../Dataset/RTMV_all/falling_google_scenes/"

        self.train_scene_list = [f"{idx:05d}" for idx in range(40)]
        # debug
        # self.train_scene_list = [f"{idx:05d}" for idx in range(1)]
        self.scene_num = len(self.train_scene_list)

        self.model_list = []

        for item in self.train_scene_list:
            self.model_list.append(self.load_scene_representation(self.model_basedir+f"{item}/{item}.th"))

        self.optimizer_list = []
        for model in self.model_list:
            grad_vars = model.get_optparam_groups()
            self.optimizer_list.append(torch.optim.Adam(grad_vars, betas=(0.9, 0.99)))


        ### get ray directions -> dataset dependent
        self.downsample = 1
        self.define_transforms()
        self.read_intrinsics()

    def read_intrinsics(self):
        with open(os.path.join(self.data_basedir, '00000', '00000.json'), 'r') as f:
            meta = json.load(f)['camera_data']

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

    def load_scene_representation(self, ckpt_path):
        # ckpt_path = f"./log/falling_google/{name}/{name}.th"

        ckpt = torch.load(ckpt_path, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})

        tensorf = eval("TensorVMSplit")(**kwargs)
        tensorf.load(ckpt)  # load pretrained model

        return tensorf

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # default
        # scene_idx = np.random.randint(0, self.scene_num)
        img_idx = np.random.randint(0, 100) # 100 training views
        # debug
        scene_idx = 0
        # img_idx = 0

        # get rgb
        img_path = os.path.join(self.data_basedir, f"{scene_idx:05d}", "images", f"{img_idx:05d}.png")
        img = Image.open(img_path)
        img = self.transform(img)  # (4, h, w)
        img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
        alpha_img = img[:, -1:] > 0.5
        # print(alpha_img)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        # get valid pix idx
        condition = alpha_img.cpu().numpy() > 0
        filtered_indices = np.where(condition)[0]
        random_indices = np.random.choice(filtered_indices, size=4096, replace=False)
        assert alpha_img[random_indices].sum() == 4096

        # get c2w
        cam_path = os.path.join(self.data_basedir, f"{scene_idx:05d}", f"{img_idx:05d}.json")
        with open(cam_path, 'r') as f:
            p = json.load(f)['camera_data']
        c2w = np.array(p['cam2world']).T[:3]
        c2w = torch.FloatTensor(c2w)
        c2w[:, 1:3] *= -1
        if 'bricks' in self.data_basedir:
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2 * self.scale  # bound in [-0.5, 0.5]

        # get ray_o,ray_d
        rays_o, rays_d = get_rays(self.directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)
        # pdb.set_trace()


        return {
            "model": self.model_list[scene_idx],
            "optimizer": self.optimizer_list[scene_idx],
            # "rays": rays,
            # "rgbs": img,
            "rays": rays[random_indices],
            "rgbs": img[random_indices],
            "scene_idx": scene_idx,
            "img_idx": img_idx,
            "alpha": alpha_img
        }
    # TODO： add optim


def merge_feature(batch_data, feat_type="density"):
    B = len(batch_data)
    meta_data = []
    concat_triplane_list = []
    for data in batch_data: # batch loop
        triplane_info = []
        padded_plane_list = []
        for i in range(3): # triplane loop
            plane_info = {}
            # get size # (1, C, H, W)
            if feat_type == "density":
                plane = data["model"].density_plane[i]
            elif feat_type == "app":
                plane = data["model"].app_plane[i]
            else:
                NotImplementedError()
            _, C, H, W = plane.size()
            # if H> 320 or W > 320:
            #     pdb.set_trace()
            pad, unpad = compute_padding(H, W, out_h=384, out_w=384, min_div=2 ** 6 )

            # spatial padding
            padded_plane = F.pad(plane, pad, mode="constant", value=0)

            # add to list
            padded_plane_list.append(padded_plane)

            # record plane info
            plane_info.update({
                "CHW": [C, H, W],
                "unpad": unpad
            })
            triplane_info.append(plane_info)

        meta_data.append(triplane_info)

        # concat triplane in single scene
        concat_triplane = torch.concat(padded_plane_list, dim=1) # (1, 3*C, H, W)
        concat_triplane_list.append(concat_triplane)

    # stack concated triplane from multiple scenes
    batch_triplane = torch.concat(concat_triplane_list, dim=0) # (B, 3*C, H, W)
    # pdb.set_trace()

    return batch_triplane, meta_data

def split_feature(batch_triplane, meta_data):
    B, C, H, W = batch_triplane.size()
    c = C // 3
    batch_list = []
    for b_idx in range(B):
        concat_triplane = batch_triplane[b_idx:b_idx+1]
        planes_list = []
        for p_idx in range(3):
            pad_plane = concat_triplane[:, p_idx*c:(p_idx+1)*c]

            unpad = meta_data[b_idx][p_idx]["unpad"]
            plane = F.pad(pad_plane, unpad)

            planes_list.append(plane)
        batch_list.append(planes_list)

    return batch_list # [[] x 3] x bs




if __name__ == "__main__":
    scene_dataset = SceneDataset()
    batch_data = [scene_dataset[i] for i in range(4)]
    pdb.set_trace()
    # test merge feature
    den_merge_planes, den_meta_data = merge_feature(batch_data, feat_type="density")
    app_merge_planes, app_meta_data = merge_feature(batch_data, feat_type="app")

    # test split feature
    den_split_planes = split_feature(den_merge_planes, den_meta_data)
    app_split_planes = split_feature(app_merge_planes, app_meta_data)

    # # 定义筛选条件，例如要求大于5的数
    # condition = scene_dataset[0]["alpha"].cpu().numpy() > 0
    #
    # # 使用条件来过滤满足条件的索引
    # filtered_indices = np.where(condition)[0]
    #
    # # 随机选择5个索引
    # random_indices = np.random.choice(filtered_indices, size=5, replace=False)

    pdb.set_trace()