import pdb

import torch
from torch.utils.data import Dataset
import glob
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    # x = normalize(np.cross(z, y_))  # (3)
    x = normalize(np.cross(y_, z))

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    # y = np.cross(x, z)  # (3)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


class MIVDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, frame_idx=0):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample
        self.frame_idx = frame_idx
        self.define_transforms()

        # self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        # self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        self.near_far = [0.0, 1.0]
        # self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.0], [1.75, 1.75, 1.0]])
        self.scene_bbox = torch.tensor([[-1.25, -1.25, -1.0], [1.25, 1.25, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        # get img name and poses
        print(f"Loading frame: {self.frame_idx:03d}")
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, f'rgbs/frame_{self.frame_idx:03d}/*')))
        # pdb.set_trace()
        with open("/work/Users/lisicheng/Dataset/INVR/Mirror/metadata.json", "r") as fp:
            metadata = json.load(fp)

        assert len(metadata["frames"]) == len(self.image_paths), \
                    'Mismatch between number of images and number of poses! Please check dataset!'


        # poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)

        # load full resolution image then resize
        poses = np.array([mat["transform_matrix"] for mat in metadata["frames"]]) # (N_images, 4, 4)
        self.near_fars = np.array(metadata["depth_range"])
        # self.near_fars = np.array([1.0, 10.0])

        ### old code
        # poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        # self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        # hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W = metadata["h"], metadata["w"]
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])

        if metadata["fl_x"] == metadata["fl_y"]:
            self.focal = metadata["fl_x"]
            self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]
        else:
            self.focal = [metadata["fl_x"], metadata["fl_y"]]
            print("Focal length is different between horizontal axis and vertical axis.")
            self.focal = [self.focal[0] * self.img_wh[0] / W, self.focal[1] * self.img_wh[1] / H]

        ### old code
        # H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        # self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        # self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        #     # # pose visualization
        # def draw_pose(ax, pose, axis_len=0.1):
        #     pose = np.array(pose)
        #     # draw position
        #     pos = pose[:3, -1].T
        #     ax.scatter(*pos, color="k")
        #
        #     # draw orientation
        #     x_axis = np.stack([pos, pos + axis_len * pose[0:3, 0]], 1)
        #     y_axis = np.stack([pos, pos + axis_len * pose[0:3, 1]], 1)
        #     z_axis = np.stack([pos, pos + axis_len * pose[0:3, 2]], 1)
        #
        #     ax.plot(*x_axis, color="r")
        #     ax.plot(*y_axis, color="g")
        #     ax.plot(*z_axis, color="b")
        # #
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.set_xlim(-3, 3)
        # ax.set_ylim(-3, 3)
        # ax.set_zlim(-3, 3)
        #
        # # 设置坐标轴标签
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        #
        # for pose in poses[:15]:
        #     draw_pose(ax, pose)
        # draw_pose(ax, np.eye(4), axis_len=1)
        #
        # # 显示图形
        # plt.show()

        # Step 2: correct poses
        poses = poses[:, 0:3] # (N_images, 3, 4)
        self.poses, self.pose_avg = center_poses(poses, np.eye(4))

        ### old code
        # # Original poses has rotation in form "down right back", change to "right up back"
        # # See https://github.com/bmild/nerf/issues/34
        # poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # # (N_images, 3, 4) exclude H, W, focal
        # self.poses, self.pose_avg = center_poses(poses, np.eye(4))

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor
        # pdb.set_trace()
        # self.poses[..., 3] -= 0.5

        # build rendering path
        N_views, N_rots = 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)
        # self.directions = get_ray_directions(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)

        # filter out test view
        self.test_view_id = metadata["test_view_id"]
        if self.split == 'traintest':
            img_list = list(set(np.arange(len(self.poses)))) # img_list: idx list
        elif self.split == 'train':
            img_list = list(set(np.arange(len(self.poses))) - set(self.test_view_id))
        elif self.split == 'test':
            img_list = list(set(self.test_view_id))

        print(f"{self.split} view id: {img_list}")
        ### old code
        # i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        # img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        for i in img_list:
            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])

            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)


            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # ax.set_xlim(-3, 3)
            # ax.set_ylim(-3, 3)
            # ax.set_zlim(-3, 3)
            #
            # # 设置坐标轴标签
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            #
            # ax.scatter(*rays_o[0], color="k")
            # ax.scatter(*(rays_o[0] + rays_d[0]), color="b")
            # ax.scatter(*(rays_o[-1] + rays_d[-1]), color="b")
            # ax.scatter(*(rays_o[-W] + rays_d[-W]), color="b")
            # ax.scatter(*(rays_o[W-1] + rays_d[W-1]), color="b")
            #
            # plt.show()
            #
            # pdb.set_trace()

            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]), h, w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]), h, w, 3)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample