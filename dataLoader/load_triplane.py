import pdb

import torch
from torch.utils.data import Dataset


class TriplaneDataset(Dataset):
    def __init__(self):
        path = f'/work/Users/lisicheng/Code/TensoRF/log/tensorf_chair_VM_codec/tensorf_chair_VM_codec.th'
        state_dict = torch.load(path)['state_dict'] # special in TensoRF
        # pdb.set_trace()
        self.density_plane = [v for k,v in state_dict.items() if 'density_plane' in k]
        self.app_plane = [v for k,v in state_dict.items() if 'app_plane' in k]


    def __len__(self):
        assert len(self.density_plane) == len(self.app_plane)
        return len(self.density_plane)

    def __getitem__(self, idx):
        return self.density_plane[idx], self.app_plane[idx]

if __name__ == "__main__":
    triplane_dataset = TriplaneDataset()
    pdb.set_trace()
    # for i in range(len(triplane_dataset)):
    #     density_plane, app_plane = triplane_dataset[i]
    #     print(density_plane)
    #     print(app_plane)