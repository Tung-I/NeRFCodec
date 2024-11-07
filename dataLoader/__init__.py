from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .rtmv import RTMVDataset
from .miv import MIVDataset
from .lvc import LVCDataset
from .lvc_ff import LVCFFDataset



dataset_dict = {'blender': BlenderDataset,
                'llff':LLFFDataset,
                'tankstemple':TanksTempleDataset,
                'nsvf':NSVF,
                'own_data':YourOwnDataset,
                'rtmv': RTMVDataset,
                'miv': MIVDataset,
                'lvc': LVCDataset,
                'lvc_ff': LVCFFDataset}

from .scene_dataset import SceneDataset