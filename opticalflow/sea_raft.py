import torch
import sys
sys.path.append('thirdparty/SEA_RAFT')
sys.path.append('thirdparty/SEA_RAFT/core')
from core.raft import RAFT
from utils.utils import load_ckpt
from config.parser import json_to_args

def convert_img(img: torch.Tensor):
    '''convert to neuflow img (RGB 1xCxHxW [0, 255] float)'''
    return ((img + 1) * 255 / 2)

class SEA_RAFT:
    def __init__(self, 
                 huggingface_path: str = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M", 
                 local_path: str = "Tartan-C-T-TSKH-spring540x960-M", 
                 cfg = "spring-M",
                 device = torch.device('cuda')) -> None:
        self.name = 'SEA-RAFT'

        args = json_to_args(f'thirdparty/SEA_RAFT/config/eval/{cfg}.json')
        
        if huggingface_path:
            model = RAFT.from_pretrained(huggingface_path, args=args)
        elif local_path:
            model = RAFT(args)
            load_ckpt(model, local_path)
        else:
            raise Exception("One of huggingface_path or local_path must be set")

        model = model.to(device)
        model.eval()

        self.model = model

    def __call__(self, end_img: torch.Tensor, start_img: torch.Tensor, iters: int = None):
        flow = self.model(convert_img(end_img), convert_img(start_img), iters=iters, test_mode=True)['flow'][-1][0]
        return flow
