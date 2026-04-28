import torch
import sys
sys.path.append('thirdparty/FlowFormerPlusPlus')
from thirdparty.FlowFormerPlusPlus.core.FlowFormer import build_flowformer
# from core.FlowFormer.PerCostFormer3.transformer import FlowFormer
from configs.submissions import get_cfg as get_flowformer_cfg

def convert_img(img: torch.Tensor):
    '''convert to neuflow img (RGB 1xCxHxW [0, 255] float)'''
    return ((img + 1) * 255 / 2)

class FlowFormerPlusPlus:
    def __init__(self, checkpoint: str = 'things_288960', device = torch.device('cuda')) -> None:
        self.name = 'FlowFormer++'

        cfg = get_flowformer_cfg()
        cfg.update({'model': f'thirdparty/FlowFormerPlusPlus/checkpoints/{checkpoint}.pth'})

        model = torch.nn.DataParallel(build_flowformer(cfg))
        model.load_state_dict(torch.load(cfg.model))

        model = model.to(device)
        model.eval()
        model.module.eval()

        self.model = model.module

    def __call__(self, end_img: torch.Tensor, start_img: torch.Tensor):
        flow = self.model(convert_img(end_img), convert_img(start_img))[0][0]
        return flow
