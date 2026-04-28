import torch
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights

class RAFT:
    def __init__(self, raft_model: str = 'large', device = torch.device('cuda')) -> None:
        self.name = 'RAFT'
        
        match raft_model:
            case 'large':
                self.model = raft_large(weights=Raft_Large_Weights)
            case 'small':
                self.model = raft_small(weights=Raft_Small_Weights)
            case _:
                raise Exception("Unhandled raft_model:", raft_model)

        self.model = self.model.to(device)
        self.model.eval()

    def __call__(self, end_img: torch.Tensor, start_img: torch.Tensor):
        flow = self.model(end_img, start_img)[-1][0]
        return flow
