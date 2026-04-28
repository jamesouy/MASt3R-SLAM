import sys
sys.path.append('thirdparty/NeuFlow_v2')
from NeuFlow import neuflow
from NeuFlow.backbone_v7 import ConvBlock

import torch

def convert_img(img: torch.Tensor):
    '''convert to neuflow img (BGR 1xCxHxW [0, 255] half)'''
    return ((img + 1) * 255 / 2)[:, [2, 1, 0], :, :].half()

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class NeuFlow:
    def __init__(self, image_height, image_width, huggingface_path: str = "Study-is-happy/neuflow-v2", local_path: str = None, device = torch.device('cuda')) -> None:
        self.name = 'NeuFlow'

        if huggingface_path:
            model = neuflow.NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(device)
        elif local_path:
            model = neuflow.NeuFlow().to(device)
            checkpoint = torch.load('thirdparty/NeuFlow_v2/neuflow_mixed.pth', map_location='cuda')
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            raise Exception("One of huggingface_path or local_path must be set")

        for m in model.modules():
            if type(m) is ConvBlock:
                m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
                m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
                delattr(m, "norm1")  # remove batchnorm
                delattr(m, "norm2")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward

        model.eval()
        model.half()

        model.init_bhwd(1, image_height, image_width, device)

        self.model = model

    def __call__(self, end_img: torch.Tensor, start_img: torch.Tensor):
        flow = self.model(convert_img(end_img), convert_img(start_img))[-1][0].float()
        return flow
