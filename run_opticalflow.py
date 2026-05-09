import argparse
import os
import time
import torch
from torchvision.utils import save_image, flow_to_image
from mast3r_slam.dataloader import MP4Dataset
from mast3r_slam.config import config, load_config

import matplotlib.pyplot as plt

device = torch.device('cuda')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, required=True)
    parser.add_argument("-o", "--out_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    load_config("config/base.yaml")
    dataset = MP4Dataset(args.video_path)
    # h, w = dataset.get_img_shape()[0]
    print(dataset.get_image(0).shape)
    h, w = dataset.get_image(0).shape[:2]
    print(h, w)

    models = []

    from opticalflow.raft import RAFT
    models.append(RAFT())

    from opticalflow.neuflow import NeuFlow
    models.append(NeuFlow(h, w))

    from opticalflow.flowformer import FlowFormer
    models.append(FlowFormer())

    # from opticalflow.flowformerpp import FlowFormerPlusPlus
    # models.append(FlowFormerPlusPlus())

    # from opticalflow.sea_raft import SEA_RAFT
    # models.append(SEA_RAFT())

    def load_img(index):
        '''loads image in RGB 1xCxHxW [-1, 1] float (RAFT/MASt3R) format'''
        img = torch.tensor(dataset.get_image(index)).to(device)
        img = img.permute(2, 0, 1).unsqueeze(0) # Convert to [1, C, H, W]
        img = img * 2 - 1 # Normalize to [-1, 1]
        return img

    def plot_img(ax, title: str, img: torch.Tensor):
        ax.imshow(img.cpu().numpy())
        ax.set_title(title)
        ax.axis("off")

    offset = 4
    img_history = []
    for i in range(0, len(dataset)):
        img = load_img(i)
        img_history.insert(0, img)
        img_history = img_history[:offset+1]
        if len(img_history) <= offset:
            continue
        last_img = img_history[offset]

        num_plots = len(models) + 1
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        plot_img(axes[0], f"Image {i}", ((img + 1) / 2)[0].permute(1, 2, 0))

        for j, model in enumerate(models):
            start_time = time.perf_counter()
            flow = model(img, last_img)
            end_time = time.perf_counter()
            print(f"{model.name} prediction time: {end_time - start_time:.4f} seconds")
            plot_img(axes[j+1], model.name, flow_to_image(flow).permute(1, 2, 0))
            

        # save_image(flow_img, f"{args.out_path}/flow_{num_flows:03}_{i-offset}_{i}.png")

        plt.tight_layout()
        plt.savefig(f"{args.out_path}/{i-offset:03d}_{i:03d}.png", bbox_inches='tight')
        plt.close(fig)