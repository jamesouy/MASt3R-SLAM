import argparse
import os
import time
import torch
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.utils import save_image, flow_to_image
from mast3r_slam.dataloader import MP4Dataset
from mast3r_slam.config import config, load_config

device = torch.device('mps')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, required=True)
    parser.add_argument("-o", "--out_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    load_config("config/base.yaml")
    dataset = MP4Dataset(args.video_path)
    h, w = dataset.get_img_shape()[0]
    print(h, w)

    model = raft_large(pretrained=True, progress=False).to(device)
    # model = raft_small(pretrained=True, progress=False).to(device)
    model = model.eval()

    def load_img(index):
        img = torch.tensor(dataset.get_image(index)).to(device)
        img = img.permute(2, 0, 1).unsqueeze(0) # Convert to [1, C, H, W]
        img = img * 2 - 1 # Normalize to [-1, 1]
        return img

    last_image_index = 0
    last_img = load_img(0)
    num_flows = 0
    for i in range(1, len(dataset)):
        if i % 5 != 0: continue
        img = load_img(i)

        start_time = time.perf_counter()
        list_of_flows = model(last_img, img)
        end_time = time.perf_counter()
        print(f"Optical flow prediction time: {end_time - start_time:.4f} seconds")

        flow_img = flow_to_image(list_of_flows[-1]).to(torch.float) / 255.0
        save_image(flow_img[0], f"{args.out_path}/flow_{num_flows:03}_{last_image_index}_{i}.png")

        last_image_index = i
        last_img = img
        num_flows += 1
