import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F_vision
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
from ultralytics import FastSAM
import torch.nn.functional as F


def plot_img(ax, title: str, img: torch.Tensor):
    ax.imshow(img.cpu().numpy())
    ax.set_title(title)
    ax.axis("off")
def plot_flow(ax, title: str, flow: torch.Tensor): # flow is [2, H, W]
    flow_img_tensor = flow_to_image(flow).squeeze(0)
    plot_img(ax, title, flow_img_tensor.permute(1, 2, 0))

class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device

        self.reset_idx_f2k()

        # ====================================================================
        os.makedirs("debug_optical_flow", exist_ok=True)
        print("Loading RAFT Optical Flow Model...")
        
        # from opticalflow.sea_raft import SEA_RAFT as OPTICAL_FLOW_MODEL
        # self.flow_model = OPTICAL_FLOW_MODEL(device=self.device)

        weights = Raft_Large_Weights.DEFAULT
        self.raft_model = raft_large(weights=weights, progress=False).to(self.device)
        self.raft_model.eval()

        print("Loading FastSAM-S Model...")
        self.fastsam = FastSAM("FastSAM-s.pt")

        self.flow_offsets = [1, 4, 8, 12] 
        self.max_offset = max(self.flow_offsets)
        self.frame_history = [] # 0=curr frame, 1=last frame, etc
        # ====================================================================

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def compute_mask(self, frame: Frame):
        offset = 4
        if len(self.frame_history) <= offset:
            return

        [[H, W]] = frame.img_true_shape
        prev_frame = self.frame_history[offset]

        num_plots = 4
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        # transform points from prev frame to curr frame to get depth
        # T_proj = frame.T_WC.inv() * prev_frame.T_WC
        # T_proj = prev_frame.T_WC.inv() * frame.T_WC
        T_proj = frame.T_WC * prev_frame.T_WC.inv()
        print(f"trans prev->curr: {T_proj.translation()}")
        X_prev = prev_frame.T_WC.act(prev_frame.X_canon) # world -> prev frame coords
        # X_prev = X_prev * torch.tensor([1, -1, 1], device=self.device)
        X_proj = frame.T_WC.inv().act(X_prev) # prev frame -> curr frame
        # X_proj = T_proj.act(prev_frame.X_canon * torch.tensor([1, -1, 1], device=self.device)) # prev frame -> curr frame
        # X_proj = X_proj * torch.tensor([1, -1, 1], device=self.device)
        depth = X_proj[:, 2]
        plot_img(axes[0], f"Depth {frame.frame_id}", depth.view(H, W))

        u_proj = (X_proj[:, 0] / depth.clamp(min=0.0001))
        v_proj = (X_proj[:, 1] / depth.clamp(min=0.0001))
        u_proj = (u_proj - u_proj.min()) * W / (u_proj.max()-u_proj.min())
        v_proj = (v_proj - v_proj.min()) * H / (v_proj.max()-v_proj.min())

        v_orig, u_orig = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )

        flow_exp = torch.stack([
            u_orig - u_proj.view(H, W),
            v_orig - v_proj.view(H, W),
        ], dim=0)  # (2, H, W)
        plot_flow(axes[1], f"Expected Flow {frame.frame_id}", flow_exp)

        flow = self.raft_model(prev_frame.img, frame.img)[-1][0] # [2, H, W]
        plot_flow(axes[2], f"Computed Flow {frame.frame_id}", flow)

        residual = (flow - flow_exp).norm(dim=0)
        residual = (residual - residual.median()).abs()
        print("residual range:", residual.min(), residual.max())
        plot_img(axes[3], f"Residual {frame.frame_id}", residual)
        
        if not os.path.exists("debug_optical_flow_depth"):
            os.makedirs("debug_optical_flow_depth")
        plt.tight_layout()
        plt.savefig(f"debug_optical_flow_depth/{frame.frame_id:04d}.png", bbox_inches='tight')
        plt.close(fig)

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        self.frame_history.insert(0, frame)
        self.frame_history = self.frame_history[:self.max_offset+1]

        # ====================================================================
        with torch.no_grad():
            if len(self.frame_history) > self.max_offset:
                # 1. RUN FASTSAM ONCE FOR THE CURRENT FRAME
                img_np_fastsam = (frame.uimg.cpu().numpy() * 255).astype("uint8")
                
                sam_results = self.fastsam(
                    img_np_fastsam, 
                    device=self.device, 
                    retina_masks=True, 
                    conf=0.4, 
                    iou=0.9, 
                    verbose=False
                )
                
                # Ultralytics has a built-in .plot() function that returns the overlay image in BGR
                fastsam_viz_bgr = sam_results[0].plot(
                    line_width=None,
                    boxes=False,
                    probs=False,
                    labels=False,
                    masks=True,
                    color_mode="instance"
                )
                fastsam_viz_rgb = fastsam_viz_bgr[..., ::-1] # Convert BGR to RGB for matplotlib
                
                # Safely extract masks for the averaging logic later
                masks = sam_results[0].masks.data if sam_results[0].masks is not None else None
                
                # Prepare the current frame (img1) for RAFT
                img1 = frame.uimg.permute(2, 0, 1).unsqueeze(0)
                img1 = (img1 * 2 - 1).to(self.device)
                
                # 2. SETUP PLOTTING GRID (3 Rows. At least 2 columns to fit Row 1)
                num_cols = max(2, len(self.flow_offsets))
                fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 15), squeeze=False)
                
                # Plot Current Frame (Row 0, Col 0)
                orig_img_np = (frame.uimg.cpu().numpy() * 255).astype("uint8")
                axes[0, 0].imshow(orig_img_np)
                axes[0, 0].set_title(f"Current Frame {frame.frame_id}")
                axes[0, 0].axis("off")
                
                # Plot FastSAM Segmentations (Row 0, Col 1)
                axes[0, 1].imshow(fastsam_viz_rgb)
                axes[0, 1].set_title("FastSAM Segmentations")
                axes[0, 1].axis("off")
                
                # Turn off unused empty axes in the top row
                for j in range(2, num_cols):
                    axes[0, j].axis("off")
                
                # 3. LOOP THROUGH TEMPORAL OFFSETS
                for i, offset in enumerate(self.flow_offsets):
                    past_frame = self.frame_history[offset]
                    
                    img2 = past_frame.uimg.permute(2, 0, 1).unsqueeze(0)
                    img2 = (img2 * 2 - 1).to(self.device)
                    
                    # Predict Flow (Current Frame -> Past Frame)
                    list_of_flows = self.raft_model(img1, img2)
                    predicted_flow = list_of_flows[-1][0] # Shape: (2, H, W)
                    
                    # --- PLOT: RAW FLOW (Row 1) ---
                    raw_flow_tensor = flow_to_image(predicted_flow)
                    raw_flow_np = raw_flow_tensor.permute(1, 2, 0).cpu().numpy()
                    
                    axes[1, i].imshow(raw_flow_np)
                    axes[1, i].set_title(f"Raw Flow (t - {offset})")
                    axes[1, i].axis("off")
                    
                    # --- PLOT: AVERAGED FLOW (Row 2) ---
                    avg_object_flow = predicted_flow.clone()
                    
                    if masks is not None:
                        h, w = avg_object_flow.shape[1], avg_object_flow.shape[2]
                        # Interpolate masks to match flow tensor dimensions
                        masks_resized = F.interpolate(
                            masks.unsqueeze(1), size=(h, w), mode='nearest'
                        ).squeeze(1).bool()
                        
                        for mask in masks_resized:
                            if mask.sum() > 0: 
                                mean_u = avg_object_flow[0][mask].mean()
                                mean_v = avg_object_flow[1][mask].mean()
                                avg_object_flow[0][mask] = mean_u
                                avg_object_flow[1][mask] = mean_v
                                
                    avg_flow_tensor = flow_to_image(avg_object_flow)
                    avg_flow_np = avg_flow_tensor.permute(1, 2, 0).cpu().numpy()
                    
                    axes[2, i].imshow(avg_flow_np)
                    axes[2, i].set_title(f"Avg Flow (t - {offset})")
                    axes[2, i].axis("off")
                
                # Turn off any remaining unused axes if there are fewer offsets than 2
                for row in [1, 2]:
                    for j in range(len(self.flow_offsets), num_cols):
                        axes[row, j].axis("off")
                
                plt.tight_layout()
                plt.savefig(f"debug_optical_flow/flow_{frame.frame_id:04d}.png", bbox_inches='tight')
                plt.close(fig)
        # ====================================================================

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # Track
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        frame.T_WC = T_WCf
        print(f"trans kf->curr: {T_CkCf.translation()}")

        self.compute_mask(frame)

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
