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


DISABLE_PLOTS = True
def start_plots(num_plots):
    if DISABLE_PLOTS: return None, [None]*num_plots
    return plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
def plot_img(ax, title: str, img: torch.Tensor):
    if DISABLE_PLOTS: return
    ax.imshow(img.cpu().numpy())
    ax.set_title(title)
    ax.axis("off")
def plot_flow(ax, title: str, flow: torch.Tensor): # flow is [2, H, W]
    if DISABLE_PLOTS: return
    flow_img_tensor = flow_to_image(flow).squeeze(0)
    plot_img(ax, title, flow_img_tensor.permute(1, 2, 0))
def save_plots(fig, path):
    if DISABLE_PLOTS: return
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

def estimate_K(X: torch.Tensor, H, W):
    '''X: [H, W, 3]'''
    v, u = torch.meshgrid(
        torch.arange(H, device=X.device, dtype=torch.float32),
        torch.arange(W, device=X.device, dtype=torch.float32),
        indexing="ij",
    )
    u = u.reshape(-1)
    v = v.reshape(-1)
    x = X[:, 0] / X[:, 2].clamp(min=1e-5)
    y = X[:, 1] / X[:, 2].clamp(min=1e-5)
    fx = torch.cov(torch.stack([u, x]))[0, 1] / torch.var(x)
    cx = torch.mean(u) - fx * torch.mean(x)

    fy = torch.cov(torch.stack([v, y]))[0,1] / torch.var(y)
    cy = torch.mean(v) - fy * torch.mean(y)
    
    return torch.tensor([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0,  1]], device=X.device)

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
        from opticalflow.raft import RAFT as OPTICAL_FLOW_MODEL
        # from opticalflow.sea_raft import SEA_RAFT as OPTICAL_FLOW_MODEL
        self.flow_model = OPTICAL_FLOW_MODEL(device=self.device)

        print("Loading FastSAM Model...")
        # self.fastsam = FastSAM("FastSAM-s.pt")
        self.fastsam = FastSAM("FastSAM-x.pt")

        self.flow_offsets = [1, 4, 8, 12] 
        self.max_offset = max(self.flow_offsets)
        self.frame_history = [] # 0=curr frame, 1=last frame, etc
        self.prev_mask = None
        # ====================================================================

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self,):
        self.idx_f2k = None
    
    def segmentation(self, frame: Frame):
        img_np_fastsam = (frame.uimg.cpu().numpy() * 255).astype("uint8")
        sam_result = self.fastsam(
            img_np_fastsam, 
            device=self.device, 
            retina_masks=True, 
            conf=0.4, 
            iou=0.9, 
            verbose=False
        )[0]
        
        # Safely extract masks for the averaging logic later
        # masks = sam_result.masks.data if sam_result.masks is not None else None
        assert sam_result.masks is not None
        masks = sam_result.masks.data.to(self.device).bool()
        [[H, W]] = frame.img_true_shape
        assert masks.shape[1] == H and masks.shape[2] == W, f"masks shape: {masks.shape}"

        # Sort area and filter to only area > 0
        areas = masks.sum(dim=(1,2))
        areas, sort_idx = torch.sort(areas, descending=True)
        masks = masks[sort_idx][areas > 0]

        return masks, sam_result # sam_result can be used for visualization
    
    def mask_prev_frame(self, frame: Frame, prev_frame: Frame):
        '''places a mask on the previous frame instead of current frame'''
        [[H, W]] = frame.img_true_shape

        fig, axes = start_plots(6)
        
        # Transform from prev frame to curr frame
        T = frame.T_WC.inv() * prev_frame.T_WC
        X_trans = T.act(prev_frame.X_canon) # prev frame coords -> curr frame coords
        depth = X_trans[:, 2]
        plot_img(axes[0], f"Depth {frame.frame_id}", depth.view(H, W))

        # Project to prev frame image plane
        K = estimate_K(frame.X_canon, H, W)
        X_proj = K @ X_trans.T
        u_proj = (X_proj[0] / X_proj[2].clamp(min=1e-5))
        v_proj = (X_proj[1] / X_proj[2].clamp(min=1e-5))

        v_orig, u_orig = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )

        # Compute flow from projected points
        flow_exp = torch.stack([
            u_proj.view(H, W) - u_orig,
            v_proj.view(H, W) - v_orig,
        ], dim=0)  # (2, H, W)
        plot_flow(axes[1], f"Expected Flow {frame.frame_id}", flow_exp)

        # print(prev_frame.frame_id, prev_frame.img.shape, prev_frame.img.min(), prev_frame.img.max())
        # mask_prev_frame called with first keyframe, which has shape 3, H, W
        flow = self.flow_model(prev_frame.img.unsqueeze(0), frame.img)
        plot_flow(axes[2], f"Computed Flow {frame.frame_id}", flow)

        residual = (flow - flow_exp).norm(dim=0)
        # residual = (residual - residual.median()).abs()
        plot_img(axes[3], f"Residual {frame.frame_id}", residual)
        
        # avg_residual = residual.clone()
        sam_masks, sam_result = self.segmentation(prev_frame)
        avg_residual = torch.zeros(H, W, dtype=float).to(self.device)
        background_mask = torch.ones(H, W, dtype=bool).to(self.device)
        for mask in sam_masks:
            background_mask[mask] = False
            avg_residual[mask] = torch.max(avg_residual[mask], residual[mask].mean())
        avg_residual[background_mask] = torch.max(avg_residual[background_mask], residual[background_mask].mean())
        avg_residual = avg_residual - avg_residual.min()

        plot_img(axes[4], f"Object Residual {frame.frame_id}", avg_residual)
        
        mask = (avg_residual > avg_residual.max()*0.1)
        # mask = (avg_residual >  torch.quantile(avg_residual, 0.70))

        # max_pool2d acts as dilation on a binary mask
        # We add batch and channel dims for the operator, then squeeze back
        dilation_size = 7 
        padding = dilation_size // 2
        mask = F.max_pool2d(
            mask.float().unsqueeze(0).unsqueeze(0), 
            kernel_size=dilation_size, 
            stride=1, 
            padding=padding
        ).bool().squeeze()
        
        plot_img(axes[5], f"Mask {frame.frame_id}", mask)
        
        save_plots(fig, f"debug_optical_flow_depth/{frame.frame_id:04d}.png")

        mask = mask.reshape(-1)
        prev_frame.N = 0 # set N to zero so update_pointmap treats this call as its first (so we won't be weighed against what is already there)
        C_masked = prev_frame.C.clone()
        C_masked[mask] = 0
        # X_masked = prev_frame.X_canon.clone()
        # X_masked[mask, 2] = 0
        prev_frame.update_pointmap(prev_frame.X_canon, C_masked)

        return mask

    def compute_mask(self, frame: Frame, sam_masks):
        offset = 4
        if len(self.frame_history) <= offset:
            self.mask_prev_frame(frame, self.keyframes.last_keyframe())
            return

        [[H, W]] = frame.img_true_shape
        prev_frame = self.frame_history[offset]

        fig, axes = start_plots(6)
        
        # Transform from curr frame to prev frame
        T = prev_frame.T_WC.inv() * frame.T_WC
        X_trans = T.act(frame.X_canon) # curr frame coords -> prev frame coords
        # X_trans = T.inv().act(prev_frame.X_canon) # curr frame coords -> prev frame coords
        depth = X_trans[:, 2]
        plot_img(axes[0], f"Depth {frame.frame_id}", depth.view(H, W))

        # Project to prev frame image plane
        K = estimate_K(prev_frame.X_canon, H, W)
        X_proj = K @ X_trans.T
        u_proj = (X_proj[0] / X_proj[2].clamp(min=1e-5))
        v_proj = (X_proj[1] / X_proj[2].clamp(min=1e-5))

        v_orig, u_orig = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )

        # Compute flow from projected points
        flow_exp = torch.stack([
            u_proj.view(H, W) - u_orig,
            v_proj.view(H, W) - v_orig,
        ], dim=0)  # (2, H, W)
        plot_flow(axes[1], f"Expected Flow {frame.frame_id}", flow_exp)

        flow = self.flow_model(frame.img, prev_frame.img)
        plot_flow(axes[2], f"Computed Flow {frame.frame_id}", flow)

        residual = (flow - flow_exp).norm(dim=0)
        # residual = (residual - residual.median()).abs()
        plot_img(axes[3], f"Residual {frame.frame_id}", residual)
        
        # avg_residual = residual.clone()
        avg_residual = torch.zeros(H, W, dtype=float).to(self.device)
        background_mask = torch.ones(H, W, dtype=bool).to(self.device)
        for mask in sam_masks:
            background_mask[mask] = False
            avg_residual[mask] = torch.max(avg_residual[mask], residual[mask].mean())
        avg_residual[background_mask] = torch.max(avg_residual[background_mask], residual[background_mask].mean())
        avg_residual = avg_residual - avg_residual.min()

        plot_img(axes[4], f"Object Residual {frame.frame_id}", avg_residual)
        
        mask = (avg_residual > avg_residual.max()*0.2)
        # mask = (avg_residual >  torch.quantile(avg_residual, 0.70))
        
        # max_pool2d acts as dilation on a binary mask
        # We add batch and channel dims for the operator, then squeeze back
        dilation_size = 7 
        padding = dilation_size // 2
        mask = F.max_pool2d(
            mask.float().unsqueeze(0).unsqueeze(0), 
            kernel_size=dilation_size, 
            stride=1, 
            padding=padding
        ).bool().squeeze()

        plot_img(axes[5], f"Mask {frame.frame_id}", mask)
        
        save_plots(fig, f"debug_optical_flow_depth/{frame.frame_id:04d}.png")

        return mask

    def track(self, frame: Frame):
        print(f"tracking frame {frame.frame_id}")
        keyframe = self.keyframes.last_keyframe()

        self.frame_history.insert(0, frame)
        self.frame_history = self.frame_history[:self.max_offset+1]

        # ====================================================================
        # 1. RUN FASTSAM ONCE FOR THE CURRENT FRAME
        # sam_masks, sam_result = self.segmentation(frame)
        
        with torch.no_grad():
            if len(self.frame_history) > self.max_offset and False:
                
                
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
                # Ultralytics has a built-in .plot() function that returns the overlay image in BGR
                fastsam_viz_bgr = sam_result.plot(
                    line_width=None,
                    boxes=False,
                    probs=False,
                    labels=False,
                    masks=True,
                    color_mode="instance"
                )
                axes[0, 1].imshow(fastsam_viz_bgr[..., ::-1]) # Convert BGR to RGB for matplotlib
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
                    predicted_flow = self.flow_model(img1, img2)
                    
                    # --- PLOT: RAW FLOW (Row 1) ---
                    raw_flow_tensor = flow_to_image(predicted_flow)
                    raw_flow_np = raw_flow_tensor.permute(1, 2, 0).cpu().numpy()
                    
                    axes[1, i].imshow(raw_flow_np)
                    axes[1, i].set_title(f"Raw Flow (t - {offset})")
                    axes[1, i].axis("off")
                    
                    # --- PLOT: AVERAGED FLOW (Row 2) ---
                    avg_object_flow = predicted_flow.clone()
                    
                    if sam_masks is not None:
                        h, w = avg_object_flow.shape[1], avg_object_flow.shape[2]
                        # Interpolate masks to match flow tensor dimensions
                        masks_resized = F.interpolate(
                            sam_masks.unsqueeze(1), size=(h, w), mode='nearest'
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

        # offset = 4
        # if len(self.frame_history) > offset and self.frame_history[offset].frame_id == keyframe.frame_id:
        #     print(f"masking keyframe {keyframe.frame_id}")
        #     self.mask_prev_frame(frame)

        # self.mask_prev_frame(frame)

        # sam_masks, sam_result = self.segmentation(frame)
        # mask = self.compute_mask(frame, sam_masks)
        # if mask is None:
        #     mask = torch.zeros(frame.X_canon.shape[0], dtype=bool) # if can't get mask, just don't save this 3D map
        # else:
        #     mask = mask.reshape(-1)
        # frame.N = 0 # set N to zero so update_pointmap treats this call as its first (so we won't be weighed against what is already there)
        # C_masked = frame.C.clone()
        # C_masked[mask] = 0
        # X_masked = frame.X_canon.clone()
        # print(X_masked.shape, mask.shape)
        # X_masked[mask, :] = X_masked[mask, :] * 1e10
        # frame.update_pointmap(X_masked, C_masked)

        sam_masks, sam_result = self.segmentation(frame)
        mask = self.compute_mask(frame, sam_masks)

        if mask is None:
            mask = torch.zeros(frame.X_canon.shape[0], dtype=bool, device=self.device)
        else:
            mask = mask.reshape(-1)

        # mask out points in current frame
        # frame.N = 0 # set N to zero so update_pointmap treats this call as its first (so we won't be weighed against what is already there)
        # # frame.C[mask] *= 0.1
        # C_masked = frame.C.clone()
        # C_masked[mask] = 0
        # frame.update_pointmap(frame.X_canon, C_masked)

        # mask out corresponding points in keyframe
        if self.prev_mask is not None:
            combined_mask = mask | self.prev_mask
            self.prev_mask = mask
            mask = combined_mask
        mask_in_kf = mask[idx_f2k]
        Ckf = Ckf.clone()
        Ckf[mask_in_kf] = 0 # Prevent NEW dynamic points from entering the map
        # keyframe.C[mask_in_kf] *= 0.5  # ACTIVE ERASURE: Degrade confidence of existing ghosts in the map
        keyframe.C[mask_in_kf] *= 0.1  # ACTIVE ERASURE: Degrade confidence of existing ghosts in the map

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
