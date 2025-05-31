import os
import copy
import time
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import imageio.v3 as iio

from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.utils import load_model, seed_everything
from seva.eval import (
    run_one_scene, 
    transform_img_and_K, 
    infer_prior_stats,
    chunk_input_and_test
)
from seva.geometry import get_default_intrinsics, get_preset_pose_fov, DEFAULT_FOV_RAD


class SevaAPI:
    """
    API wrapper for Stable Virtual Camera model inference.
    Provides clean interface for novel view synthesis without Gradio dependencies.
    """
    
    def __init__(self, device: str = "cuda", compile_model: bool = False):
        self.device = device
        self.compile_model = compile_model
        
        self._load_models()
        
        self.version_dict = {
            "H": 576,
            "W": 576, 
            "T": [21, 21],
            "C": 4,
            "f": 8,
            "options": self._get_default_options()
        }

    def _load_models(self):
        """Load all required model components."""
        print("Loading Stable Virtual Camera models...")
        
        self.model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(self.device)
        
        self.autoencoder = AutoEncoder(chunk_size=1).to(self.device)
        self.conditioner = CLIPConditioner().to(self.device)
        
        discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(
            discretization=discretization,
            num_idx=1000,
            device=self.device
        )
        
        if self.compile_model:
            self.model = torch.compile(self.model)
            self.conditioner = torch.compile(self.conditioner)
            self.autoencoder = torch.compile(self.autoencoder)
            
        print("Models loaded successfully!")
    
    def _get_default_options(self) -> Dict[str, Any]:
        """Get default inference options."""
        return {
            "chunk_strategy": "nearest",
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": [1, 2],
            "cfg": [3.0, 2.0],
            "camera_scale": 1.0,
            "num_steps": 50,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
            "transform_input": "crop",
            "transform_target": "crop", 
            "transform_scale": 1.0
        }

    def preprocess_single_image(self, image_path: str, shorter_side: int = 576) -> Dict[str, Any]:
        """
        Preprocess a single input image for basic novel view synthesis.
        
        Args:
            image_path: Path to input image
            shorter_side: Target size for shorter side (must be multiple of 64)
            
        Returns:
            Dictionary containing preprocessed data
        """
        shorter_side = round(shorter_side / 64) * 64
        
        input_img = torch.as_tensor(
            iio.imread(image_path) / 255.0, dtype=torch.float32
        )[None, ..., :3]
        
        input_img, _ = transform_img_and_K(
            input_img.permute(0, 3, 1, 2),
            shorter_side,
            K=None,
            size_stride=64,
        )
        input_img = input_img.permute(0, 2, 3, 1)
        
        input_K = get_default_intrinsics(
            aspect_ratio=input_img.shape[2] / input_img.shape[1]
        )[None]
        input_c2w = torch.eye(4)[None]
        
        return {
            "input_imgs": input_img,
            "input_Ks": input_K, 
            "input_c2ws": input_c2w,
            "input_wh": (input_img.shape[2], input_img.shape[1]),
            "points": [np.zeros((0, 3))],
            "point_colors": [np.zeros((0, 3))],
            "scene_scale": 1.0,
        }
    
    def preprocess_multi_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Preprocess multiple input images using DUST3R for camera estimation.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Dictionary containing preprocessed data
        """
        raise NotImplementedError("Multi-image preprocessing requires DUST3R integration")

    def generate_trajectory_cameras(
        self,
        input_c2w: torch.Tensor,
        input_K: torch.Tensor, 
        trajectory_type: Literal[
            "orbit", "spiral", "lemniscate", "zoom-in", "zoom-out",
            "dolly zoom-in", "dolly zoom-out", "move-forward", "move-backward",
            "move-up", "move-down", "move-left", "move-right"
        ],
        num_frames: int,
        zoom_factor: Optional[float] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate camera trajectory for novel view synthesis.
        
        Args:
            input_c2w: Input camera-to-world matrix [1, 4, 4]
            input_K: Input camera intrinsics [1, 3, 3] 
            trajectory_type: Type of camera trajectory
            num_frames: Number of frames to generate
            zoom_factor: Zoom factor for zoom trajectories
            
        Returns:
            Tuple of (target_c2ws, target_Ks) tensors
        """
        img_wh = (int(input_K[0, 0, 2].item() * 2), int(input_K[0, 1, 2].item() * 2))
        start_c2w = input_c2w[0]
        start_w2c = torch.linalg.inv(start_c2w)
        look_at = torch.tensor([0, 0, 10])
        start_fov = DEFAULT_FOV_RAD
        
        target_c2ws, target_fovs = get_preset_pose_fov(
            trajectory_type,
            num_frames,
            start_w2c,
            look_at,
            -start_c2w[:3, 1],
            start_fov,
            spiral_radii=[1.0, 1.0, 0.5],
            zoom_factor=zoom_factor,
        )
        target_c2ws = torch.as_tensor(target_c2ws)
        target_fovs = torch.as_tensor(target_fovs)
        target_Ks = get_default_intrinsics(
            target_fovs,
            aspect_ratio=img_wh[0] / img_wh[1],
        )
        if target_Ks.dim() == 2:
            target_Ks = target_Ks.unsqueeze(0)
        
        return target_c2ws, target_Ks

    def render_novel_views(
        self,
        preprocessed_data: Dict[str, Any],
        trajectory_type: Optional[str] = None,
        num_frames: Optional[int] = None,
        zoom_factor: Optional[float] = None,
        target_c2ws: Optional[torch.Tensor] = None,
        target_Ks: Optional[torch.Tensor] = None,
        cfg_scale: float = 3.0,
        seed: int = 42,
        chunk_strategy: str = "nearest"
    ) -> Dict[str, Any]:
        """
        Render novel views from preprocessed input data.
        
        Args:
            preprocessed_data: Output from preprocess_single_image or preprocess_multi_images
            trajectory_type: Type of camera trajectory (if not providing custom cameras)
            num_frames: Number of frames to generate
            zoom_factor: Zoom factor for zoom trajectories
            target_c2ws: Custom target camera poses [N, 4, 4]
            target_Ks: Custom target camera intrinsics [N, 3, 3]
            cfg_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            chunk_strategy: Chunking strategy for processing
            
        Returns:
            Dictionary containing rendered images and metadata
        """
        input_imgs = preprocessed_data["input_imgs"]
        input_Ks = preprocessed_data["input_Ks"] 
        input_c2ws = preprocessed_data["input_c2ws"]
        W, H = preprocessed_data["input_wh"]
        
        if target_c2ws is None or target_Ks is None:
            if trajectory_type is None or num_frames is None:
                raise ValueError("Must provide either target cameras or trajectory parameters")
            target_c2ws, target_Ks = self.generate_trajectory_cameras(
                input_c2ws, input_Ks, trajectory_type, num_frames, zoom_factor
            )
        
        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_Ks = torch.cat([input_Ks, target_Ks], 0)
        all_Ks = all_Ks * all_Ks.new_tensor([W, H, 1])[:, None]
        
        num_inputs = len(input_imgs)
        num_targets = len(target_c2ws)
        input_indices = list(range(num_inputs))
        target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()
        
        T = self.version_dict["T"]
        version_dict = copy.deepcopy(self.version_dict)
        num_anchors = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )
        T = version_dict["T"]
        assert isinstance(num_anchors, int)
        anchor_indices = np.linspace(
            num_inputs,
            num_inputs + num_targets - 1,
            num_anchors,
        )
        anchor_indices_int = [round(ind) for ind in anchor_indices]
        anchor_c2ws = all_c2ws[anchor_indices_int]
        anchor_Ks = all_Ks[anchor_indices_int]
        
        padded_imgs = F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0)
        img_array = (padded_imgs.numpy() * 255.0).astype(np.uint8)
        
        image_cond = {
            "img": img_array,
            "input_indices": input_indices,
            "prior_indices": anchor_indices_int
        }
        
        camera_cond = {
            "c2w": all_c2ws,
            "K": all_Ks,
            "input_indices": list(range(num_inputs + num_targets))
        }
        
        options = copy.deepcopy(self.version_dict["options"])
        options.update({
            "chunk_strategy": chunk_strategy,
            "cfg": [cfg_scale, 3.0 if num_inputs >= 9 else 2.0]
        })
        
        seed_everything(seed)
        
        results = run_one_scene(
            task="img2trajvid",
            version_dict={**self.version_dict, "H": H, "W": W, "options": options},
            model=self.model,
            ae=self.autoencoder,
            conditioner=self.conditioner,
            denoiser=self.denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=None,
            use_traj_prior=True,
            traj_prior_c2ws=anchor_c2ws,
            traj_prior_Ks=anchor_Ks,
            seed=seed,
            gradio=False
        )
        
        return {
            "rendered_images": results,
            "target_cameras": {"c2ws": target_c2ws, "Ks": target_Ks},
            "metadata": {"num_frames": num_targets, "seed": seed}
        }
