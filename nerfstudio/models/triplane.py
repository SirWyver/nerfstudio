# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TensorRF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.encoding import get_encoder
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import NeRFEncoding, TensorVMEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.tensorf_field import TensoRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

### from torch ngp



@dataclass
class TriplaneModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = field(default_factory=lambda: TriplaneModel)
    # """target class to instantiate"""
    # resolution: int = 64
    # """specifies a list of iteration step numbers to perform upsampling"""
    # loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})
    # """Loss specific weights."""
    # num_samples: int = 256
    # """Number of samples in field evaluation"""
    # num_den_components: int = 16
    # """Number of components in density encoding"""
    # num_color_components: int = 48
    # """Number of components in color encoding"""
    # appearance_dim: int = 27
    # """Number of channels for color encoding"""

    encoding_dir="sphere_harmonics"
    resolution=[64] * 3
    triplane_dim=12
    num_layers=3
    hidden_dim=64
    geo_feat_dim=15
    num_layers_color=3
    hidden_dim_color=64
    bound=1
    sigma_exp_scale=False
    normalize_triplane=False
    sigma_activation="trunc_exp"
    num_samples=128

class TriplaneModel(Model):
    """Triplane Model

    Args:
        config: Triplane configuration to instantiate model
    """

    def __init__(
        self,
        config: TriplaneModelConfig,
        **kwargs,
    ) -> None:
        self.resolution = config.resolution
        self.triplane_dim = config.triplane_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.geo_feat_dim = config.geo_feat_dim
        self.bound = config.bound
        self.num_samples = config.num_samples

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        self.triplane = self.init_one_svd(3*[self.triplane_dim], self.resolution)
        self.in_dim = 3*self.triplane_dim


        # render module (default to freq feat + freq dir)
        self.num_layers = config.num_layers
        self.hidden_dim = confighidden_dim

        sigma_net = []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == self.num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = self.hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = self.num_layers_color        
        self.hidden_dim_color = self.hidden_dim_color 
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color 
            
            if l == self.num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color 
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        self.bg_net = None

        self.sigma_exp_scale = config.sigma_exp_scale
        self.normalize_triplane = config.normalize_triplane
        self.sigma_activation = config.sigma_activation
        super().__init__(config=config, **kwargs)

    def init_one_svd(self, n_component, resolution, scale=0.1):
        mat = []
        for i in range(len(self.vec_ids)):
            mat_id_0, mat_id_1 = self.mat_ids[i]
            mat.append(nn.Parameter(scale * torch.randn((1, n_component[i], resolution[mat_id_1], resolution[mat_id_0])))) # [1, R, H, W]

        return nn.ParameterList(mat)



    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples // 2, single_jitter=True)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["fields"] = (
            list( self.triplane.parameters())
            + list(self.sigma_net.parameters())
            + list(self.color_net.parameters())
        )
        param_groups["encodings"] = list( self.encoder_dir.parameters()) 

        return param_groups

    def get_encoder_feat(self, x, bound):
        # x: [N, 3], in [-bound, bound]
        x = x / bound # map to [0, 1]

        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]
        mat_feat= []

        for i in range(len(self.triplane)):
            mat_feat.append(F.grid_sample(self.triplane[i], mat_coord[[i]], align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
        
        mat_feat = torch.cat(mat_feat, dim=0) # [3 * R, N]
        if self.normalize_triplane:
            mat_feat = 15*torch.clamp(mat_feat,-1,1)
        return mat_feat.T
    
    
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.get_encoder_feat(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        if self.sigma_activation == 'trunc_exp':
            sigma = trunc_exp(h[..., 0])
        else:
            sigma= h[...,0]
        if self.sigma_exp_scale:
            sigma = 10000*sigma
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color


    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.get_encoder_feat(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        
        if self.sigma_activation == 'trunc_exp':
            sigma = trunc_exp(h[..., 0])
        else:
            sigma= h[...,0]
        if self.sigma_exp_scale:
            sigma = 10000*sigma
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }



    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict
