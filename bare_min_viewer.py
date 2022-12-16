
import os
from pathlib import Path

import torch
import yaml

from nerfstudio.data.scene_box import SceneBox

# from nerfstudio.utils.writer import GLOBAL_BUFFER
from nerfstudio.viewer.server.viewer_utils import ViewerState

log_filename=Path("bare_min_viewer/test.log")
device  = "cuda"

load_dir = Path("outputs/data-nerfstudio-posters_v3/nerfacto/2022-12-16_145240")
load_dir = Path("outputs/-mnt-ndata-data-scape_nerf2-shape02935_rank01_pair12589/instant-ngp/2022-12-16_212834")

# config = Config() # dummy
config =  yaml.load(open(Path(load_dir, "config.yml"),'r'), Loader=yaml.Loader) # yaml.load(config.trainer.load_config.read_text(), Loader=yaml.Loader)


viewer_config = config.viewer
viewer_config.skip_openrelay = True
viewer_config.websocket_port = 8008
viewer_config.local_viewer_port = 4000
viewer_state = ViewerState(viewer_config, log_filename=log_filename)
model_config = config.pipeline.model
# restore a graph
load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(Path(load_dir,"nerfstudio_models")))[-1]
load_path = load_dir / "nerfstudio_models"/ f"step-{load_step:09d}.ckpt"
loaded_state = torch.load(load_path, map_location="cpu")
model_state = loaded_state['pipeline']
model_state['model'] = {k.replace("_model.", ""): v for k,v in model_state.items() if "_model" in k}
try:
    num_train_data = model_state['model']['field.embedding_appearance.embedding.weight'].shape[0]
except:
    num_train_data = 200
model = model_config.setup(scene_box=SceneBox(model_state["model"]["field.aabb"]), num_train_data=num_train_data).to(device)

model.load_model(model_state)

# import types
# from collections import defaultdict
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Tuple, Type

# from nerfstudio.cameras.rays import RayBundle


# @torch.no_grad()
# def get_outputs_for_camera_ray_bundle2(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
#     """Takes in camera parameters and computes the output of the model.

#     Args:
#         camera_ray_bundle: ray bundle to calculate outputs over
#     """
#     num_rays_per_chunk = self.config.eval_num_rays_per_chunk
#     image_height, image_width = camera_ray_bundle.origins.shape[:2]
#     num_rays = len(camera_ray_bundle)
#     outputs_lists = defaultdict(list)
#     for i in range(0, num_rays, num_rays_per_chunk):
#         start_idx = i
#         end_idx = i + num_rays_per_chunk
#         ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
#         outputs = self.forward(ray_bundle=ray_bundle)
#         for output_name, output in outputs.items():  # type: ignore
#             outputs_lists[output_name].append(output)
#     outputs = {}
#     for output_name, outputs_list in outputs_lists.items():
#         if output_name == "rgb":
#             if not torch.is_tensor(outputs_list[0]):
#                 # TODO: handle lists of tensors as well
#                 continue
#             outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
#     return outputs




# model.get_outputs_for_camera_ray_bundle = types.MethodType( get_outputs_for_camera_ray_bundle2,model)
from nerfstudio.utils.writer import GLOBAL_BUFFER

GLOBAL_BUFFER["events"] = {}
GLOBAL_BUFFER["max_buffer_size"] = 20 #config.logging.max_buffer_size
num_rays_per_batch = config.pipeline.datamanager.train_num_rays_per_batch
step = 0

# model just calls get_outputs_for_camera_ray_bundle:
# RayBundle -> dict: rgb -> H,W,3, depth -> H,W,1, accumulation -> H,W,1, prop_depth_0, prop_depth_1

while True:
    try:
        viewer_state.update_scene(None, step, model, num_rays_per_batch)
    except RuntimeError:
        time.sleep(0.03)  # sleep to allow buffer to reset
        assert self.viewer_state.vis is not None
        self.viewer_state.vis["renderingState/log_errors"].write(
            "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
        )