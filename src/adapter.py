import os
import torch
from safetensors import safe_open
import torch.nn.functional as F
from my_utils.cyclegan_turbo import load_unet_from_state_dict
from diffusers import AutoencoderKL, UNet2DConditionModel

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from attention_processor import (
        CAAttnProcessor2_0 as CAAttnProcessor,
    )
else:
    from attention_processor import AttnProcessor, CAAttnProcessor

class ProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_embeds = self.norm(clip_extra_context_tokens)

        return clip_embeds

class CAAdapter:
    def __init__(self, model_path, pretrained_path, device, num_tokens=4):
        self.device = device
        self.pretrained_path = pretrained_path
        self.model_path = model_path
        self.num_tokens = num_tokens
        self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_path, subfolder="unet").to(self.device)
        self.set_ca_adapter()
        self.proj_model = self.init_proj()
        self.load_ca_adapter()

    def init_proj(self):
        proj_model = ProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=self.num_tokens
        ).to(self.device, dtype=torch.float32)
        return proj_model

    def set_ca_adapter(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = CAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float32)
        unet.set_attn_processor(attn_procs)

    def load_ca_adapter(self):
        if os.path.splitext(self.model_path)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ca_adapter": {}}
            with safe_open(self.model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ca_adapter."):
                        state_dict["ca_adapter"][key.replace("ca_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.model_path, map_location="cpu")
        self.proj_model.load_state_dict(state_dict["image_proj"])
        ca_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ca_layers.load_state_dict(state_dict["ca_adapter"])

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, CAAttnProcessor):
                attn_processor.scale = scale
    
    def forward_infer(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ca_tokens = self.proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ca_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
