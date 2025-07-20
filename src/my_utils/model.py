import os
import sys
import torch
from torch.nn import functional as F
from diffusers import DDPMScheduler
p = "src/"
sys.path.append(p)
from CLIP.clip import load, tokenize

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, clip_model='ViT-B/32', clip_model_path='./clip_model'):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, clip_preprocess = load(clip_model, device=self.device, download_root=clip_model_path)
        self.clip_preprocess = clip_preprocess

        self.cos = torch.nn.CosineSimilarity()
        self.model.requires_grad_(False)

    def tokenize(self, strings: list):
        return tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def text_encoder(self, tokens: list, tokenized_prompts) -> torch.Tensor:
        return self.model.text_encoder(tokens, tokenized_prompts)
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images, pos_embedding=False)
    
    def get_text_features(self, class_str: str, norm: bool = True) -> torch.Tensor:
        template_text = [class_str]
        tokens = tokenize(template_text).to(self.device)
        text_features = self.encode_text(tokens).detach()
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features, conv_feautures = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features, conv_feautures
    

def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample
