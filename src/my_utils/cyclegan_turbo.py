import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
import torch.nn.functional as F
p = "src/"
sys.path.append(p)
from my_utils.model import my_vae_encoder_fwd, my_vae_decoder_fwd
    
class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded

class REF_vae(nn.Module):
    def __init__(self, vae, vae_b2a=None, vae_ref=None):
        super(REF_vae, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a
        self.vae_ref = vae_ref
        self.sig = nn.Sigmoid()

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        self.vae_ref.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = self.sig(self.vae_ref.decode(x / self.vae_ref.config.scaling_factor).sample)
        return x_decoded
    
def initialize_unet(model_path, rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet

def initialize_vae(model_path, rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False #_____________________________________________
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]

    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae

def load_unet_from_state_dict(model_path, pretrained_path, load_module=True):
    sd = torch.load(pretrained_path)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
    lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
    lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    for n, p in unet.named_parameters():
        name_sd = n.replace(".default_encoder.weight", ".weight")
        if "lora" in n and "default_encoder" in n:
            p.data.copy_(sd["sd_encoder"][name_sd])
    for n, p in unet.named_parameters():
        name_sd = n.replace(".default_decoder.weight", ".weight")
        if "lora" in n and "default_decoder" in n:
            p.data.copy_(sd["sd_decoder"][name_sd])
    for n, p in unet.named_parameters():
        name_sd = n.replace(".default_others.weight", ".weight")
        if "lora" in n and "default_others" in n:
            p.data.copy_(sd["sd_other"][name_sd])
    unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
    if load_module:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet

class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, model_path=None, pretrained_path=None, tokenizer=None, scheduler=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder").cuda()
        self.sched = scheduler
        vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae
    
        sd = torch.load(model_path)
        self.load_vae_ckpt_from_state_dict(sd)
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.caption = None
        self.direction = None

        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def load_vae_ckpt_from_state_dict(self, sd):
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])
    
    @staticmethod
    def get_traininable_params_vae_ref(vae_a2b, vae_b2a, vae_ref):     
        params_gen = list()   
        # add all vae_a2b parameters
        for n,p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        for n,p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())

        # add all vae_ref parameters
        if vae_ref is not None:
            for n,p in vae_ref.decoder.named_parameters():
                if "lora" in n and "vae_skip" in n:
                    assert p.requires_grad
                    params_gen.append(p)
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
        return params_gen

    @staticmethod
    def forward_unet_out(x, direction, vae_enc, unet, sched, timesteps, text_emb, mask=None):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        if mask is not None:
            x_enc = x_enc * mask
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        return x_out

    @staticmethod
    def forward_adapter_dec(x, direction, vae_enc, vae_dec, ref_dec, sched, timesteps, text_emb, ca_adapter, img_emb):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = vae_enc(x, direction=direction)
        model_pred = ca_adapter(x_enc, timesteps, text_emb, img_emb)
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_ref = ref_dec(x_out, direction=direction)
        x_out_decoded = vae_dec(x_out, direction=direction)
        return x_out_decoded, x_out, x_ref
    
    @staticmethod
    def forward_adapter(x, direction, vae_enc, vae_dec, sched, timesteps, text_emb, ca_adapter, img_emb):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred = ca_adapter(x_enc, timesteps, text_emb, img_emb)
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        return x_out_decoded, x_out

    @staticmethod
    def forward_adapter_infer(x, direction, vae_enc, vae_dec, sched, timesteps, text_emb, ca_adapter, img_emb):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred = ca_adapter.forward_infer(x_enc, timesteps, text_emb, img_emb)
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        return x_out_decoded
    
    def forward_infer(self, x_t, direction=None, caption=None, caption_emb=None, ca_adapter=None, img_emb=None):
        if direction is None:
            assert self.direction is not None
            direction = self.direction
        if caption is None and caption_emb is None:
            assert self.caption is not None
            caption = self.caption
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
        return self.forward_adapter_infer(x_t, direction, self.vae_enc, self.vae_dec, self.sched, self.timesteps, caption_enc, ca_adapter, img_emb)