import os
import gc
import copy
import lpips
import torch
import wandb
import torch.nn.functional as F
import torch.nn as nn
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from diffusers import UNet2DConditionModel
import vision_aided_loss
from my_utils.model import CLIPLoss, make_1step_sched
from my_utils.cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, REF_vae, initialize_vae
from my_utils.training_utils import UnpairedCaptionDataset, build_transform, parse_args_refine_training
from my_utils.enhancer import RetinexNet
from torchmetrics.functional import structural_similarity_index_measure as ssim

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from attention_processor import CAAttnProcessor2_0 as CAAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from attention_processor import CAAttnProcessor as CAAttnProcessor, AttnProcessor as AttnProcessor

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

class CAAdapter(torch.nn.Module):
    """CA-Adapter"""
    def __init__(self, unet, proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.proj_model = proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ca_tokens = self.proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ca_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ca_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ca_adapter"], strict=True)

        # Calculate new checksums
        new_ca_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ca_proj_sum != new_ca_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

def main(args):

    if args.report_to != 'None':
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    else:
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_diffusion_model, subfolder="tokenizer", revision=args.revision, use_fast=False,)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_diffusion_model, subfolder="text_encoder").cuda()
    vae_a2b, vae_lora_target_modules = initialize_vae(args.pretrained_diffusion_model, args.lora_rank_vae, return_lora_module_names=True)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_diffusion_model, subfolder="unet")
    unet_distill = copy.deepcopy(unet)
    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet_distill.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet_distill.requires_grad_(False)

    net_enh = RetinexNet()
    net_enh.load_state_dict(torch.load(args.pretrained_retinexnet_path + 'decomp.pth'))
    net_enh.to(accelerator.device, dtype=weight_dtype)
    net_enh.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_a.cv_ensemble.requires_grad_(False)
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    vae_b2a = copy.deepcopy(vae_a2b)
    vae_rex = copy.deepcopy(vae_a2b)
    params_vae = CycleGAN_Turbo.get_traininable_params_vae_ref(vae_a2b, vae_b2a, vae_rex)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)
    vae_ref = REF_vae(vae_a2b, vae_b2a=vae_b2a, vae_ref=vae_rex)
    
    CLIPloss = CLIPLoss(device=accelerator.device, clip_model='ViT-B/32', clip_model_path=args.clip_model_path)
    COSloss = torch.nn.CosineSimilarity(dim=1, eps=1e-07)

    proj_model = ProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    )

    # init adapter modules
    attn_procs = {}  
    unet_sd = unet.state_dict()

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
            layer_name = name.split(".processor")[0]
            custom_weight = torch.randn(hidden_size, 4)
            weights = {
                "to_q_ca.weight": nn.Parameter(custom_weight).to(dtype=torch.float32),
                "to_k_ca.weight": unet_sd[layer_name + ".to_k.weight"].to(dtype=torch.float32),
                "to_v_ca.weight": unet_sd[layer_name + ".to_v.weight"].to(dtype=torch.float32),
            }
            attn_procs[name] = CAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=4)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ca_adapter = CAAdapter(unet, proj_model, adapter_modules, args.pretrained_ca_adapter_path)
    ca_adapter.to(dtype=weight_dtype)

    # *******************************************resume***********************************
    if args.resume_path:
        print("load resume")
        sd = torch.load(args.resume_path)
        vae_enc.load_state_dict(sd["sd_vae_enc"])
        vae_dec.load_state_dict(sd["sd_vae_dec"])
        vae_ref.load_state_dict(sd["sd_vae_ref"])
        ca_adapter = CAAdapter(unet, proj_model, adapter_modules, args.resume_path) 

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)
    
    params_gen = params_vae + list(ca_adapter.proj_model.parameters()) + list(ca_adapter.adapter_modules.parameters())
    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    dataset_train = UnpairedCaptionDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)

    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_lsrw", ext)))
    
    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    # del text_encoder, tokenizer  # free up some memory

    # Prepare everything with our `accelerator`
    ca_adapter, unet, vae_enc, vae_dec, vae_ref, net_disc_a, net_disc_b = accelerator.prepare(ca_adapter, unet, vae_enc, vae_dec, vae_ref, net_disc_a, net_disc_b)
    optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    
    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    net_enh.eval()
    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, unet_distill, net_disc_a, net_disc_b, vae_enc, vae_dec, vae_ref, ca_adapter, net_enh]
            with accelerator.accumulate(*l_acc):

                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                img_net_a = batch["img_net_src"].to(dtype=weight_dtype)
                img_net_b = batch["img_net_tgt"].to(dtype=weight_dtype)
                img_a_v = batch["src_v"].to(dtype=weight_dtype)
                img_b_v = batch["tgt_v"].to(dtype=weight_dtype)
                img_a_v_r = batch["src_v_r"].to(dtype=weight_dtype)
                img_b_v_r = batch["tgt_v_r"].to(dtype=weight_dtype)
                src_caption = batch["src_caption"]
                tgt_caption = batch["tgt_caption"]
                img_a_v_3 = img_a_v.repeat(1,3,1,1)
                img_b_v_3 = img_b_v.repeat(1,3,1,1)
                img_a_v_r_3 = img_a_v_r.repeat(1,3,1,1)
                img_b_v_r_3 = img_b_v_r.repeat(1,3,1,1)
                R_a_gt, I_a = net_enh(img_net_a)
                R_b_gt, I_b = net_enh(img_net_b)

                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device)

                img_a_emb, _ = CLIPloss.get_image_features(img_a_v_3)
                img_a_emb = img_a_emb.to(dtype=weight_dtype).detach()
                img_b_emb, _ = CLIPloss.get_image_features(img_b_v_3)
                img_b_emb = img_b_emb.to(dtype=weight_dtype).detach()
                img_a_r_emb, _ = CLIPloss.get_image_features(img_a_v_r_3)
                img_a_r_emb = img_a_r_emb.to(dtype=weight_dtype).detach()
                img_b_r_emb, _ = CLIPloss.get_image_features(img_b_v_r_3)
                img_b_r_emb = img_b_r_emb.to(dtype=weight_dtype).detach()

                src_caption_tokens = tokenizer(src_caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
                src_caption_emb_base = text_encoder(src_caption_tokens.cuda().unsqueeze(0))[0].detach()
                tgt_caption_tokens = tokenizer(tgt_caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
                tgt_caption_emb_base = text_encoder(tgt_caption_tokens.cuda().unsqueeze(0))[0].detach()
                src_caption_emb = src_caption_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                tgt_caption_emb = tgt_caption_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                
                """
                Cycle Objective
                """
                # A -> fake B -> rec A
                src_caption_out = CycleGAN_Turbo.forward_unet_out(img_a, "a2b", vae_enc, unet_distill, noise_scheduler_1step, timesteps, src_caption_emb)
                cyc_fake_b, cyc_fake_b_out, cyc_fake_b_R = CycleGAN_Turbo.forward_adapter_dec(img_a, "a2b", vae_enc, vae_dec, vae_ref, noise_scheduler_1step, timesteps, fixed_a2b_emb, ca_adapter, img_a_r_emb)
                cyc_rec_a, cyc_rec_a_out, cyc_rec_a_R = CycleGAN_Turbo.forward_adapter_dec(cyc_fake_b, "b2a", vae_enc, vae_dec, vae_ref, noise_scheduler_1step, timesteps, fixed_b2a_emb, ca_adapter, img_a_emb)                    
                
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips  
                loss_retx_a = F.l1_loss(cyc_rec_a_R, R_a_gt.detach()) + (1. - ssim(cyc_rec_a_R, R_a_gt.detach())) + net_enh.enhance_loss(cyc_fake_b_R, cyc_rec_a_R, img_a, I_a.detach())
                loss_cycle_a += loss_retx_a 
                score = COSloss(cyc_rec_a_out, src_caption_out)
                loss_distill_a = 1.0 - torch.mean(score)
                loss_cycle_a += loss_distill_a
                accelerator.backward(loss_cycle_a, retain_graph=False)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Cycle Objective
                """
                # B -> fake A -> rec B
                tgt_caption_out = CycleGAN_Turbo.forward_unet_out(img_b, "b2a", vae_enc, unet_distill, noise_scheduler_1step, timesteps, tgt_caption_emb)
                cyc_fake_a, cyc_fake_a_out, cyc_fake_a_R = CycleGAN_Turbo.forward_adapter_dec(img_b, "b2a", vae_enc, vae_dec, vae_ref, noise_scheduler_1step, timesteps, fixed_b2a_emb, ca_adapter, img_b_r_emb)                                  
                cyc_rec_b, cyc_rec_b_out, cyc_rec_b_R = CycleGAN_Turbo.forward_adapter_dec(cyc_fake_a, "a2b", vae_enc, vae_dec, vae_ref, noise_scheduler_1step, timesteps, fixed_a2b_emb, ca_adapter, img_b_emb)
                
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips  
                loss_retx_b = F.l1_loss(cyc_rec_b_R, R_b_gt.detach()) + (1. - ssim(cyc_rec_b_R, R_b_gt.detach())) + net_enh.enhance_loss(cyc_fake_a_R, cyc_rec_b_R, img_b, I_b.detach())
                loss_cycle_b +=  loss_retx_b 
                score = COSloss(cyc_rec_b_out, tgt_caption_out)
                loss_distill_b = 1.0 - torch.mean(score)
                loss_cycle_b += loss_distill_b
                accelerator.backward(loss_cycle_b, retain_graph=False)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """
                fake_a, _ = CycleGAN_Turbo.forward_adapter(img_b, "b2a", vae_enc, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb, ca_adapter, img_b_r_emb)
                fake_b, _ = CycleGAN_Turbo.forward_adapter(img_a, "a2b", vae_enc, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb, ca_adapter, img_a_r_emb)
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                loss_gan = loss_gan_a + loss_gan_b
                accelerator.backward(loss_gan, retain_graph=False)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Identity Objective
                """
                src_caption_out = CycleGAN_Turbo.forward_unet_out(img_a, "a2b", vae_enc, unet_distill, noise_scheduler_1step, timesteps, src_caption_emb)
                tgt_caption_out = CycleGAN_Turbo.forward_unet_out(img_b, "b2a", vae_enc, unet_distill, noise_scheduler_1step, timesteps, tgt_caption_emb)

                idt_a, idt_a_out, idt_a_R = CycleGAN_Turbo.forward_adapter_dec(img_b, "a2b", vae_enc, vae_dec, vae_ref, noise_scheduler_1step, timesteps, fixed_a2b_emb, ca_adapter, img_b_emb)
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips  

                idt_b, idt_b_out, idt_b_R = CycleGAN_Turbo.forward_adapter_dec(img_a, "b2a", vae_enc, vae_dec, vae_ref, noise_scheduler_1step, timesteps, fixed_b2a_emb, ca_adapter, img_a_emb)
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips  

                loss_ref_2 = F.l1_loss(idt_a_R, R_b_gt.detach()) + F.l1_loss(idt_b_R, R_a_gt.detach()) + (1. - ssim(idt_a_R, R_b_gt.detach())) + (1. - ssim(idt_b_R, R_a_gt.detach()))

                score = COSloss(idt_a_out, tgt_caption_out)
                loss_distill_a_1 = 1.0 - torch.mean(score)
                score = COSloss(idt_b_out, src_caption_out)
                loss_distill_b_1 = 1.0 - torch.mean(score)

                loss_g_idt = loss_idt_a + loss_idt_b + loss_distill_a_1 + loss_distill_b_1 + loss_ref_2 
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                
                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) 
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) 
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {}
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["distill"] = loss_distill_b_1.detach().item() + loss_distill_b.detach().item() + loss_distill_a.detach().item() + + loss_distill_a_1.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["disc_a"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs["disc_b"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs["idt_a"] = loss_idt_a.detach().item()
            logs["idt_b"] = loss_idt_b.detach().item()
            logs["ref"] =  loss_ref_2.detach().item() + loss_retx_a.detach().item() + loss_retx_b.detach().item() 

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_ref = accelerator.unwrap_model(vae_ref)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    eval_ca_adapter = accelerator.unwrap_model(ca_adapter)
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                }
                                log_dict["train/rec_a"] = [wandb.Image(cyc_rec_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/rec_b"] = [wandb.Image(cyc_rec_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_b"] = [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_a"] = [wandb.Image(fake_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        sd["sd_vae_ref"] = eval_vae_ref.state_dict()
                        sd["image_proj"] = proj_model.state_dict()
                        sd["ca_adapter"] = adapter_modules.state_dict()
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    if global_step % args.validation_steps == 1:
                        _timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device="cuda")
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)

                        # get val input images from domain a
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if idx > args.validation_num_images and args.validation_num_images > 0:
                                break
                            file_name = os.path.basename(input_img_path)
                            outf = os.path.join(fid_output_dir, file_name)
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_hsv = input_img.convert('HSV')
                                img_a = transforms.ToTensor()(input_img)
                                img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
                                img_t_hsv = transforms.ToTensor()(img_hsv).unsqueeze(0).cuda()
                                img_t_v_r = 1 - img_t_hsv[:,2,:,:]
                                img_t_v_r_3 = img_t_v_r.repeat(1,3,1,1)
                                img_v_emb, _ = CLIPloss.get_image_features(img_t_v_r_3)
                                img_a = img_a.to(dtype=weight_dtype)
                                img_v_emb = img_v_emb.to(dtype=weight_dtype)

                                fake_b_val, _ = CycleGAN_Turbo.forward_adapter(img_a, "a2b", eval_vae_enc, eval_vae_dec, noise_scheduler_1step, 
                                                                            _timesteps, fixed_a2b_emb[0:1], eval_ca_adapter, img_v_emb)
                                eval_fake_b_pil = transforms.ToPILImage()(fake_b_val[0] * 0.5 + 0.5)
                                eval_fake_b_pil.save(outf)

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_refine_training()
    main(args)