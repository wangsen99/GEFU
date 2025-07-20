import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
import pandas as pd
import numpy as np

def parse_args_refine_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--input_filename", default=None, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument("--pretrained_diffusion_model", default="stabilityai/sd-turbo")
    parser.add_argument("--pretrained_ca_adapter_path", type=str, default=None)
    parser.add_argument("--pretrained_retinexnet_path", type=str, default="retinextnet/")
    parser.add_argument('--clip_model_path', type=str, default="clip_model/", help='path to a local model state dict to be used')
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--prompt_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--resume_path', type=str, default=None, help='path to a local model state dict to be used')

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=100,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=100)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args

def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    elif image_prep in ["resize_320"]:
        T = transforms.Compose([
            transforms.Resize((320, 320), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "randomcrop_hflip":
        T = transforms.Compose([
            transforms.Resize((384, 384), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    return T
    
class UnpairedCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, image_prep, img_key='filepath', caption_key='title', sep="\t", tokenizer=None):
        super().__init__()

        self.source_file = os.path.join(dataset_folder, "enlight_low.csv")
        self.target_file = os.path.join(dataset_folder, "enlight_high.csv")

        source_df = pd.read_csv(self.source_file, sep=sep)
        target_df = pd.read_csv(self.target_file, sep=sep)

        self.l_imgs_src = source_df[img_key].tolist()
        self.src_captions = source_df[caption_key].tolist()
        self.l_imgs_tgt = target_df[img_key].tolist()
        self.tgt_captions = target_df[caption_key].tolist()
        self.tokenize = tokenizer

        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
        self.T = build_transform(image_prep)

        sorted(self.l_imgs_src)
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src)

    def __getitem__(self, index):
        idx = random.randint(0, len(self.l_imgs_tgt)-1)
        img_path_tgt = self.l_imgs_tgt[idx]
        img_path_src = self.l_imgs_src[index]

        
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")

        img_t_src = self.T(img_pil_src)
        img_t_tgt = self.T(img_pil_tgt)

        img_src_hsv = img_t_src.convert('HSV')
        img_tgt_hsv = img_t_tgt.convert('HSV')

        img_t_src = F.to_tensor(img_t_src)
        img_t_tgt = F.to_tensor(img_t_tgt)
        img_net_src = img_t_src
        img_net_tgt = img_t_tgt
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])

        img_t_src_hsv = F.to_tensor(img_src_hsv)
        img_t_tgt_hsv = F.to_tensor(img_tgt_hsv)
        img_t_src_v_r = 1 - img_t_src_hsv[2,:,:]
        img_t_src_v = img_t_src_hsv[2,:,:]
        img_t_tgt_v = img_t_tgt_hsv[2,:,:]
        img_t_tgt_v_r = 1- img_t_tgt_hsv[2,:,:]

        src_texts = str(self.src_captions[index])
        tgt_texts = str(self.tgt_captions[idx])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "img_net_src": img_net_src,
            "img_net_tgt": img_net_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "src_v_r" : img_t_src_v_r,
            "src_v" : img_t_src_v,
            "tgt_v" : img_t_tgt_v,
            "tgt_v_r" : img_t_tgt_v_r,
            "src_caption" : src_texts,
            "tgt_caption": tgt_texts
        }
