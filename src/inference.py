import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from my_utils.cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from adapter import CAAdapter
from transformers import AutoTokenizer
from my_utils.model import make_1step_sched, CLIPLoss
from tqdm import tqdm 

def get_image_paths(folder_path):
    image_paths = []  
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    for root, dirs, files in os.walk(folder_path):  
        for file in files:  
            if os.path.splitext(file)[1].lower() in valid_image_extensions:  
                image_path = os.path.join(root, file)  
                image_paths.append(image_path)  
    return image_paths  

def resize_image_to_nearest_factor(image, factor=8):
    width, height = image.size
    new_width = (width // factor) * factor if width % factor == 0 else ((width // factor) + 1) * factor
    new_height = (height // factor) * factor if height % factor == 0 else ((height // factor) + 1) * factor
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='path to the input path')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='no_resize', help='the image preparation method')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument("--pretrained_diffusion_model", default="stabilityai/sd-turbo")
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_diffusion_model, subfolder="tokenizer")
    sched = make_1step_sched()
    cycle_model = CycleGAN_Turbo(model_path=args.model_path, pretrained_path=args.pretrained_diffusion_model, tokenizer=tokenizer, scheduler=sched)
    cycle_model.eval()

    adapter = CAAdapter(model_path=args.model_path, pretrained_path=args.pretrained_diffusion_model, device='cuda')
    CLIPloss = CLIPLoss('cuda', clip_model ='ViT-B/32')

    T_val = build_transform(args.image_prep)
    image_paths = get_image_paths(args.input_file)

    import time
    start_time = time.time()
    length = len(image_paths)
    for path in tqdm(image_paths):  
        input_image = Image.open(path).convert('RGB')
        resize_img = resize_image_to_nearest_factor(input_image)
        with torch.no_grad():
            resize_img = T_val(resize_img)
            x_t = transforms.ToTensor()(resize_img)
            img_hsv = resize_img.convert('HSV')
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            img_t_hsv = transforms.ToTensor()(img_hsv).unsqueeze(0).cuda()
            img_t_v_r = (1 - img_t_hsv[:,2,:,:])
            img_t_v_r_3 = img_t_v_r.repeat(1,3,1,1)
            img_v_emb, _ = CLIPloss.get_image_features(img_t_v_r_3)
            img_v_emb = img_v_emb.to(dtype=torch.float32)
            output = cycle_model.forward_infer(x_t, direction=args.direction, caption=args.prompt, ca_adapter=adapter, img_emb=img_v_emb)
        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)
        # save the output image
        bname = os.path.basename(path)
        os.makedirs(args.output_dir, exist_ok=True)
        output_pil.save(os.path.join(args.output_dir, bname))
