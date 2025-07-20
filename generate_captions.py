import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from clip_interrogator import Config, Interrogator

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# Default night-to-day mapping
NIGHT2DAY = {
    'night': 'day',
    'evening': 'morning',
    'dark': 'bright',
    'dim': 'clear',
    'moonlit': 'sunny',
    'shadows': 'sunlight',
    'nighttime': 'daytime',
    'low-light': 'normal-light',
    'low light': 'normal light'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate image captions with CLIP.")
    parser.add_argument('--dataroot', type=str, required=True, help='Path to input images.')
    parser.add_argument('--clip_model_path', type=str, required=True, help='Path to CLIP model.')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save output CSV.')
    parser.add_argument('--output_name', type=str, default='captions.csv', help='Output CSV file name.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id (e.g., 0).')
    return parser.parse_args()


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def get_image_paths(root):
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for fname in fnames:
            if is_image_file(fname):
                paths.append(os.path.join(dirpath, fname))
    if not paths:
        raise RuntimeError(f"No valid images found in {root}")
    return sorted(paths)


def replace_night_to_day(text, mapping):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def generate_captions(dataroot, save_path, clip_model_path, gpu, output_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai",
                             clip_model_path=clip_model_path))

    image_paths = get_image_paths(dataroot)
    records = {"filepath": [], "caption": []}

    for image_path in tqdm(image_paths, desc="Generating captions"):
        image = Image.open(image_path).convert('RGB')
        caption = ci.generate_caption(image)
        caption = replace_night_to_day(caption, NIGHT2DAY)
        records["filepath"].append(image_path)
        records["caption"].append(caption)

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, output_name)
    pd.DataFrame(records).to_csv(output_file, index=False, sep="\t")
    print(f"Saved captions to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    generate_captions(args.dataroot, args.save_path, args.clip_model_path, args.gpu, args.output_name)
