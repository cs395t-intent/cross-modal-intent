import argparse
import os
import json
import torch
import numpy as np

from datetime import datetime
from tqdm import tqdm
from PIL import Image

from dataset import get_transform
from model import *

DIR = '../data/2020intent/'
THRESHOLD = 0.3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', help='dataset to process', choices=['train', 'val', 'test'],
                        default='train')
    args = parser.parse_args()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device {device}.")

    models = []
    cwd = os.getcwd()
    paths = [
        ("tiny", "swin_tiny"),
        ("tiny", "swin_tiny_noweight"),
        ("small", "swin_small"),
        ("small", "swin_small_noweight")
    ]
    for model_type, name in paths:
        model_path = os.path.join(cwd, 'models', name, 'latest.pt')
        checkpoint = torch.load(model_path, map_location='cpu')
        model = SwinTransformerVisual(model_size=model_type, out_dim=997)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model.to(device).eval())

    # Vocab
    vocab = ["" for _ in range(997)]
    with open(os.path.join(cwd, "vocab_index.txt"), "r") as vf:
        vocablines = vf.readlines()
        for line in vocablines:
            token, idx = line.split()
            vocab[int(idx)] = token
    vocab = np.array(vocab, dtype=str)

    # Tagging
    transform = get_transform('test')
    annotation_path = os.path.join(DIR, "annotations", f"intentonomy_{args.set}2020.json")
    annotation_save_path = os.path.join(DIR, "annotations", f"intentonomy_{args.set}2020_ht.json")
    with open(annotation_path, "r") as f:
        data = json.load(f)
        for idx in tqdm(range(len(data["annotations"]))):
            img_path = os.path.join(DIR, "images", "low", data["annotations"][idx]["image_id"] + ".jpg")

            hashtags = []
            if os.path.exists(img_path):
                with torch.no_grad():
                    img = transform(Image.open(img_path))
                    img = img.unsqueeze(0).to(device)
                    preds = [torch.sigmoid(m(img)).squeeze(0) for m in models]
                    pred = sum(preds) / len(preds)
                    pred = pred.detach().cpu().numpy()
                    hashtags = vocab[pred >= THRESHOLD].tolist()
            else:
                print(f"WARNING: Ignoring {img_path} because it doesn't exist.")

            data['annotations'][idx]['hashtags'] = hashtags

        data['info']['date_hashtags_created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(annotation_save_path, "w") as wf:
            wf.write(json.dumps(data, indent=4))
