import argparse
import os
import json
import torch
import numpy as np

from datetime import datetime
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from dataset import get_transform

DIR = './data/2020intent/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', help='dataset to process', choices=['train', 'val', 'test'],
                        default='train')
    parser.add_argument('--type', help='embedding type', choices=['bert'],
                        default='bert')
    args = parser.parse_args()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device {device}.")

    # Tagging
    annotation_path = os.path.join(DIR, "annotations", f"intentonomy_{args.set}2020_ht.json")
    annotation_save_path = os.path.join(DIR, "annotations", f"intentonomy_{args.set}2020_ht_{args.type}.json")
    with open(annotation_path, "r") as f:
        data = json.load(f)
        if args.type == 'bert':
            model = BertModel.from_pretrained('bert-base-uncased')
            model.to(device).eval()
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            with torch.no_grad():
                for idx in tqdm(range(len(data["annotations"]))):
                    text = " ".join(data['annotations'][idx]['hashtags'])
                    inputs = tokenizer(text, return_tensors='pt')
                    for key in inputs:
                        inputs[key] = inputs[key].to(device)
                    outputs = model(**inputs)
                    cls_state = outputs.last_hidden_state.squeeze(0)[0]
                    embed = cls_state.detach().cpu().numpy()
                    data['annotations'][idx]['embed'] = embed.tolist()
        else:
            raise NotImplementedError("Other types of embeddings are not implemented.")

        data['info']['date_embed_created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(annotation_save_path, "w") as wf:
            wf.write(json.dumps(data, indent=4))
