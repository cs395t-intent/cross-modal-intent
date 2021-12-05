import argparse
import json
import os
import torch
import numpy as np

from sklearn.metrics import precision_score, f1_score
from tqdm import tqdm

from dataset import Dataset, get_transform
from model import *


def initialize(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device {device}.")

    # Data paths
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, 'HARRISON')

    # Get validation set
    val_dataset = Dataset(img_dir, get_transform(type='val'), type='val')
    #val_dataset = torch.utils.data.Subset(val_dataset, range(64))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=6, drop_last=False)

    # Model loading
    checkpoint = None
    metadata = {}
    if args.model_type == 'ensemble':
        checkpoint = []
        metadata = []
        for model_str in args.name.split(","):
            model_tag = str(args.load_tag) if isinstance(args.load_tag, int) else args.load_tag
            model_path = os.path.join(cwd, 'models', model_str.split(":")[1], model_tag + ".pt")
            if os.path.exists(model_path):
                checkpoint.append(torch.load(model_path, map_location='cpu'))
                print(f"Found model {model_path}, which was trained for {checkpoint[-1]['epochs_trained']} epochs.")
                metadata.append(checkpoint[-1]['metadata'])
            else:
                raise ValueError(f"Model {model_path} does not exist.")
    else:
        checkpoint = None
        metadata = {}
        model_tag = str(args.load_tag) if isinstance(args.load_tag, int) else args.load_tag
        model_path = os.path.join(cwd, 'models', args.name, model_tag + ".pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Found model {model_path}, which was trained for {checkpoint['epochs_trained']} epochs.")
            metadata = checkpoint['metadata']
        else:
            raise ValueError(f"Model {model_path} does not exist.")

    # Model specification
    if args.model_type == 'ensemble':
        model = []
        model_strs = args.name.split(",")
        for idx in range(len(model_strs)):
            model_str = model_strs[idx]
            model_type = model_str.split(":")[0]
            if model_type == 'vis_baseline':
                model.append(ResNetVisualBaseline(out_dim=997))
            elif model_type == 'vis_virtex':
                model.append(VirtexVisual(out_dim=997))
            elif model_type == 'vis_swin_tiny':
                model.append(SwinTransformerVisual(model_size='tiny', out_dim=997))
            elif model_type == 'vis_swin_small':
                model.append(SwinTransformerVisual(model_size='small', out_dim=997))
            elif model_type == 'vis_swin_base':
                model.append(SwinTransformerVisual(model_size='base', out_dim=997))
            else:
                raise NotImplementedError("Other models are not implemented.")
            model[-1].load_state_dict(checkpoint[idx]['model_state_dict'])
    else:
        model = None
        if args.model_type == 'vis_baseline':
            model = ResNetVisualBaseline(out_dim=997)
        elif args.model_type == 'vis_virtex':
            model = VirtexVisual(out_dim=997)
        elif args.model_type == 'vis_swin_tiny':
            model = SwinTransformerVisual(model_size='tiny', out_dim=997)
        elif args.model_type == 'vis_swin_small':
            model = SwinTransformerVisual(model_size='small', out_dim=997)
        elif args.model_type == 'vis_swin_base':
            model = SwinTransformerVisual(model_size='base', out_dim=997)
        else:
            raise NotImplementedError("Other models are not implemented.")
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])

    return val_dataloader, model, device, metadata, cwd


def validate_threshold(dataloader, model, device, num=20):
    if isinstance(model, list):
        for idx in range(len(model)):
            model[idx] = model[idx].to(device).eval()
    else:
        model = model.to(device).eval()
    preds, targets = [], []
    print("Getting predictions...")
    with torch.no_grad():
        for batch, labels, _ in tqdm(dataloader):
            batch = batch.to(device)
            if isinstance(model, list):
                batch_preds = [torch.sigmoid(m(batch)) for m in model]
                batch_pred = sum(batch_preds) / len(batch_preds)
                batch_pred = batch_pred.detach().cpu()
            else:
                logits = model(batch)
                batch_pred = torch.sigmoid(logits).detach().cpu()

            preds.extend(batch_pred.numpy())
            targets.extend(labels.numpy())

    preds = np.array(preds, dtype=np.float32)
    targets = np.array(targets, dtype=np.uint8)
    thresholds = np.linspace(0, 1, num=num+1)[:-1]
    metrics = []
    print("Calculating metrics...")
    for threshold in tqdm(thresholds):
        inds = (preds >= threshold).astype(np.uint8)
        correct = (inds == targets)
        correct_pos = correct * targets
        accuracy = correct.sum() / np.prod(correct.shape)
        macro_prec = precision_score(targets, inds, average='macro')
        micro_prec = precision_score(targets, inds, average='micro')
        macro_f1 = f1_score(targets, inds, average='macro')
        micro_f1 = f1_score(targets, inds, average='micro')
        metrics.append({
            "threshold": threshold,
            "positive_preds": int(inds.sum()),
            "positive_targets": int(targets.sum()),
            "positive_preds_correct": int(correct_pos.sum()),
            "accuracy": accuracy,
            "macro_precision": macro_prec,
            "micro_precision": micro_prec,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        })

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', help='model type from model.py', choices=['vis_baseline', 'vis_virtex', 'vis_swin_tiny', 'vis_swin_small', 'vis_swin_base', 'ensemble'],
                        default='vis_baseline')
    parser.add_argument('--name', help='name of the model, saved at models/',
                        default='model')
    parser.add_argument('--load_tag', help=('load a particular tag of a saved model, can be "best", "latest", or an int;'
                        'used after continue training enabled'),
                        default='latest')
    parser.add_argument('--bs', help='batch size to feed to model', type=int,
                        default=50)

    args = parser.parse_args()
    val_dataloader, model, device, metadata, cwd = initialize(args)

    metrics = validate_threshold(val_dataloader, model, device)
    max_accuracy = max(metrics, key=lambda m: m['accuracy'])
    max_macro_prec = max(metrics, key=lambda m: m['macro_precision'])
    max_micro_prec = max(metrics, key=lambda m: m['micro_precision'])
    max_macro_f1 = max(metrics, key=lambda m: m['macro_f1'])
    max_micro_f1 = max(metrics, key=lambda m: m['micro_f1'])

    print(metrics)
    print("Accuracy:", max_accuracy)
    print("Macro Precision:", max_macro_prec)
    print("Micro Precision:", max_micro_prec)
    print("Macro F1:", max_macro_f1)
    print("Micro F1:", max_micro_f1)

    os.makedirs("thresholds", exist_ok=True)
    output_path = os.path.join(cwd, "thresholds", args.name + "_" + args.load_tag + ".txt")
    with open(output_path, "w") as f:
        f.write("Max Accuracy:\n")
        f.write(json.dumps(max_accuracy, indent=4) + "\n")
        f.write("Max Macro Precision:\n")
        f.write(json.dumps(max_macro_prec, indent=4) + "\n")
        f.write("Max Micro Precision:\n")
        f.write(json.dumps(max_micro_prec, indent=4) + "\n")
        f.write("Max Macro F1:\n")
        f.write(json.dumps(max_macro_f1, indent=4) + "\n")
        f.write("Max Micro F1:\n")
        f.write(json.dumps(max_micro_f1, indent=4) + "\n")
        f.write("All Metrics:\n")
        f.write(json.dumps(metrics, indent=4))
