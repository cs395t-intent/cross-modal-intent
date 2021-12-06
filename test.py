import argparse
import os
import torch

from tqdm import tqdm

from dataset import Dataset, get_transform
from model import *


def initialize(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Data paths
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, 'data', '2020intent', 'images', 'low')
    val_annotation_path = os.path.join(cwd, 'data', '2020intent', 'annotations', 'intentonomy_val2020_ht_bert.json')
    test_annotation_path = os.path.join(cwd, 'data', '2020intent', 'annotations', 'intentonomy_test2020_ht_bert.json')

    # Validation and test set
    val_dataset = Dataset(img_dir, val_annotation_path, get_transform(type='val'), type='val', use_hashtags=args.use_hashtags)
    #val_dataset = torch.utils.data.Subset(val_dataset, range(64))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=6, drop_last=False)
    test_dataset = Dataset(img_dir, test_annotation_path, get_transform(type='test'), type='test', use_hashtags=args.use_hashtags)
    #test_dataset = torch.utils.data.Subset(test_dataset, range(64))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=6, drop_last=False)

    # Setup directories
    models_dir = os.path.join(cwd, 'models', args.name)

    # Model loading
    model_tag = str(args.load_tag) if isinstance(args.load_tag, int) else args.load_tag
    model_path = os.path.join(models_dir, model_tag + ".pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        epochs = checkpoint['epochs_trained']
        metadata = checkpoint['metadata']
        print(f"Model has been trained for {epochs} epochs.")
    else:
        raise ValueError(f"Model {model_path} does not exist.")

    # Model specification
    model = None
    if args.model_type == 'vis_baseline':
        model = ResNetVisualBaseline()
    elif args.model_type == 'baseline':
        model = ResNetBaseline()
    elif args.model_type == 'vis_virtex':
        model = VirtexVisual()
    elif args.model_type == 'virtex':
        model = Virtex()
    elif args.model_type == 'vis_swin_tiny':
        model = SwinTransformerVisual(model_size='tiny')
    elif args.model_type == 'vis_swin_small':
        model = SwinTransformerVisual(model_size='small')
    elif args.model_type == 'vis_swin_base':
        model = SwinTransformerVisual(model_size='base')
    elif args.model_type == 'swin_tiny':
        model = SwinTransformer(model_size='tiny')
    elif args.model_type == 'swin_small':
        model = SwinTransformer(model_size='small')
    elif args.model_type == 'swin_base':
        model = SwinTransformer(model_size='base')
    else:
        raise NotImplementedError("Other models are not implemented.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    return val_dataloader, test_dataloader, model, epochs, device, metadata, cwd


def test(val_dataloader, test_dataloader, model, epochs, device, metadata, cwd, args):
    # Loop over epochs
    model = model.to(device).eval()
    d_dict = {'epochs_trained': epochs, 'metadata': metadata, 'model_type': args.model_type, 'load_tag': args.load_tag}

    with torch.set_grad_enabled(False):
        d_dict['val_scores'] = []
        d_dict['val_targets'] = []
        d_dict['test_scores'] = []
        d_dict['test_targets'] = []

        # Setup labels
        indices = torch.arange(28)

        # Validation
        for values in tqdm(val_dataloader):
            if args.use_hashtags:
                local_batch, local_ht_embeds, local_labels, ids = values
                local_batch = local_batch.to(device)
                local_ht_embeds = local_ht_embeds.to(device)
                logits = model(local_batch, local_ht_embeds)
            else:
                local_batch, local_labels, ids = values
                local_batch = local_batch.to(device)
                logits = model(local_batch)
            preds = torch.nn.functional.softmax(logits.detach().cpu(), dim=-1)
            preds = preds.detach().cpu()
            d_dict['val_scores'].extend(preds.tolist())
            for label_encoding in local_labels:
                labels = indices[label_encoding == 1]
                d_dict['val_targets'].append(labels.tolist())

        # Test
        for values in tqdm(test_dataloader):
            if args.use_hashtags:
                local_batch, local_ht_embeds, local_labels, ids = values
                local_batch = local_batch.to(device)
                local_ht_embeds = local_ht_embeds.to(device)
                logits = model(local_batch, local_ht_embeds)
            else:
                local_batch, local_labels, ids = values
                local_batch = local_batch.to(device)
                logits = model(local_batch)
            preds = torch.nn.functional.softmax(logits.detach().cpu(), dim=-1)
            preds = preds.detach().cpu()
            d_dict['test_scores'].extend(preds.tolist())
            for label_encoding in local_labels:
                labels = indices[label_encoding == 1]
                d_dict['test_targets'].append(labels.tolist())

    # Save Scores
    scores_dir = os.path.join(cwd, 'scores')
    os.makedirs(scores_dir, exist_ok=True)
    scores_path = os.path.join(scores_dir, args.save_name + ".pt")
    torch.save(d_dict, scores_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Load Params
    parser.add_argument('--model_type', help='model type from model.py', choices=['vis_baseline', 'vis_virtex', 'vis_swin_tiny', 'vis_swin_small', 'vis_swin_base', 'baseline', 'virtex', 'swin_tiny', 'swin_small', 'swin_base'],
                        default='vis_baseline')
    parser.add_argument('--name', help='name of the model, saved at models/',
                        default='model')
    parser.add_argument('--load_tag', help='load a particular tag of a saved model, can be "best", "latest", or an int',
                        default='latest')
    parser.add_argument('--save_name', help='savefile name for the test outputs',
                        default='model')

    # Dataset Params
    parser.add_argument('--bs', help='batch size to feed to model', type=int,
                        default=50)
    parser.add_argument('--use_hashtags', help='use hashtags',
                        action='store_true')

    args = parser.parse_args()

    # Initialize
    val_dataloader, test_dataloader, model, epochs, device, metadata, cwd = initialize(args)

    # Test the model
    test(val_dataloader, test_dataloader, model, epochs, device, metadata, cwd, args)
