import argparse
import os
import torch

from torchsummaryX import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import Dataset, get_transform
from loc_loss import *
from model import *
from utils import save_model


def initialize(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Data paths
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, 'data', '2020intent', 'images', 'low')
    train_annotation_path = os.path.join(cwd, 'data', '2020intent', 'annotations', 'intentonomy_train2020.json')
    val_annotation_path = os.path.join(cwd, 'data', '2020intent', 'annotations', 'intentonomy_val2020.json')

    # Training and validation set
    train_transform_type = 'train'
    if args.train_no_aug:
        train_transform_type = 'train_no_aug'
    train_dataset = Dataset(img_dir, train_annotation_path, get_transform(type=train_transform_type), type='train')
    #train_dataset = torch.utils.data.Subset(train_dataset, range(64))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=6, drop_last=True)
    val_dataset = Dataset(img_dir, val_annotation_path, get_transform(type='val'), type='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=6, drop_last=False)

    # Setup directories
    models_dir = os.path.join(cwd, 'models', args.name)
    logs_dir = os.path.join(cwd, 'logs', args.name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Model loading
    checkpoint = None
    start_epoch = 0
    metadata = {}
    if args.continue_training:
        model_tag = str(args.load_tag) if isinstance(args.load_tag, int) else args.load_tag
        model_path = os.path.join(models_dir, model_tag + ".pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            start_epoch = checkpoint['epochs_trained']
            metadata = checkpoint['metadata']
            print(f"Training from epoch {start_epoch}...")
        else:
            raise ValueError(f"Model {model_path} does not exist.")

    # Model specification
    model = None
    if args.model_type == 'vis_baseline':
        model = ResNetVisualBaseline()
    else:
        raise NotImplementedError("Other models are not implemented.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer specification
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mtm)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Scheduler specification
    scheduler = None
    if args.linear_warmup:
        train_len = len(train_dataloader)
        warmup_steps = args.warmup_epochs*train_len
        def warmup_lambda_gen(warmup_steps):
            def warmup_lambda(epoch):
                if epoch >= warmup_steps:
                    return 1
                return epoch / warmup_steps
            return warmup_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                lr_lambda=warmup_lambda_gen(warmup_steps),
                last_epoch=start_epoch*train_len - 1)

    # Loss function specification
    ce_loss_fn = nn.CrossEntropyLoss()
    if args.use_loc_loss:
        loc_loss_fn = LocalizationLoss(cam_alpha=args.loc_loss_alpha)

    def loss_fn(model, batch, logits, target, ids):
        total_loss, loss_info = 0, {}

        # Cross Entropy
        ce_loss = ce_loss_fn(logits, target)
        loss_info['ce_loss'] = ce_loss
        total_loss += ce_loss

        # Localization
        if args.use_loc_loss:
            with torch.set_grad_enabled(True):
                loc_loss = loc_loss_fn(model, batch, ids, target)
            loss_info['loc_loss'] = loc_loss
            total_loss += loc_loss

        return total_loss, loss_info

    return train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn, \
        start_epoch, device, metadata, cwd


def train(train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn,
          start_epoch, device, metadata, cwd, args):
    # Setup logger
    logs_dir = os.path.join(cwd, 'logs', args.name)
    writer = SummaryWriter(log_dir=logs_dir)

    # Set metadata defaults
    metadata.setdefault('train', {})
    metadata.setdefault('val', {})
    metadata['train'].setdefault('loss', [])
    metadata['train'].setdefault('macro_f1', [])
    metadata['train'].setdefault('micro_f1', [])
    metadata['train'].setdefault('samples_f1', [])
    metadata['val'].setdefault('loss', [])
    metadata['val'].setdefault('macro_f1', [])
    metadata['val'].setdefault('micro_f1', [])
    metadata['val'].setdefault('samples_f1', [])

    # Loop over epochs
    model = model.to(device)
    step = start_epoch * len(train_dataloader)
    for epoch in range(start_epoch, args.epochs):
        # Training
        model.train()
        train_losses, train_preds, train_targets = [], [], []
        for local_batch, local_labels, ids in tqdm(train_dataloader):
            # Log the LR to be used
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            logits = model(local_batch)
            loss, loss_info = loss_fn(model, local_batch, logits, local_labels, ids)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Log everything
            loss = loss.detach().cpu().numpy()
            preds = torch.nn.functional.softmax(logits.detach().cpu(), dim=-1)
            preds = (preds >= 1/28).type(torch.LongTensor).tolist()
            targets = (local_labels.detach().cpu() >= 1/28).type(torch.LongTensor).tolist()
            train_losses.append(loss)
            train_preds.extend(preds)
            train_targets.extend(targets)
            writer.add_scalar('train/loss', loss, step)
            for key in loss_info:
                part_loss = loss_info[key].detach().cpu().numpy()
                writer.add_scalar(f'train/{key}', part_loss, step)

            step += 1

        # Train Metadata Logs
        loss = sum(train_losses) / len(train_losses)
        micro_f1 = f1_score(train_targets, train_preds, average='micro')
        macro_f1 = f1_score(train_targets, train_preds, average='macro')
        samples_f1 = f1_score(train_targets, train_preds, average='samples')
        metadata['train']['loss'].append(loss)
        metadata['train']['micro_f1'].append(micro_f1)
        metadata['train']['macro_f1'].append(macro_f1)
        metadata['train']['samples_f1'].append(samples_f1)
        writer.add_scalar('train/micro_f1', micro_f1, step)
        writer.add_scalar('train/macro_f1', macro_f1, step)
        writer.add_scalar('train/samples_f1', samples_f1, step)
        print(f"Epoch {epoch} Train | Loss {loss:.4f} | Micro F1 {micro_f1:.4f} | Macro F1 {macro_f1:.4f} | Samples F1 {samples_f1:.4f}")

        # Validation
        with torch.set_grad_enabled(False):
            model.eval()
            val_losses, val_count, val_preds, val_targets = [], [], [], []
            for local_batch, local_labels, ids in tqdm(val_dataloader):
                local_batch = local_batch.to(device)

                # Log everything
                logits = model(local_batch)
                loss, loss_info = loss_fn(model, local_batch, logits,
                                  (local_labels / torch.sum(local_labels, dim=-1)[:, None]).to(device),
                                  ids)
                loss = loss.detach().cpu().numpy()
                for key in loss_info:
                    loss_info[key].detach().cpu()
                preds = torch.nn.functional.softmax(logits.detach().cpu(), dim=-1)
                preds = (preds >= 1/28).type(torch.LongTensor).tolist()
                val_losses.append(loss * local_batch.shape[0])
                val_count.append(local_batch.shape[0])
                val_preds.extend(preds)
                val_targets.extend(local_labels.tolist())

        # Validation Metadata Logs
        loss = sum(val_losses) / sum(val_count)
        micro_f1 = f1_score(val_targets, val_preds, average='micro')
        macro_f1 = f1_score(val_targets, val_preds, average='macro')
        samples_f1 = f1_score(val_targets, val_preds, average='samples')
        best_loss = loss < min(metadata['val']['loss'], default=10000)
        best_micro_f1 = micro_f1 > max(metadata['val']['micro_f1'], default=0)
        best_macro_f1 = macro_f1 > max(metadata['val']['macro_f1'], default=0)
        best_samples_f1 = samples_f1 > max(metadata['val']['samples_f1'], default=0)
        metadata['val']['loss'].append(loss)
        metadata['val']['micro_f1'].append(micro_f1)
        metadata['val']['macro_f1'].append(macro_f1)
        metadata['val']['samples_f1'].append(samples_f1)
        writer.add_scalar('val/loss', loss, step)
        writer.add_scalar('val/micro_f1', micro_f1, step)
        writer.add_scalar('val/macro_f1', macro_f1, step)
        writer.add_scalar('val/samples_f1', samples_f1, step)
        print(f"Epoch {epoch} Val | Loss {loss:.4f} | Micro F1 {micro_f1:.4f} | Macro F1 {macro_f1:.4f} | Samples F1 {samples_f1:.4f}")

        # Save Model
        if (epoch + 1) % args.save_epochs == 0:
            model_path = os.path.join(cwd, 'models', args.name, str(epoch) + ".pt")
            save_model(model_path, model, optimizer, epoch, metadata)
        if best_loss:
            model_path = os.path.join(cwd, 'models', args.name, "best_loss.pt")
            save_model(model_path, model, optimizer, epoch, metadata)
        if best_micro_f1:
            model_path = os.path.join(cwd, 'models', args.name, "best_micro_f1.pt")
            save_model(model_path, model, optimizer, epoch, metadata)
        if best_macro_f1:
            model_path = os.path.join(cwd, 'models', args.name, "best_macro_f1.pt")
            save_model(model_path, model, optimizer, epoch, metadata)
        if best_samples_f1:
            model_path = os.path.join(cwd, 'models', args.name, "best_samples_f1.pt")
            save_model(model_path, model, optimizer, epoch, metadata)
        model_path = os.path.join(cwd, 'models', args.name, "latest.pt")
        save_model(model_path, model, optimizer, epoch, metadata)

    model_path = os.path.join(cwd, 'models', args.name, "completed.pt")
    save_model(model_path, model, optimizer, epoch, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Train Params
    parser.add_argument('--model_type', help='model type from model.py', choices=['vis_baseline'],
                        default='vis_baseline')
    parser.add_argument('--name', help='name of the model, saved at models/',
                        default='model')
    parser.add_argument('-c', '--continue_training', help='continue training at some epoch',
                        action='store_true')
    parser.add_argument('--load_tag', help=('load a particular tag of a saved model, can be "best", "latest", or an int;'
                        'used after continue training enabled'),
                        default='latest')
    parser.add_argument('--save_epochs', help='every how many epochs to save the model', type=int,
                        default=10)

    # Optimizer Params
    parser.add_argument('--bs', help='batch size to feed to model', type=int,
                        default=50)
    parser.add_argument('--lr', help='max learning rate', type=float,
                        default=1e-3)
    parser.add_argument('--mtm', help='sgd momentum', type=float,
                        default=0.9)
    parser.add_argument('--linear_warmup', help='linearly increase lr in the first few epochs from lr 0',
                        action='store_true')
    parser.add_argument('--warmup_epochs', help='epochs to linearly increase lr', type=int,
                        default=5)
    parser.add_argument('--epochs', help='train epochs', type=int,
                        default=50)

    # Dataset Params
    parser.add_argument('--train_no_aug', help='do not use data augmentation during training',
                        action='store_true')

    # Loss Params
    parser.add_argument('--use_loc_loss', help='use localization loss', action='store_true')
    parser.add_argument('--loc_loss_alpha', help='localization loss weight', type=float,
                        default=0.1)

    args = parser.parse_args()

    # Initialize
    train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn, \
            start_epoch, device, metadata, cwd = initialize(args)

    # Get model summary
    model.eval()
    summary(model, torch.zeros(args.bs, 3, 224, 224))
    print(model)

    # Train the model
    train(train_dataloader, val_dataloader, model, optimizer, scheduler, loss_fn,
          start_epoch, device, metadata, cwd, args)
