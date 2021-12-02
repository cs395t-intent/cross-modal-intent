import torch


def save_model(model_path, model, optimizer, epoch, metadata):
    checkpoint = {}
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['epochs_trained'] = epoch + 1
    checkpoint['metadata'] = metadata
    torch.save(checkpoint, model_path)
