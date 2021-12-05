import os
import torch
import torchvision
import json
import sys

from PIL import Image
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_path, transform=None, type='train', use_hashtags=False):
        self.transform = transform
        self.type = type
        self.use_hashtags = use_hashtags
        # Parse json file and extract the IDs for each image
        file = open(annotation_path)
        self.annotations = json.load(file)
        self.img_dir = img_dir
        self.data = []
        for annotation in tqdm(self.annotations['annotations']):
            id = annotation['image_id']
            # Now add '.jpg' to each ID and join with the image directory to get each image filename
            filename = os.path.join(self.img_dir, id + '.jpg')
            if os.path.exists(filename):
                data = {'filename': filename, 'id': id}

                # Get the corresponding label for this image
                # The training dataset returns probabilities for each category
                # The validation dataset returns the ID of each category
                if self.type == 'train':
                    label = torch.FloatTensor(annotation['category_ids_softprob'])
                else:
                    label = torch.zeros(28).type(torch.LongTensor)
                    label.scatter_(-1, torch.LongTensor(annotation['category_ids']), 1)
                data['label'] = label

                if use_hashtags:
                    data['embed'] = torch.tensor(annotation['embed'])

                self.data.append(data)
            else:
                print(f"WARNING: Skipping {filename} because it doesn't exist.", file=sys.stderr)
        file.close()

    def __getitem__(self, index):
        filename = self.data[index]['filename']
        img = Image.open(filename)
        if self.transform:
            # Can add further transformations in the get_transform() method below
            img = self.transform(img)
        label = self.data[index]['label']
        id = self.data[index]['id']
        if self.use_hashtags:
            embed = self.data[index]['embed']
            return img, embed, label, id
        return img, None, label, id

    def __len__(self):
        return len(self.data)


def get_transform(type):
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=224),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train_no_aug': torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[type]
