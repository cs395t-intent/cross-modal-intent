import os
import torch
import torchvision
import json
from PIL import Image

class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, annotation_path, transform=None, type='train'):
        self.transform = transform
        self.type = type
        # Parse json file and extract the IDs for each image
        file = open(annotation_path)
        self.annotations = json.load(file)
        self.img_dir = img_dir
        self.data = []
        for annotation in self.annotations['annotations']:
            id = annotation['image_id']
            # Now add '.jpg' to each ID and join with the image directory to get each image filename
            filename = os.path.join(self.img_dir, id + '.jpg')
            if os.path.exists(filename):
                data = {'filename': filename}

                # Get the corresponding label for this image
                # The training dataset returns probabilities for each category
                # The validation dataset returns the ID of each category
                if self.type == 'train':
                    label = torch.FloatTensor(annotation['category_ids_softprob'])
                else:
                    label = torch.FloatTensor(annotation['category_ids'])
                data['label'] = label

                self.data.append(data)
            else:
                print(f"WARNING: Skipping {filename} because it doesn't exist.")
        file.close()

    def __getitem__(self, index):
        filename = self.data[index]['filename']
        img = Image.open(filename)
        if self.transform:
            # Can add further transformations in the get_transform() method below
            img = self.transform(img)
        label = self.data[index]['label']
        return img, label

    def __len__(self):
        return len(self.data)

def get_transform():
    custom_transforms = []
    # The paper said they randomly resize crop to 224 × 224
    custom_transforms.append(torchvision.transforms.RandomResizedCrop(size=(224, 224)))
    # The paper also performs a random horizontal flip
    custom_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    # Transform image to 3 layer RGB tensor
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

