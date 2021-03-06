import os
import torch
import torchvision
import sys
import numpy as np

from tqdm import tqdm

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, type='train'):
        self.transform = transform
        self.type = type
        # Parse data/tag files

        cwd = os.getcwd()

        datafilename = os.path.join(cwd,'data_list_shuf.txt')
        datafile = open(datafilename)

        tagfilename = os.path.join(cwd,'tag_list_shuf.txt')
        tagfile = open(tagfilename)

        # Create vocab dictionary
        self.vocab = {}
        vocabfile = open(os.path.join(cwd,'vocab_index.txt'))
        vocablines = vocabfile.readlines()
        for line in vocablines:
            token, idx = line.split()
            self.vocab[token] = {'id': int(idx), 'count': 0}
        vocabfile.close()

        # setup data
        self.img_dir = img_dir
        self.data = []

        datalines = datafile.readlines()
        taglines = tagfile.readlines()

        # slice data in train/val set based on type
        datalen = len(datalines)
        dataslice = datalen - 6000  # datalen * 3 // 4
        if self.type == 'train':
            datalines = datalines[:dataslice]
            taglines = taglines[:dataslice]
        elif self.type == 'val':
            datalines = datalines[dataslice:]
            taglines = taglines[dataslice:]

        for id, (line, tags) in tqdm(enumerate(zip(datalines, taglines)), total=len(datalines)):
            # create dictionary for filepath and id
            data = {}

            filename = os.path.join(self.img_dir, line.rstrip())
            #print(filename)
            data['filename'] = filename
            if os.path.exists(filename):
                if self.type == 'train':
                    data['id'] = id
                elif self.type == 'val':
                    data['id'] = id + dataslice

                # create label
                taglist = tags.strip().split()
                vocablist = []
                for t in taglist:
                    vocablist.append(self.vocab[t]['id'])
                    self.vocab[t]['count'] += 1
                label = torch.zeros(997).type(torch.LongTensor)
                label.scatter_(-1, torch.LongTensor(vocablist), 1)
                data['label'] = label

                self.data.append(data)
            else:
                print(f"WARNING: Skipping {filename} because it doesn't exist.", file=sys.stderr)

        datafile.close()
        tagfile.close()
        #print(self.data)

    def __getitem__(self, index):
        filename = self.data[index]['filename']
        img = Image.open(filename)
        img = img.convert('RGB')
        if self.transform:
            # Can add further transformations in the get_transform() method below
            img = self.transform(img)
        label = self.data[index]['label']
        id = self.data[index]['id']
        return img, label, id

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


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cwd = os.getcwd()
    img_dir = os.path.join(cwd, "HARRISON")
    train = Dataset(img_dir, type="train")
    val = Dataset(img_dir, type="val")

    x = np.arange(997)
    vocab = ["" for _ in range(997)]
    counts = [0 for _ in range(997)]
    for key in train.vocab:
        vocab[train.vocab[key]['id']] = key
        counts[train.vocab[key]['id']] = train.vocab[key]['count']
    print(f"strawberry: {train.vocab['strawberry']['count']}")
    print(f"icecream: {train.vocab['icecream']['count']}")

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(x[:30], counts[:30], tick_label=vocab[:30])
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    fig.savefig("figs/train_distribution_common.jpg")
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(x[-30:], counts[-30:], tick_label=vocab[-30:])
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    fig.savefig("figs/train_distribution_uncommon.jpg")
    plt.clf()

    for key in val.vocab:
        counts[val.vocab[key]['id']] = val.vocab[key]['count']

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(x[:30], counts[:30], tick_label=vocab[:30])
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    fig.savefig("figs/val_distribution_common.jpg")
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(x[-30:], counts[-30:], tick_label=vocab[-30:])
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    fig.savefig("figs/val_distribution_uncommon.jpg")
    plt.clf()
