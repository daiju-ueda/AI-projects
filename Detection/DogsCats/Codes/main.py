import numpy as np
import pandas as pd

from PIL import Image
from glob import glob
import os
import argparse

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

target_data = 'DogsCats'
# Arguments
parser = argparse.ArgumentParser(description= target_data + 'PyTorch Training')
parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--val_batch_size', type=int, default=50, metavar='N', help='input batch size for val (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 10)')
args = parser.parse_args()

img_size = 'Original'
csvs_path = glob('../Datasets/' + img_size + '/Split/*.csv')
csv_path = sorted(csvs_path, key=lambda f: os.stat(f).st_mtime, reverse=True)[0] # Fetch newest.

df = pd.read_csv(csv_path)
# Background class (0) is needed. Dog and cat labels should start from 1.
df["label"] = df["label"] + 1

class MyDataset(Dataset):
    def __init__(self, df, split):
        super().__init__()
        #self.df_original = pd.read_csv(csv_path)
        self.df = df[df['split']==split] # Split dataset according to csv.
        self.image_index = self.df.columns.get_loc('image_id')
        self.label_index = self.df.columns.get_loc('label')
        #self.path_index = self.df.columns.get_loc('image_id')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    def __getitem__(self, index):
        # Load image
        image_idx = self.df.iat[index, self.image_index]
        image = Image.open(csv_path.rsplit('/',2)[0] + '/Images/' + image_idx + '.jpg')
        # Process image
        if self.transform:
            image = self.transform(image)

        # アノテーションデータの読み込み
        records = self.df[:index]
        #records = self.df[self.df["image_id"] == image_id]
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.tensor(records["label"].values, dtype=torch.int64)

        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"]= labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target, image_idx

    def __len__(self):
        return len(self.df)

# Load data
train_data = MyDataset(df, 'train')
val_data   = MyDataset(df, 'val')

print('train_data = ', len(train_data))
print('val_data = ', len(val_data))

# レポジトリのutils.pyの中のcollate_fn関数を呼び出している。これが大事っぽい。多分画像の整形とか一気にしてくれている。
def collate_fn(batch):
    return tuple(zip(*batch))

# Set data loader
train_loader = torch.utils.data.DataLoader(
      dataset=train_data,  # set dataset
      batch_size=args.batch_size,  # set batch size
      shuffle=True,  # shuffle or not
      num_workers=0,
      collate_fn=collate_fn)  # set number of cores

valid_loader = torch.utils.data.DataLoader(
      dataset=val_data,
      batch_size=args.val_batch_size,
      shuffle=True,
      num_workers=0,
      collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

num_classes = 3 # background, dog, cat
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = args.epochs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        images, targets, image_ids = batch

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        pred = model(images, targets)
        losses = sum(loss for loss in pred.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
          print(f"epoch #{epoch+1} Iteration #{i+1} loss: {loss_value}")
