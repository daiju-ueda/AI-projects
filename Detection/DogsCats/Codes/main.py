import numpy as np
import pandas as pd

from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class xml2list(object):

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path):

        ret = []
        xml = ET.parse(xml_path).getroot()

        for size in xml.iter("size"):
            width = float(size.find("width").text)
            height = float(size.find("height").text)

        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            bndbox = [width, height]
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                cur_pixel =  float(bbox.find(pt).text)
                bndbox.append(cur_pixel)
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            ret += [bndbox]

        return np.array(ret) # [width, height, xmin, ymin, xamx, ymax, label_idx]

xml_paths = glob("../Datasets/annotations/xmls/*.xml")
classes = ["dog", "cat"]

transform_anno = xml2list(classes)

df = pd.DataFrame(columns=["image_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "class"])

for path in xml_paths:
    image_id = path.split("/")[-1].split(".")[0]
    bboxs = transform_anno(path)

    for bbox in bboxs:
        tmp = pd.Series(bbox, index=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
        tmp["image_id"] = image_id
        df = df.append(tmp, ignore_index=True)

df = df.sort_values(by="image_id", ascending=True)

# 背景のクラス（0）が必要のため、dog, cat のラベルは1スタートにする
df["class"] = df["class"] + 1

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, df, image_dir):

        super().__init__()

        self.image_ids = df["image_id"].unique()
        self.df = df
        self.image_dir = image_dir

    def __getitem__(self, index):

        transform = transforms.Compose([
                                        transforms.ToTensor()
        ])

        # 入力画像の読み込み
        image_id = self.image_ids[index]
        image = Image.open(f"{self.image_dir}/{image_id}.jpg")
        image = transform(image)

        # アノテーションデータの読み込み
        records = self.df[self.df["image_id"] == image_id]
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.tensor(records["class"].values, dtype=torch.int64)

        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"]= labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]

image_dir = "../Datasets/images/"
dataset = MyDataset(df, image_dir)


torch.manual_seed(2020)

n_train = int(len(dataset) * 0.7)
n_val = len(dataset) - n_train

train, val = torch.utils.data.random_split(dataset, [n_train, n_val])

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=4, shuffle=False, collate_fn=collate_fn)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

num_classes = 3 # background, dog, cat
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

for epoch in range(num_epochs):

    model.train()

    for i, batch in enumerate(train_dataloader):

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
