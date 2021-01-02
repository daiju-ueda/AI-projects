
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import argparse
import os

# load csv
csv_train_path = '../../Datasets/csv/complete_regression_train.csv'
csv_val_path = '../../Datasets/csv/complete_regression_val.csv'
images_root_dir = '../../Datasets/Images/'
classes = pd.read_csv(csv_train_path)['label'].nunique()

# Dataset Preparation
label_idx = 1 # columns_no for index
img_idx = 0 # columns_no for image

class MyDataSet(Dataset):
    def __init__(self, csv_path, root_dir):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # Read img path and labels from dataframe
        label = self.df.iat[idx, label_idx]
        img_name = os.path.join(self.root_dir, self.df.iat[idx, img_idx])
        # Read images
        image = Image.open(img_name)
        # Process image
        if self.transform:
            image = self.transform(image)
        return image, label

# Arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='resnet18', help='model selection(default: resnet18)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--val_batch_size', type=int, default=1000, metavar='N', help='input batch size for val (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
args = parser.parse_args()

# Define model
if args.model == 'ResNet':
    print('ResNet50 is selected for the model')
    from torchvision.models import resnet50
    net = resnet50(num_classes=classes)
elif args.model == 'Inception':
    print('Inceptionv3 is selected for the model')
    from torchvision.models import inception_v3
    net = inception_v3(num_classes=classes)
elif args.model == 'DenseNet':
    print('DenseNet121 is selected for the model')
    from torchvision.models import densenet121
    net = densenet121(num_classes=classes)
elif args.model == 'SqueezeNet':
    print('SqueezeNet is selected for the model')
    from torchvision.models import squeezenet1_0
    net = squeezenet1_0(num_classes=classes)
elif args.model == 'VGG':
    print('VGG is selected for the model')
    from torchvision.models import VGG16
    net = VGG16(num_classes=classes)
else:
    print('Use default ResNet18.')
    from torchvision.models import resnet18
    net = resnet18(num_classes=classes)

# load data
train_data = MyDataSet(csv_train_path, images_root_dir)
val_data = MyDataSet(csv_val_path, images_root_dir)

print('train_data = ', len(train_data))
print('val_data = ', len(val_data))

# set data loader
train_loader = torch.utils.data.DataLoader(
      dataset=train_data,  # set dataset
      batch_size=args.batch_size,  # set batch size
      shuffle=True,  # shuffle or not
      num_workers=2)  # set number of cores

valid_loader = torch.utils.data.DataLoader(
      dataset=val_data,
      batch_size=args.val_batch_size,
      shuffle=False,
      num_workers=2)

# select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## This is for grayscale images. If 3 channels, comment out this code.
# optimizing
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

###  training
print ('training is starting ...')
num_epochs = args.epochs

# initialize list for plot graph after training
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

for epoch in range(num_epochs):
    # initialize each epoch
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

    # ====== train mode ======
    net.train()
    for i, (images, labels) in enumerate(train_loader):  # ミニバッチ回数実行
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 勾配リセット
        outputs = net(images)  # 順伝播の計算
        loss = criterion(outputs, labels)  # lossの計算
        train_loss += loss.item()  # train_loss に結果を蓄積
        acc = (outputs.max(1)[1] == labels).sum()  #  予測とラベルが合っている数の合計
        train_acc += acc.item()  # train_acc に結果を蓄積
        loss.backward()  # 逆伝播の計算
        optimizer.step()  # 重みの更新
        avg_train_loss = train_loss / len(train_loader.dataset)  # lossの平均を計算
        avg_train_acc = train_acc / len(train_loader.dataset)  # accの平均を計算

    # ====== valid mode ======
    net.eval()
    with torch.no_grad():  # 必要のない計算を停止
      for images, labels in valid_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = net(images)
          loss = criterion(outputs, labels)
          val_loss += loss.item()
          acc = (outputs.max(1)[1] == labels).sum()
          val_acc += acc.item()
    avg_val_loss = val_loss / len(valid_loader.dataset)
    avg_val_acc = val_acc / len(valid_loader.dataset)

    # print log
    print ('epoch [{}/{}], train_loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
                   .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

    # append list for polt graph after training
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

print ('training finished !')

# save weights
if args.save_model:
    import datetime
    dt_now = datetime.datetime.now()
    dt_name = dt_now.strftime('%Y-%m-%d-%H-%M')
    weights_path = '../../Weights/' + args.model + '_' + 'cifar-categorical' + '_' + dt_name + '.ckpt'
    torch.save(net.state_dict(), weights_path)
