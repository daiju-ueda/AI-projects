
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
import glob
import os

# load csv
csv_test_path = '../../Datasets/csv/complete_binary_test.csv'
images_root_dir = '../../Datasets/Images/'
results_csv_dir = '../../Results/csv'
classes = pd.read_csv(csv_test_path)['label'].nunique()

# Arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
parser.add_argument('--model', default='resnet18', help='model selection(default: resnet18)')
parser.add_argument('--batch_size', type=int, default=1000, metavar='N', help='input batch size for training (default: 1000)')
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

# Model loading
weights_dir = '../../Weights/' + args.model + '_*'
weights_path = glob.glob(weights_dir)
# Choose recent one.
weights_sorted = sorted(weights_path, key=lambda f: os.stat(f).st_mtime, reverse=True)
image_save_point = torch.load(weights_sorted[0])
net.load_state_dict(image_save_point)

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
        img_path = os.path.join(self.root_dir, self.df.iat[idx, img_idx])
        img_name = self.df.iat[idx, img_idx]
        # Read images
        image = Image.open(img_path)
        # Process image
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

# load data
test_data = MyDataSet(csv_test_path, images_root_dir)

print('test_data = ', len(test_data))

# set data loader
test_loader = torch.utils.data.DataLoader(
      dataset=test_data,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=2)

# select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== test mode ======
print ('test is starting ...')
net.eval()
with torch.no_grad():
    total = 0
    test_acc = 0
    for images, labels, img_names in test_loader:
        images, labels = images.to(device), labels.to(device),
        outputs = net(images)
        # Predict with values between 0-1
        likelihood_ratio = nn.functional.softmax(outputs, dim=1)
        test_acc += (outputs.max(1)[1] == labels).sum().item()
        total += labels.size(0)
    print('test_accuracy: {} %'.format(100 * test_acc / total))
print ('test finished !')

labels = labels.to('cpu').detach().numpy().copy()
likelihood_ratio = likelihood_ratio.to('cpu').detach().numpy().copy()

s_img_names = pd.Series(img_names, name='id')
s_labels = pd.Series(labels, name='label')
df_likelihood_ratio = pd.DataFrame(likelihood_ratio, columns=['0', '1'])

df_img_name_label = pd.concat([s_img_names, s_labels], axis=1)
df_concat = pd.concat([df_img_name_label, df_likelihood_ratio], axis=1)
df_concat.to_csv(results_csv_dir + 'inferenced_results.csv', index=False)
