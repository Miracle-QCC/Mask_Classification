import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Resize


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


def KeepResize(image_tensor, out_size):
    image_size = image_tensor.shape

    im_ratio = float(image_size[1] / image_size[2])  # H / W
    model_ratio = float(out_size[0] / out_size[1])
    if im_ratio > model_ratio:
        new_height = out_size[0]
        new_width = int(new_height / im_ratio)
    else:
        new_width = out_size[1]
        new_height = int(new_width * im_ratio)
    resize = Resize((new_height ,new_width))

    resized_img = resize(image_tensor)
    big_tensor = torch.zeros((3,out_size[0],out_size[1]),dtype=torch.float32)
    big_tensor[:,:new_height,:new_width] = resized_img
    return big_tensor

class MaskFaceDataset(Dataset):
    def __init__(self, ann_path, train=False):
        super(MaskFaceDataset, self).__init__()
        self.images = []
        self.labels = []
        self.load_annotations(ann_path)
        self.train = train
        self.torch_resize = Resize([256, 256],)  # 定义Resize类对象

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = image.astype(np.float32)
        image = torch.from_numpy(image).div(255.0)
        label = self.labels[index]
        label = int(label)
        return image, label


    def load_annotations(self, ann_file):

        for line in open(ann_file, 'r'):

            img_path, label = line.split("#")
            label = float(label)
            self.images.append(img_path)
            self.labels.append(label)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        if self.train:
            image = train_transform(image)
        else:
            image = val_transform(image)
        image = KeepResize(image, (112,112))
        label = self.labels[index]
        label = int(label)
        return image, label

if __name__ == '__main__':
    data = MaskFaceDataset('train.txt')
    train_dataloder = DataLoader(data, batch_size=2,
                                 num_workers=0, drop_last=True)
    for img,lab in train_dataloder:
        # print(img)
        pass