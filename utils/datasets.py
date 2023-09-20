#!coding:utf-8
import numpy as np
from PIL import Image
import torchvision as tv
import torchvision.transforms as transforms
from utils.cifar import Download_Cifar
from torch.utils.data import Dataset
from utils.randAug import RandAugmentMC
from utils.data_utils import NO_LABEL
from utils.data_utils import TransformWeakStrong as wstwice

load = {}
def register_dataset(dataset):
    def warpper(f):
        load[dataset] = f
        return f
    return warpper


@register_dataset('wscifar10')
def wscifar10(data_root='./data-local/cifar10/', num_classes=None, num_label=None, num_unlabel=None, num_query=None):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                         std = [0.2023, 0.1994, 0.2010])

    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        RandAugmentMC(n=2, m=10),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    train_transform = wstwice(weak, strong)

    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])



@register_dataset('cifar10')
def cifar10(data_root='./data-local/cifar10/', num_classes=None, num_label=None, num_unlabel=None, num_query=None):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                         std = [0.2023, 0.1994, 0.2010])

    weak_transform = transforms.Compose([

        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    strong_transform = transforms.Compose([

       # transforms.RandomHorizontalFlip(),
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.1),

        transforms.RandomCrop(32),
       # transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        #transforms.RandomErasing(value="1234"),
        transforms.Normalize(**channel_stats)
    ])

    double_transform = wstwice(strong_transform, strong_transform)

    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    train_data, train_targets, query_data, query_targets, label_data, label_targets, unlabel_data, unlabel_targets = Download_Cifar(
        root=data_root, download=True).prepare_data(num_classes, num_label, num_unlabel, num_query)

    #处理Dataset
    #有标记数据集处理
    train_label = Cifar_Dataset(label_data, label_targets, transform=strong_transform)
    #使用有标记无+标记整体作为无标记数据集训练
    train_unlabel = Cifar_Dataset(train_data, train_targets, transform=double_transform)
    #处理 database 和 queryset
    ### 检索集合使用无标记数据集  根据需求选用不同的数据集
    database = Cifar_Dataset(unlabel_data, unlabel_targets, transform=eval_transform)
    #database = Cifar_Dataset(train_data, train_targets, transform=eval_transform)
    queryset = Cifar_Dataset(query_data, query_targets, transform=eval_transform)

    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset      | # number")
    print("  ------------------------------")
    print("  queryset    | {:8d}".format(len(queryset.data)))
    print("  labeled     | {:8d}".format(len(train_label.data)))
    print("  unlabeled   | {:8d}".format(len(train_unlabel.data)))
    print("  database    | {:8d}".format(len(database.data)))
    print("  ------------------------------")


    return {
        'labelset': train_label,
        'unlabelset': train_unlabel,
        'queryset': queryset,
        'database': database
    }


class Cifar_Dataset(Dataset):
    def __init__(self, data, label, transform):
        super(Cifar_Dataset).__init__()
        self.transform = transform
        self.data = data
        self.targets = label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

