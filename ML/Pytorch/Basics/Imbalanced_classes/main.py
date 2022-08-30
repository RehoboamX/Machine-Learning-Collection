import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods for dealing with imbalanced datasets:
# 1. Oversampling(更常用)
# 2. Class weighting

# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))  # Class weighting: 当处理到Swedish elkhound中的数据时，将损失x50
def get_loader(root_dir, batchsize):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)

    #class_weights = [1, 50]
    class_weights = []  # 避免去每个文件夹数文件数的通用做法
    for root, subdir, files in os.walk(root_dir):   # root返回文件夹的绝对路径，subdir和files分别返回子文件夹名和子文件名的列表
        if len(files) > 0:
            class_weights.append(1/len(files))

    sample_weights = [0] * len(dataset)  # dataset中的每一个数据都要有一个weight

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]  # 根据类别分配权重
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=
                                    len(sample_weights), replacement=True)  # 需要用到Oversampling时要将replacement设为True
    loader = DataLoader(dataset, batch_size=batchsize, sampler=sampler)  # 传入sampler
    return loader


def main():
    loader = get_loader(root_dir="dataset", batchsize=8)

    num_retrievers = 0
    num_elkhounds = 0
    for epoch in range(10):
        for data, labels in loader:
            num_retrievers += torch.sum(labels==0)
            num_elkhounds += torch.sum(labels==1)

    print(num_retrievers)
    print(num_elkhounds)


if __name__ == "__main__":
    main()
