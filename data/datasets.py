from torch.utils.data import Dataset

class CifarDataset(Dataset):
    def __init__(self, dataset : Dataset, transformation = None):
        super().__init__()
        self.dataset = dataset
        self.transformation = transformation

    def __getitem__(self, index : int):
        sample = self.dataset[index]
        image = sample['img']
        if self.transformation is not None:
            image = self.transformation(image)
        label = sample['label']
        return image, label

    def __len__(self):
        return len(self.dataset)

class ImageNetDataset(Dataset):
    def __init__(self, dataset : Dataset, transformation = None):
        super().__init__()
        self.dataset = dataset
        self.transformation = transformation

    def __getitem__(self, index : int):
        sample = self.dataset[index]
        image = sample['image']
        if self.transformation is not None:
            image = self.transformation(image)
        label = sample['label']
        return image, label

    def __len__(self):
        return len(self.dataset)
