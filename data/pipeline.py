import cv2 as cv
import numpy as np
import torch 
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, random_split, ConcatDataset, Dataset, DataLoader


# class SiftDataset(Dataset):
#   def __init__(self, features, labels):
#     assert features.shape[0] == labels.shape[0]
#     self.features = features
#     self.labels = labels
  
#   def __len__(self):
#     return len(self.labels)
  
#   def __getitem__(self, index):
#     feature = self.features[index]
#     label = self.labels[index]
#     return feature, label

class AddGaussianNoise(object):
  def __init__(self, mean=0., std=1.):
    self.std = std
    self.mean = mean
  
  def __call__(self, tensor):
    return tensor + torch.randn(tensor.size()) * self.std + self.mean
  
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SiftExtractor:
  def __init__(self, dataset, sample_size=15, vocab_size=200, debug=False):
    self.debug = debug

    self._img_dataset = dataset
    self.sample_size = sample_size
    self.vocab_size = vocab_size if not self.debug else 20
    self.vocab, self.features, self.labels = self._extract_sift_train()
  
  def extract_sift(self, dataset):
    """
    Extract SIFT features from test set using vocab built from train set
    """
    sift = cv.SIFT_create()
    idx = 0

    labels = np.empty(len(dataset))
    features = np.zeros((len(dataset), self.vocab_size))

    for i in range(len(dataset) if not self.debug else 200): # len(imgs)):
      img, label = dataset[i]
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      kp, ds = sift.detectAndCompute(img, None)
      if ds is None:
        continue
      ds = ds.astype(np.double)
      clusters, counts = np.unique(self.vocab.predict(ds), return_counts=True)

      # normalize
      features[idx, clusters] = counts / ds.shape[0]
      labels[idx] = label
      idx += 1  

    return features[:idx], labels[:idx]

  def _extract_sift_train(self):
    """
    Called during init. Create vocab from train set.
    """
    sift = cv.SIFT_create()
    descriptors = np.zeros((self.sample_size * len(self._img_dataset), 128))
    labels = np.empty(len(self._img_dataset))

    # some images might not have descriptors, exclude them
    insert = 0
    for i in range(len(self._img_dataset) if not self.debug else 200):
      img, label = self._img_dataset[i]
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      kp, ds = sift.detectAndCompute(img, None)
      if ds is None:
        continue

      descriptors[insert*self.sample_size:(insert + 1)*self.sample_size] = ds[np.random.randint(ds.shape[0], size=self.sample_size)]
      labels[insert] = label
      insert += 1

      if (i + 1) % 200 == 0:
        print(f"Processed {i + 1} Images")

    # truncate
    descriptors = descriptors[:insert*self.sample_size, :]
    labels = labels[:insert]
    
    print("")
    print("Extracting Vocabulary...")
    print("")
    
    vocab = KMeans(n_clusters=self.vocab_size).fit(descriptors)

    features = np.zeros((descriptors.shape[0] // self.sample_size, self.vocab_size))

    idx = 0
    for i in range(len(self._img_dataset) if not self.debug else 200):
      img, label = self._img_dataset[i]
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      kp, ds = sift.detectAndCompute(img, None)
      if ds is None:
        continue
      ds = ds.astype(np.double)
      clusters, counts = np.unique(vocab.predict(ds), return_counts=True)

      # normalize
      features[idx, clusters] = counts / ds.shape[0]
      idx += 1

      if (i + 1) % 200 == 0:
        print(f"Collected Features from {i + 1} Images")

    return vocab, features, labels


def generate_split(data_dir="data/dataset-resized", fracs=[0.8, 0.1, 0.1], seed=42):
    generator = torch.Generator().manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(0., 1.)
    ])

    imgs = ImageFolder(data_dir, transform)
    labels, counts = np.unique(imgs.targets, return_counts=True)
    dataset = {
        "train": [],
        "val": [], 
        "test": [], 
    }

    start = 0
    for i in range(len(labels)):
        train, val, test = random_split(Subset(imgs, np.arange(start, start + counts[i])), fracs, generator)
        dataset["train"].append(train)
        dataset["val"].append(val)
        dataset["test"].append(test)
        start += counts[i]
    
    dataset["train"] = ConcatDataset(dataset["train"])
    dataset["val"] = ConcatDataset(dataset["val"])
    dataset["test"] = ConcatDataset(dataset["test"])

    return dataset
