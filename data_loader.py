import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, CoarseDropout
from albumentations.pytorch.transforms import ToTensorV2


class album_Compose_train:
    def __init__(self):
        self.albumentations_transform = Compose([
          PadIfNeeded(40),
          RandomCrop(32,32),
          HorizontalFlip(p=0.5),
          CoarseDropout(max_holes = 1
                         , max_height=8,
                         max_width=8,
                         min_holes = 1,
                         min_height=8,
                         min_width=8,
                         fill_value=(0.4914, 0.4822, 0.4465),
                         mask_fill_value = None),
         Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
          ToTensorV2()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class album_Compose_test():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img


class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, batch_size=128):
        SEED=1
        # for CUDA
        cuda = torch.cuda.is_available()
        print("CUDA available: ", cuda)

        # for Apple GPU
        use_mps = torch.backends.mps.is_available()
        print("mps: ", use_mps)

        if cuda:
            torch.cuda.manual_seed(SEED)
        elif use_mps:
            torch.mps.manual_seed(SEED)
        else:
            torch.manual_seed(SEED)


        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 2, pin_memory = True) if (use_mps|cuda) else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag):

        # Transformations data augmentation (only for training)
        if train_flag :
            return datasets.CIFAR10('./data',
                            train=train_flag,
                            transform=album_Compose_train(),
                            download=True)

        # Testing transformation - normalization adder
        else:
            return datasets.CIFAR10('./data',
                                train=train_flag,
                                transform=album_Compose_test(),
                                download=True)

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))


    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True).data
        test_data = self.data(train_flag=False).data

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

    def sample_pictures(self, train_flag=True, return_flag = False):

        # get some random training images
        images,labels = next(iter(self.loader(train_flag)))

        sample_size=25 if train_flag else 5

        images = images[0:sample_size]
        labels = labels[0:sample_size]

        fig = plt.figure(figsize=(10, 10))

        # Show images
        for idx in np.arange(len(labels.numpy())):
            ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
            npimg = unnormalize(images[idx])
            ax.imshow(npimg, cmap='gray')
            ax.set_title("Label={}".format(str(self.classes[labels[idx]])))

        fig.tight_layout()  
        plt.show()

        if return_flag:
            return images,labels
def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))