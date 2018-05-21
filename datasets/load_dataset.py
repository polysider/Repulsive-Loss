import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from datasets.fashion import FASHION
from datasets.online_prod import SOP
from utils.sampler import SubsetSequentialSampler

home_dir = os.path.expanduser('~')
root_dir = os.path.join(home_dir, 'sharedLocal/Datasets')


def load_dataset(dataset, train_batch_size, test_batch_size, sampler='Random'):
    """
        Loads the dataset specified
    """

    # MNIST dataset
    if dataset == 'MNIST':

        num_classes = 10
        mnist_dir = os.path.join(root_dir, 'MNIST')

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print("Downloading MNIST data...")
        trainset = MNIST(root=mnist_dir, train=True, download=True, transform=transform)
        testset = MNIST(root=mnist_dir, train=False, transform=transform)

    # CIFAR-10 dataset
    if dataset == 'CIFAR10':

        num_classes = 10
        cifar_dir = os.path.join(root_dir, 'CIFAR-10')
        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = CIFAR10(root=cifar_dir, train=True, transform=transform_train, download=True)
        testset = CIFAR10(root=cifar_dir, train=False, transform=transform_test, download=True)

    # fashion-MNIST dataset
    if dataset == 'FASHION':

        num_classes = 10
        fashion_dir = os.path.join(root_dir, 'Fashion_mnist')

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        trainset = FASHION(root=fashion_dir,
                           train=True,
                           transform=transform,
                           download=True
                           )

        testset = FASHION(root=fashion_dir,
                          train=False,
                          transform=transform,
                          download=True)

    # online products dataset
    if dataset == 'ONLINE_PRODUCTS':

        num_classes = 22634
        online_prods_dir = os.path.join(root_dir, 'Stanford_Online_Products')
        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = SOP(root=online_prods_dir, train=True, transform=transform_train)
        testset = SOP(root=online_prods_dir, train=False, transform=transform_test)


    # Deep Metric Learning
    if sampler == 'MAGNET':

        # n_train = len(trainset)
        train_sampler = SubsetSequentialSampler(range(len(trainset)), range(train_batch_size))
        trainloader = DataLoader(trainset,
                                 batch_size=train_batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 sampler=train_sampler)

        testloader = DataLoader(testset,
                                batch_size=test_batch_size,
                                shuffle=True,
                                num_workers=1)
    # Random sampling
    else:
        # n_train = len(trainset)
        trainloader = DataLoader(trainset,
                                 batch_size=train_batch_size,
                                 shuffle=False,
                                 num_workers=2)

        testloader = DataLoader(testset,
                                batch_size=test_batch_size,
                                shuffle=False,
                                num_workers=1)

    return trainloader, testloader, trainset, testset, num_classes