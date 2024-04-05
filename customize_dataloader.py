import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import os

from torchvision.datasets import ImageFolder

class ImageFolderWithFilename(ImageFolder):
    """Custom ImageFolder dataset class that also returns the filenames and sub class (artist name)."""

    def __getitem__(self, index):
        """
        Overrides the default __getitem__ method to return image, label, filename and sub_class.
        """
        path, target = self.samples[index]
        # Extract filename from path
        filename = os.path.basename(path)
        sub_class= os.path.basename(os.path.dirname(path))
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, filename, sub_class




###############################################################################
# Data Loading
def get_data_loader(path = "./wikiart_subset", batch_size=64,num_workers=1, overfit = False):
    """ Loads images of cats and dogs, splits the data into training, validation
    and testing datasets. Returns data loaders for the three preprocessed datasets.

    Args:
        path: path to the wikiart_subset image folder that has a bunch of 
        batch size: set batchsize for the loaders
        num_workers size: set num_workers for the loaders
        overfit: for testing purposes where we try small dataset that test the 
        model capability of overfitting. 

    Returns:
        raw_dataset: iterable dataset that get the whole instance of the dataset
        train_loader: iterable training dataset organized according to batch size
        val_loader: iterable validation dataset organized according to batch size
        test_loader: iterable testing dataset organized according to batch size
        classes: A list of strings denoting the name of each class
    """

    classes = get_folder_structure(path)


    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose(
        [transforms.Resize((224,224)),transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if overfit == False:
      raw_dataset = ImageFolderWithFilename(path, transform=transform)
    else:
      #TODO: change this to suitable dataset
      raw_dataset = ImageFolderWithFilename(path, transform=transform)

    # Get the list of indices to sample from
    #relevant_indices = get_relevant_indices(raw_dataset, classes)

    # Split into train and validation
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling
    indices = np.arange(len(raw_dataset))
    np.random.shuffle(indices)
    train_split = int(len(indices) * 0.8)

    # dividing by 2 will assign 10% to val and 10% to test
    # if the train is 0.8
    testval_split = train_split + int(len(indices) * (1 - 0.8)/2)



    # split into training and validation indices
    relevant_train_indices, relevant_val_indices,test_indices = indices[:train_split], indices[train_split:testval_split] ,indices[testval_split:]

    train_sampler = SubsetRandomSampler(relevant_train_indices)
    train_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=batch_size,
                                               num_workers=num_workers, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(relevant_val_indices)
    val_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=batch_size,
                                              num_workers=num_workers, sampler=val_sampler)
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=batch_size,
                                             num_workers=num_workers, sampler=test_sampler)
    
    print('training examples: ',len(train_loader))
    print('validation examples: ',len(val_loader))
    print('testing examples: ', len(test_loader))

    return raw_dataset, train_loader, val_loader, test_loader, classes



def get_folder_structure(directory):
    """
    Get the folder structure within a directory and organize it into a dictionary.

    Args:
    - directory (str): The path to the directory.

    Returns:
    - folder_structure (dict): A dictionary representing the folder structure.
    """
    # Initialize an empty dictionary to store folder structure
    folder_structure = {}

    # Iterate over items in the directory
    for item in os.listdir(directory):
        # Check if the item is a directory (folder)
        if os.path.isdir(os.path.join(directory, item)):
            # Get the subfolders of the current folder
            subfolders = [subfolder for subfolder in os.listdir(os.path.join(directory, item))
                          if os.path.isdir(os.path.join(directory, item, subfolder))]
            # Organize subfolders into a dictionary
            folder_structure[item] = subfolders

    return folder_structure