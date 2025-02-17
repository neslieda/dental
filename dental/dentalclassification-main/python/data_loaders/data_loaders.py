from torch.utils.data import DataLoader
# data_loaders.py
from dataloader.datatest import image_datasets

# İlgili kodlar burada olacak

def create_dataloaders(batch_size):
  dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
    "validation": DataLoader(image_datasets["validation"], batch_size=batch_size, shuffle=False),
    "test": DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False)
  }

  return dataloaders
