import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
import torch.optim as optim
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as pl
import torchmetrics

wandb_logger = WandbLogger(log_model="all",project="VOCSegmentation",name='Assignment 04')

# Check if CUDA is available and choose the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision.models as models

#dataset link : https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data
ALL_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
    'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 
    'sheep', 'sofa', 'train', 'tv/monitor'
]

LABEL_COLORS_LIST = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128]
]
jaccard = torchmetrics.JaccardIndex(task="multiclass",num_classes=len(ALL_CLASSES)).to(device)

# Normalize colors to range [0, 1]
normalized_colors = [[r/255, g/255, b/255] for r, g, b in LABEL_COLORS_LIST]

# Create colormap
cmap = ListedColormap(normalized_colors)

class VOCDataSet(Dataset):
    def __init__(self, root_dir, dataset_type='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if dataset_type == 'train':
            self.image_folder = os.path.join(root_dir, 'train_images')
            self.label_folder = os.path.join(root_dir, 'train_labels')
        elif dataset_type == 'val':
            self.image_folder = os.path.join(root_dir, 'valid_images')
            self.label_folder = os.path.join(root_dir, 'valid_labels')
        else:
            raise ValueError("Invalid dataset_type. Use 'train' or 'val'.")

        self.image_list = os.listdir(self.image_folder)
        self.label_list = os.listdir(self.label_folder)

    # Convert RGB label to an integer label
    def rgb_to_integer(self,label_rgb):
        label_integer = np.zeros(label_rgb.shape[:2], dtype=np.uint8)
        for i, color in enumerate(LABEL_COLORS_LIST):
            mask = np.all(label_rgb == color, axis=-1)
            label_integer[mask] = i
        return label_integer

    def __len__(self):
        return len(self.image_list)

    #read image and mask for a single image and apply transform 
    #be careful with applying transformations on the Mask (it should remain integer)
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_list[idx])
        label_name = os.path.join(self.label_folder, self.label_list[idx])
        image = Image.open(img_name)
        label = Image.open(label_name).convert('RGB')      
        label_array = np.array(label)
        label_integer = self.rgb_to_integer(label_array)      
        
        if self.transform:
            image = self.transform(image)         
            
        imSize = self.transform.transforms[0].size[0]
        label_integer=(torch.tensor(label_integer).unsqueeze(0)).unsqueeze(0)
        label_integer = F.interpolate(label_integer, size=(imSize,imSize), mode='nearest').squeeze(0).squeeze(0).long() #only use NN
               
        return image, label_integer


# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a train dataset object
train_dataset = VOCDataSet(root_dir='data', dataset_type='train', transform=transform)
# Create a DataLoader for the train dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a validation dataset object
val_dataset = VOCDataSet(root_dir='data', dataset_type='val', transform=transform)
# Create a DataLoader for the train dataset
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Get a batch of data & display it (just to see we correctly read the dataset!)
images, masks = next(iter(train_loader))
# Display the batch
fig, axs = plt.subplots(6, 2, figsize=(10, 30))
for i in range(6):
    # Display image
    axs[i, 0].imshow(images[i].permute(1, 2, 0))
    axs[i, 0].axis('off')    
    #use NN to display exact
    axs[i, 1].imshow(np.squeeze(masks[i]),cmap=cmap,interpolation='nearest',vmin=0, vmax=len(ALL_CLASSES)-1)
    axs[i, 1].axis('off')
    #axs[i, 1].set_title('Mask')
plt.tight_layout()
plt.show(block=True)




# Update VOCUNet class to use EfficientNet as encoder
class VOCUNet(pl.LightningModule):
    def __init__(self, n_classes=21, learning_rate=1e-3):
        super(VOCUNet, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()      
        
        # Load the pretrained EfficientNet model
        self.encoder = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        
        # Modify the last layer to output the desired number of classes
        self.encoder.classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        
        # Set the encoder to non-trainable
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Modify the classifier to be trainable
        for param in self.encoder.classifier[-1].parameters():
            param.requires_grad = True

    # Define the forward method
    def forward(self, x):
        return self.encoder(x)['out']

    # Define the training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# Fine-tune hyperparameters
learning_rate = 1e-4  # Adjust as needed
batch_size = 16  # Adjust as needed

# Create a train DataLoader with the updated batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create a validation DataLoader with the updated batch size
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




# Initialize the model with the updated hyperparameters
model = VOCUNet(learning_rate=learning_rate)
callbacks = model.configure_callbacks()
# Initialize the Trainer with the updated hyperparameters
trainer = pl.Trainer(logger=wandb_logger, max_epochs=10, devices=1, accelerator="auto", callbacks=callbacks)

# Train the model with the updated DataLoader and hyperparameters
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



