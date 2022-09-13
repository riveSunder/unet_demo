import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelPruning

import skimage

def intersection_over_union(prediction, target):
    """
    compute intersection over union
    """

    intersection = prediction.round() * target

    union = prediction.round() + target

    iou = intersection.sum() / union.sum()

    return iou.sum()

class UNet(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.input_channels = 1
        self.output_channels = 1
        # hidden channel size (base)
        self.h = 4
        # kernel size
        self.k = 3
        # upscale conv kernel size (conv transpose)
        self.up_k = 2
        # padding
        self.p = 1

        self.lr = 3e-4
        self.l2_penalty = 1e-5

        if "use_skips" in kwargs.keys():
            self.use_skips = kwargs["use_skips"]
        else:
            self.use_skips = 1
        
        self.conv_0 = nn.Conv2d(self.input_channels,\
                self.h, self.k, stride=1, padding=self.p)
        self.conv_1 = nn.Conv2d(self.h, self.h,\
                self.k, stride=1, padding=self.p)

        self.conv_2 = nn.Conv2d(self.h, self.h*2,\
                self.k, stride=1, padding=self.p)
        self.conv_3 = nn.Conv2d(self.h*2, self.h*2,\
                self.k, stride=1, padding=self.p)

        self.conv_4 = nn.Conv2d(self.h*2, self.h*4,\
                self.k, stride=1, padding=self.p)
        self.conv_5 = nn.Conv2d(self.h*4, self.h*4,\
                self.k, stride=1, padding=self.p)

        self.conv_6 = nn.Conv2d(self.h*4, self.h*8,\
                self.k, stride=1, padding=self.p)
        self.conv_7 = nn.Conv2d(self.h*8, self.h*8,\
                self.k, stride=1, padding=self.p)

        self.conv_8 = nn.Conv2d(self.h*8, self.h*16,\
                self.k, stride=1, padding=self.p)
        self.conv_9 = nn.Conv2d(self.h*16, self.h*16,\
                self.k, stride=1, padding=self.p)

        self.conv_up_9_10 = nn.ConvTranspose2d(\
                self.h*16, self.h*8, self.up_k, \
                self.up_k)

        self.conv_10 = nn.Conv2d(self.h*8, self.h*8,\
                self.k, stride=1, padding=self.p)
        self.conv_11 = nn.Conv2d(self.h*8, self.h*8,\
                self.k, stride=1, padding=self.p)

        self.conv_up_11_12 = nn.ConvTranspose2d(\
                self.h*8, self.h*4, self.up_k, \
                self.up_k)

        self.conv_12 = nn.Conv2d(self.h*4, self.h*4,\
                self.k, stride=1, padding=self.p)
        self.conv_13 = nn.Conv2d(self.h*4, self.h*4,\
                self.k, stride=1, padding=self.p)

        self.conv_up_13_14 = nn.ConvTranspose2d(\
                self.h*4, self.h*2, self.up_k, \
                self.up_k)

        self.conv_14 = nn.Conv2d(self.h*2, self.h*2,\
                self.k, stride=1, padding=self.p)
        self.conv_15 = nn.Conv2d(self.h*2, self.h*2,\
                self.k, stride=1, padding=self.p)

        self.conv_up_15_16 = nn.ConvTranspose2d(\
                self.h*2, self.h, self.up_k, \
                self.up_k)

        self.conv_16 = nn.Conv2d(self.h, self.h,\
                self.k, stride=1, padding=self.p)
        self.conv_17 = nn.Conv2d(self.h, self.h,\
                self.k, stride=1, padding=self.p)

        self.conv_18 = nn.Conv2d(self.h, self.output_channels,\
                kernel_size=1, stride=1)

    def forward(self, x):

        x = torch.relu(self.conv_0(x))
        x = torch.relu(self.conv_1(x))
        x1 = 1.0 * x

        x = F.max_pool2d(x, 2)

        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x3 = 1.0 * x

        x = F.max_pool2d(x, 2)

        x = torch.relu(self.conv_4(x))
        x = torch.relu(self.conv_5(x))
        x5 = 1.0 * x

        x = F.max_pool2d(x, 2)

        x = torch.relu(self.conv_6(x))
        x = torch.relu(self.conv_7(x))
        x7 = 1.0 * x

        x = F.max_pool2d(x, 2)

        x = torch.relu(self.conv_8(x))
        x = torch.relu(self.conv_9(x))

        x = self.conv_up_9_10(x)
        if self.use_skips:
            x += x7

        x = torch.relu(self.conv_10(x))
        x = torch.relu(self.conv_11(x))

        x = self.conv_up_11_12(x)
        if self.use_skips:
            x += x5

        x = torch.relu(self.conv_12(x))
        x = torch.relu(self.conv_13(x))

        x = self.conv_up_13_14(x)
        if self.use_skips:
            x += x3

        x = torch.relu(self.conv_14(x))
        x = torch.relu(self.conv_15(x))

        x = self.conv_up_15_16(x)
        if self.use_skips:
            x += x1

        x = torch.relu(self.conv_16(x))
        x = torch.relu(self.conv_17(x))
        x = torch.sigmoid(self.conv_18(x))

        return x

    def training_step(self, batch, batch_idx):
        
        data_x, targets = batch[0], batch[1]

        predictions = self.forward(data_x)

        loss = F.binary_cross_entropy(\
                predictions, targets)
        accuracy = torchmetrics.functional.accuracy(\
                predictions, targets.long())
        iou = intersection_over_union(\
                predictions, targets.long())

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("train_iou", iou)

        return loss

    def validation_step(self, batch, batch_idx):
        
        data_x, targets = batch[0], batch[1]

        predictions = self.forward(data_x)

        validation_loss = F.binary_cross_entropy(\
                predictions, targets)
        validation_accuracy = torchmetrics.functional.accuracy(\
                predictions, targets.long())
        val_iou = intersection_over_union(\
                predictions, targets.long())


        self.log("val_loss", validation_loss)
        self.log("val_accuracy", validation_accuracy)
        self.log("val_iou", val_iou)
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), \
                lr=self.lr, \
                weight_decay=self.l2_penalty)

        return optimizer

if __name__ == "__main__":

    max_epochs = 20
    num_workers = 2
    batch_size = 16
    dropout_rate = 0.5
    l2 = 1e-6
    lr=3e-4
    dim_h = 1024
    my_seeds = [1, 13, 42] 

    data_x = torch.rand(1024, 1, 256, 256)
    target = np.random.randint(0,2,(1024,1,128,128))

    target = skimage.transform.resize(target, (1024,1,256,256))
    target[target > 0] = 1.0

    target = torch.tensor(target, dtype=torch.float32)


    my_seed = 13

    for use_skips in [0,1]:
        model = UNet(use_skips=use_skips)

        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        test_x, test_y = data_x[-100:], target[-100:]
        dataset = TensorDataset(data_x[:400], target[:400]) 
        val_dataset = TensorDataset(\
                data_x[-400:-100], target[-400:-100]) 
        train_dataloader = DataLoader(dataset, \
                batch_size=batch_size, \
                num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, \
                batch_size=batch_size, \
                num_workers=num_workers)


        if torch.cuda.is_available():
            trainer = pl.Trainer(accelerator="gpu", \
                    devices=1, max_epochs=max_epochs)
        else:
            trainer = pl.Trainer(max_epochs=max_epochs)

        if use_skips:
            print("training with skip connections enabled")
        else:
            print("training without skip connections")

        trainer.fit(model=model, \
                train_dataloaders=train_dataloader,\
                val_dataloaders=val_dataloader)

