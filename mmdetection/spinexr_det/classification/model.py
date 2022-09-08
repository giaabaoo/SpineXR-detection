import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import numpy as np
import pytorch_lightning as pl
import timm.optim
from classification.loss_cls import FocalLoss, SupConLoss
import torch.optim as optim
from classification.metrics import accuracy, f1_score
from classification.utils_cls import *


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        elif self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class Block(nn.Module):
    def __init__(self, dim, drop_path_rate=0., layer_scale_init_value=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                padding=3, groups=dim) 
        self.norm = LayerNorm(dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path_rate=0, layer_scale_init_value=1):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim),
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.GELU(),
            nn.Conv2d(in_dim, 4 * in_dim, kernel_size=1, padding=0),
            nn.InstanceNorm2d(4 * in_dim, affine=True),
            nn.GELU(),
            nn.Conv2d(4 * in_dim, in_dim, kernel_size=1, padding=0),
            )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        res = x
        x = self.conv_layer(x)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        x = res + self.drop_path(x)
        return x


class ClassifyNet(nn.Module):
    def __init__(self, in_dim=1, num_classes=2, depths=[3, 3, 9, 3], dims_encoder=[96, 192, 384, 768], drop_path_rate=0.5):
        super().__init__()
        ######################## encoder ##################################################
        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_dim, dims_encoder[0], kernel_size=4, stride=4),
            nn.InstanceNorm2d(dims_encoder[0], affine=True)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm2d(dims_encoder[i], affine=True),
                nn.Conv2d(dims_encoder[i], dims_encoder[i+1],
                          kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item()
                    for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvBlock(dims_encoder[i], dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims_encoder[-1], eps=1e-6)
        self.head = nn.Sequential(nn.Linear(dims_encoder[-1], num_classes),
                                  nn.Softmax(dim=1)
                                  )

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x


class ClassifyNet_vs2(nn.Module):
    def __init__(self, in_dim=3, num_classes=2, depths=[3, 3, 9, 3],
    dims_encoder=[96, 192, 384, 768], drop_path_rate=0.5, sup_loss=config["SUP_LOSS"], proj_head=128):
        super().__init__()
        self.sup_loss = sup_loss
        ######################## encoder ##################################################
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_dim, dims_encoder[0], kernel_size=4, stride=4),
            LayerNorm(dims_encoder[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims_encoder[i], eps=1e-6,
                              data_format="channels_first"),
                    nn.Conv2d(
                        dims_encoder[i], dims_encoder[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item()
                           for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dims_encoder[i], dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims_encoder[-1], eps=1e-6)
        if self.sup_loss:
            self.project_head = nn.Sequential(nn.Linear(dims_encoder[-1], dims_encoder[-1]),
                                                nn.GELU(),
                                                nn.Linear(dims_encoder[-1], proj_head),  
                                                )
        else:
            self.head = nn.Sequential(nn.Linear(dims_encoder[-1], num_classes),
                                  nn.Softmax(dim=1)
                                  )

    def forward(self, x):
        for i in range(4):
            x=self.downsample_layers[i](x)
            x=self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        x=self.norm(x.mean([-2, -1]))
        
        if self.sup_loss:
            x = F.normalize(self.project_head(x), dim=1)
        else:
            x=self.head(x)
        return x

class Classifier(pl.LightningModule):
    def __init__(self, model, class_weight, num_classes, learning_rate):
        super().__init__()
        self.model=model
        self.class_weight=class_weight
        self.num_classes=num_classes
        self.learning_rate=learning_rate

    def forward(self, x):
        return self.model(x)

    def get_metrics(self):
        # don't show the version number
        items=super().get_metrics()
        items.pop("v_num", None)
        return items

    def _step(self, batch):
        image, y_true=batch
        y_pred=self.model(image)
        loss_focal=FocalLoss(
            self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        # loss_contour = ActiveContourLoss(self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        # loss = loss_focal + loss_contour
        acc, f1=accuracy(y_true, y_pred), f1_score(y_true, y_pred)
        return loss_focal, acc, f1

    def training_step(self, batch, batch_idx):
        loss, acc, f1=self._step(batch)
        metrics={"loss": loss, "train_acc": acc, "train_f1": f1}
        self.log_dict(metrics, on_step = True,
                      on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1=self._step(batch)
        metrics={"test_loss": loss, "test_acc": acc, "test_f1": f1}
        self.log_dict(metrics, prog_bar = True)
        return metrics
    def test_step(self, batch, batch_idx):
        loss, acc, f1=self._step(batch)
        metrics={"test_loss": loss, "test_acc": acc, "test_f1": f1}
        self.log_dict(metrics, prog_bar = True)
        return metrics
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat.cpu().numpy()

    def configure_optimizers(self):
        optimizer=timm.optim.Nadam(self.parameters(), lr = self.learning_rate)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max",
                                                         factor = 0.5, patience = 20, verbose = True)
        lr_schedulers={"scheduler": scheduler, "monitor": "test_f1"}
        return [optimizer], lr_schedulers

class ClassifierSupcon(pl.LightningModule):
    def __init__(self, model, class_weight, num_classes, learning_rate):
        super().__init__()
        self.model=model
        self.class_weight=class_weight
        self.num_classes=num_classes
        self.learning_rate=learning_rate

    def forward(self, x):
        return self.model(x)

    def get_metrics(self):
        # don't show the version number
        items=super().get_metrics()
        items.pop("v_num", None)
        return items

    def _step(self, batch):
        image_0, image_1, y_true = batch
        images = torch.cat([image_0, image_1], dim=0)
        bsz = y_true.size(0)
        features = self.model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = SupConLoss(self.device)(features, y_true)

        return loss

    def training_step(self, batch, batch_idx):
        loss=self._step(batch)
        metrics={"loss": loss}
        self.log_dict(metrics, on_step = True,
                      on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss=self._step(batch)
        metrics={"test_loss": loss}
        self.log_dict(metrics, prog_bar = True)
        return metrics
    def test_step(self, batch, batch_idx):
        loss=self._step(batch)
        metrics={"test_loss": loss}
        self.log_dict(metrics, prog_bar = True)
        return metrics
        
    def configure_optimizers(self):
        optimizer = timm.optim.Nadam(self.parameters(), lr = self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min",
                                                         factor = 0.5, patience = 20, verbose = True)
        lr_schedulers={"scheduler": scheduler, "monitor": "test_loss"}
        return [optimizer], lr_schedulers