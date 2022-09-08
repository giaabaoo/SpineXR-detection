import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from metrics import HistoryLogger
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ClsLoader
from model import *
from IPython.display import clear_output
from pytorch_lightning.loggers import WandbLogger
import pdb
from utils import *


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_csv = pd.read_csv(
        "../vindr-spinexr-dataset/train_classify.csv", index_col=0)
    test_csv = pd.read_csv(
        "../vindr-spinexr-dataset/test_classify.csv", index_col=0)
    train_path = "../vindr-spinexr-dataset/train_images_png_224/"
    test_path = "../vindr-spinexr-dataset/test_images_png_224/"

    # wandb.login()
    train_dataset = DataLoader(ClsLoader(train_path, train_csv), batch_size=32, pin_memory=True,
                               shuffle=True, num_workers=4,
                               drop_last=True, prefetch_factor=16)
    test_dataset = DataLoader(ClsLoader(test_path, test_csv, typeData="test"), batch_size=64,
                              num_workers=4, prefetch_factor=32)
    model = ClassifyNet_vs2()
    
    # model = ClassifyNet()
    if config["SUP_LOSS"]:
        classifier = ClassifierSupcon(model=model, class_weight=config['CLASS_WEIGHT'],
                                    num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
        check_point = pl.callbacks.model_checkpoint.ModelCheckpoint("./weights/", filename="ckpt{test_loss:0.4f}",
                                                                monitor="test_loss", mode="min", save_top_k=1,
                                                                verbose=True, save_weights_only=True,
                                                                auto_insert_metric_name=False, save_last=True
                                                                # every_n_epochs=100
                                                                )
    else:
        classifier = Classifier(model=model, class_weight=config['CLASS_WEIGHT'],
                            num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
        check_point = pl.callbacks.model_checkpoint.ModelCheckpoint("./weights/", filename="ckpt{test_f1:0.4f}",
                                                            monitor="test_f1", mode="max", save_top_k=1,
                                                            verbose=True, save_weights_only=True,
                                                            auto_insert_metric_name=False, save_last=False
                                                            # every_n_epochs=100
                                                            )
    chk_path = "./weights/ckpt6.7623.ckpt"
    classifier = classifier.load_from_checkpoint(checkpoint_path=chk_path, model=model, class_weight=config['CLASS_WEIGHT'], num_classes=config["NUM_CLASS"], 
    learning_rate=config["LEARNING_RATE"], strict=False)

    # for parameter in list(classifier.parameters())[:-4]:
    #     parameter.requires_grad = False

    
    history_logger = HistoryLogger()
    wandb_logger = WandbLogger(project="spinexr")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    swa = pl.callbacks.StochasticWeightAveraging(
        swa_epoch_start=5, swa_lrs=config["LEARNING_RATE"])
    PARAMS = {"accelerator": 'gpu', "devices": 1, "benchmark": True, "enable_progress_bar": True,
              #   "callbacks" : [progress_bar],
              #    "overfit_batches" :1,
              "logger": wandb_logger,
              "callbacks": [check_point, lr_monitor],
              "log_every_n_steps": 1, "num_sanity_val_steps": 2, "max_epochs": 300,
              #   "precision":16,
              }

    trainer = pl.Trainer(**PARAMS)


    trainer.fit(classifier, train_dataset, test_dataset)
