import pytorch_lightning as pl
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ClsLoader
from model import *
from utils import *

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_csv = pd.read_csv(
        "../vindr-spinexr-dataset/test_classify.csv", index_col=0)
    test_path = "../vindr-spinexr-dataset/test_images_png_224/"

    config = load_config("config.yaml")

    test_dataset = DataLoader(ClsLoader(test_path, test_csv, typeData="test"), batch_size=128,
                              num_workers=4, prefetch_factor=64)
    model = ClassifyNet_vs2()
    classifier = Classifier(model=model, class_weight=config['CLASS_WEIGHT'],
                            num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
    PARAMS = {"accelerator": 'gpu', "devices": 1, "benchmark": True, "enable_progress_bar": True,
              #   "callbacks" : [progress_bar],
              #    "overfit_batches" :1,
              "logger": False,
              #   "callbacks": [check_point, logger, lr_monitor, swa],
              "log_every_n_steps": 1, "num_sanity_val_steps": 2, "max_epochs": 15,
              #   "precision":16,
              }

    trainer = pl.Trainer(**PARAMS)
    chk_path = "./weights/ckpt0.8045.ckpt"
    classifier = classifier.load_from_checkpoint(checkpoint_path=chk_path, model=ClassifyNet_vs2(), class_weight=config['CLASS_WEIGHT'],
                                                 num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])

    # trainer.test(classifier, test_dataset)
    # predictions = trainer.predict(classifier, dataloaders=test_dataset)
    # print(predictions.shape)

