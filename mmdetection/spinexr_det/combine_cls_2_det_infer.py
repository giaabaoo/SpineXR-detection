import pytorch_lightning as pl
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from classification.dataset import ClsLoader
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from classification.model import ClassifyNet_vs2, Classifier
from classification.utils_cls import load_config

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from inference_config import cfg
import numpy as np
import pdb


config = load_config("classification/config.yaml")

path_dataset = "/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset"

def find_threshold_4_det(test_csv, test_path=f"{path_dataset}/test_images_png_224/", config=config,
                         chk_path="classification/weights/ckpt0.8045.ckpt"):
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
    classifier = classifier.load_from_checkpoint(checkpoint_path=chk_path, model=ClassifyNet_vs2(), class_weight=config['CLASS_WEIGHT'],
                                                 num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
    predictions = np.vstack(trainer.predict(
        classifier, dataloaders=test_dataset))[:, 1]  # get probs for abnormal

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    J = tpr - fpr
    thresh_i = np.argmax(J)
    threshold = thresholds[thresh_i]
    return threshold

if __name__ == "__main__":
    test_csv = pd.read_csv(
        f"{path_dataset}/test_classify.csv", index_col=0)
    test_path = f"{path_dataset}/test_images_png_224/"

    config = load_config("classification/config.yaml")

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
    chk_path = "classification/weights/ckpt0.8045.ckpt"
    classifier = classifier.load_from_checkpoint(checkpoint_path=chk_path, model=ClassifyNet_vs2(), class_weight=config['CLASS_WEIGHT'],
                                                 num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
    classifier.eval()
    labels = np.array(test_csv["abnormal"])
    ids = test_csv["image_id"]

    ##################### evaluate #############################################
    # pred = np.vstack(trainer.predict(classifier, test_dataset))
    # y_pred = np.argmax(pred, axis=1)
    # # print(pred.shape)
    # print("f1:", f1_score(labels, y_pred))
    # tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
    # specificity = tn / (tn+fp)
    # sensitivity = tp / (tp+fn)
    # print("specificity:", specificity)
    # print("sensitivity:", sensitivity)
    # print("roc_auc_score:", roc_auc_score(labels, pred[:,1]))
    ################################################################################

    ### NORMAL IMAGES
    # path = f"{path_dataset}/test_pngs/dd451c106a4992de94d47c663903817b.png"
    path = f"{path_dataset}/test_pngs/0d58791cc0815fcf95cfdcf1606d87bf.png"

    ### ABNORMAL IMAGES
    # path = f"{path_dataset}/test_pngs/3ac504655919e5c97111b8d644209ff7.png"
    # path = f"{path_dataset}/test_pngs/3ac504655919e5c97111b8d644209ff7.png"
    
    image_224 = np.array(Image.open(path).resize((224, 224)))
    data = np.repeat(image_224[..., np.newaxis], 3, axis=-1)
    image_224 = transforms.ToTensor()(data)
    image_224 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image_224)
    image_224 = torch.unsqueeze(image_224, 0)
    probs_abnormal = classifier(image_224)[0, 1].detach().numpy()
    print(probs_abnormal)
    # print(find_threshold_4_det(test_csv))
    
    # Setup a checkpoint file to load
    checkpoint = './spinexr_exps/latest.pth'

    # Set the device to be used for evaluation
    device='cuda:0'

    # Initialize the detector
    model = build_detector(cfg.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = cfg

    # Convert the model into evaluation mode
    model.eval()

    # img = '/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/test_pngs/3f503dc886d4105b86e199f6ef3bd722.png'
    img = path
    img = mmcv.imread(img)
    img_ngu = mmcv.imread(img)
    result = inference_detector(model, img)
    model.show_result(img_ngu, result,out_file="lesion_ngu.png")


    bbox_result = result.copy()
    num_samples_each_class = [bbox_result[i].shape[0] for i in range(len(bbox_result))]

    abnormal_threshold = config['THRESHOLD_4_DET']
    abnormal_prob = probs_abnormal
    keep_threshold = 0.5

    normal = abnormal_prob < abnormal_threshold

    if normal:
        for i in range(7):
            # pdb.set_trace()
            for j, instance in enumerate(bbox_result[i]):
                # if instance[-1] < keep_threshold: #score
                bbox_result[i][j] = np.array([0,0,0,0,0])

    model.show_result(img, bbox_result,out_file="lesion.png")
    

