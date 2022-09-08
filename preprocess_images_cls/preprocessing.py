from IPython.display import clear_output
import torch.nn as nn
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import pydicom as dicom
import pandas as pd
clear_output()
NUM_CLASS = 2
from pathlib import Path
from joblib import Parallel, delayed

def dicom2png_cls(image_path, label_csv, saved_image_path, size = (160, 256)):
        n = len(label_csv.image_id)
        for i in tqdm(range(n)):
            dcm_file_path = f"{image_path}{label_csv.image_id[i]}.dicom"
            dcm_file = dicom.dcmread(dcm_file_path)
            if dcm_file.BitsStored in (10,12):
                    dcm_file.BitsStored = 16
            data = dcm_file.pixel_array
            if dcm_file.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data

            data = (data - np.min(data)) / np.max(data)
            data = (data * 255.0).astype(np.uint8)
            image = Image.fromarray(data).resize(size)
            image.save(f"{saved_image_path}{label_csv.image_id[i]}.png")

if __name__ == "__main__":
    Path("/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/train_images_png").mkdir(parents=True, exist_ok=True)
    Path("/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/test_images_png").mkdir(parents=True, exist_ok=True)


    image_train = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/train_images/"
    label_train = pd.read_csv("/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/annotations/train_classify.csv")
    saved_image_cls_train = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/train_images_png/"
    image_test = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/test_images/"
    label_test = pd.read_csv("/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/annotations/test_classify.csv")
    saved_image_cls_test = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/test_images_png/"

    executor = Parallel(n_jobs=20, backend='multiprocessing', prefer='processes', verbose=1)
    delayed(dicom2png_cls(image_train, label_train, saved_image_cls_train))

    executor = Parallel(n_jobs=20, backend='multiprocessing', prefer='processes', verbose=1)
    delayed(dicom2png_cls(image_test, label_test, saved_image_cls_test))
    