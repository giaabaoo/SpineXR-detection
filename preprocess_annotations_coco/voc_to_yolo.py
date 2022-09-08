import os
import numpy as np
from tqdm import tqdm
import os
import cv2
import json
import pdb
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    # Path(os.path.join("/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/annotations/",'COCO')).mkdir(parents=True, exist_ok=True)

    # ### TRAIN SET
    # train_labels_path = "/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/annotations/train.csv"
    # train_images_path = "/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/train_pngs"
    # train_labels_yolo_path = "/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/train_pngs/YOLO_darknet"

    # train_df = pd.read_csv(train_labels_path)
    # dropped_cols = ["study_id","series_id","rad_id"]
    # train_df.drop(labels=dropped_cols, inplace=True,axis=1)

    # train_df.drop(train_df[train_df['lesion_type'] == "No finding"].index, inplace = True)

    # print(train_df['lesion_type'].unique())

    # cat2label = {k: i for i, k in enumerate(train_df['lesion_type'].unique())}
    
    # image_dict_train = dict()

    # for idx, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    #     image_id, lesion_type, xmin, ymin, xmax, ymax = row['image_id'], row['lesion_type'], row['xmin'], row['ymin'], row['xmax'], row['ymax']
    #     image_name = image_id + '.png'
    #     image_path = os.path.join(train_images_path, image_name)

    #     box_width = np.abs(np.abs(xmax) - np.abs(xmin))
    #     box_height = np.abs(np.abs(ymax) - np.abs(ymin))

    #     height, width, _ = cv2.imread(image_path).shape

    #     x = (xmax + xmin)/2/width
    #     y = (ymax + ymin)/2./height
    #     w = box_width / width
    #     h = box_height / height

    #     bounding_boxes = [lesion_type, x, y, w, h]

    #     if idx == 20:
    #         break

    #     try:
    #         image_dict_train[image_name].append(bounding_boxes)
    #     except:
    #         image_dict_train[image_name] = [bounding_boxes]

    # with open('./train_set_by_image_name.json','w') as f:
    #     json.dump(image_dict_train, f, indent=4)


    # Path(train_labels_yolo_path).mkdir(parents=True, exist_ok=True)

    # for image_name, bounding_boxes in tqdm(image_dict_train.items()):
    #     label_path = image_name.replace(".png",'.txt')

    #     with open(f"{train_labels_yolo_path}/{label_path}","w") as f:
    #         lines = []
    #         for bounding_box in bounding_boxes:
    #             lesion_type, x, y, w, h = bounding_box
    #             lesion_type = cat2label[lesion_type]
    #             lines.append(f"{lesion_type} {x} {y} {w} {h}\n")

    #         f.writelines(lines)

    ### TEST SET
    test_labels_path = "/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/annotations/test.csv"
    test_images_path = "/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/test_pngs"
    test_labels_yolo_path = "/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/test_pngs/YOLO_darknet"

    test_df = pd.read_csv(test_labels_path)
    dropped_cols = ["study_id","series_id","rad_id"]
    test_df.drop(labels=dropped_cols, inplace=True,axis=1)

    # test_df.drop(test_df[test_df['lesion_type'] == "No finding"].index, inplace = True)

    print(test_df['lesion_type'].unique())

    image_dict_test = dict()

    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        image_id, lesion_type, xmin, ymin, xmax, ymax = row['image_id'], row['lesion_type'], row['xmin'], row['ymin'], row['xmax'], row['ymax']
        image_name = image_id + '.png'
        image_path = os.path.join(test_images_path, image_name)

        if lesion_type != "No finding":
            box_width = np.abs(np.abs(xmax) - np.abs(xmin))
            box_height = np.abs(np.abs(ymax) - np.abs(ymin))

            height, width, _ = cv2.imread(image_path).shape

            x = (xmax + xmin)/2/width
            y = (ymax + ymin)/2./height
            w = box_width / width
            h = box_height / height

            bounding_boxes = [lesion_type, x, y, w, h]
        else:
            bounding_boxes = []

        try:
            image_dict_test[image_name].append(bounding_boxes)
        except:
            image_dict_test[image_name] = [bounding_boxes]

        # if idx == 20:
        #     break

    with open('./test_set_by_image_name.json','w') as f:
        json.dump(image_dict_test, f, indent=4)
        # image_dict_test = json.load(f)


    Path(test_labels_yolo_path).mkdir(parents=True, exist_ok=True)
    cat2label = {k: i for i, k in enumerate(test_df['lesion_type'].unique())}


    for image_name, bounding_boxes in tqdm(image_dict_test.items()):
        label_path = image_name.replace(".png",'.txt')

        with open(f"{test_labels_yolo_path}/{label_path}","w") as f:
            lines = []
            for bounding_box in bounding_boxes:
                if len(bounding_box) == 0:
                    continue
                lesion_type, x, y, w, h = bounding_box
                lesion_type = cat2label[lesion_type]
                lines.append(f"{lesion_type} {x} {y} {w} {h}\n")

            f.writelines(lines)
