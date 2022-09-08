from pathlib import Path
import pandas as pd
from bounding_box import bounding_box as bb
import os
import cv2
import pdb
from tqdm import tqdm 

lesion_types = ['Osteophytes', 'Other lesions', 'Spondylolysthesis', 'Disc space narrowing',
 'Vertebral collapse', 'Foraminal stenosis', 'Surgical implant']
colors = ['yellow','purple','teal','red','green','fuchsia','blue']
color_dict = dict(zip(lesion_types, colors))


if __name__ == "__main__":
    ### Visualizing test image bounding boxes
    train_img_path = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/train_pngs"
    train_label_path = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/annotations/train.csv"

    Path("../dataset-vindr-spinexr/visualize_bounding_boxes/train_bb/").mkdir(parents=True, exist_ok=True)
    df_train = pd.read_csv(train_label_path)
    dropped_cols = ["study_id","series_id","rad_id"]
    df_train.drop(labels=dropped_cols, inplace=True,axis=1)
    
    df_train.drop(df_train[df_train['lesion_type'] == "No finding"].index, inplace = True)

    print(df_train.head(10))
    print(df_train['lesion_type'].unique())    
    print(df_train.info())

    train_imgs_dict = {}

    for index, row in df_train.iterrows():
        image_id = row['image_id'] + ".png"
        label = row['lesion_type']
        left, top, right, bottom = row['xmin'],row['ymin'],\
                                    row['xmax'],row['ymax']

        try: 
            train_imgs_dict[image_id].append([label, left, top, right, bottom])
        except:
            train_imgs_dict[image_id] = [[label, left, top, right, bottom]]

    for image_id, box_list in tqdm(train_imgs_dict.items()):
        img_path = os.path.join(train_img_path,image_id)
        image = cv2.imread(img_path)

        for bbox in box_list:

            label, left, top, right, bottom = bbox

            bb.add(image, left, top, right, bottom, label, color_dict[label])
            
        cv2.imwrite(f"../dataset-vindr-spinexr/visualize_bounding_boxes/train_bb/{image_id}", image)

    ### Visualizing test image bounding boxes
    test_img_path = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/test_pngs"
    test_label_path = "/home/dhgbao/Machine_Learning/SpineXR/dataset-vindr-spinexr/annotations/test.csv"

    Path("../dataset-vindr-spinexr/visualize_bounding_boxes/test_bb/").mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(test_label_path)
    dropped_cols = ["study_id","series_id","rad_id"]
    df_test.drop(labels=dropped_cols, inplace=True,axis=1)
    
    df_test.drop(df_test[df_test['lesion_type'] == "No finding"].index, inplace = True)

    print(df_test.head(10))
    print(df_test['lesion_type'].unique())    
    print(df_test.info())

    test_imgs_dict = {}

    for index, row in df_test.iterrows():
        image_id = row['image_id'] + ".png"
        label = row['lesion_type']
        left, top, right, bottom = row['xmin'],row['ymin'],\
                                    row['xmax'],row['ymax']

        try: 
            test_imgs_dict[image_id].append([label, left, top, right, bottom])
        except:
            test_imgs_dict[image_id] = [[label, left, top, right, bottom]]

    for image_id, box_list in tqdm(test_imgs_dict.items()):
        img_path = os.path.join(test_img_path,image_id)
        image = cv2.imread(img_path)

        for bbox in box_list:

            label, left, top, right, bottom = bbox

            bb.add(image, left, top, right, bottom, label, color_dict[label])
        cv2.imwrite(f"../dataset-vindr-spinexr/visualize_bounding_boxes/test_bb/{image_id}", image)


