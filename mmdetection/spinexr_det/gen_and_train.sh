python3 /home/dhgbao/Machine_Learning/SpineXR/detection_task/scripts/voc_to_yolo.py

python3 /home/dhgbao/Machine_Learning/SpineXR/detection_task/scripts/Yolo-to-COCO-format-converter/main.py --yolo-subdir  --path /home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/train_pngs --output /home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/annotations/COCO/train.json
python3 /home/dhgbao/Machine_Learning/SpineXR/detection_task/scripts/Yolo-to-COCO-format-converter/main.py --yolo-subdir --path /home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/test_pngs --output /home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/annotations/COCO/test.json

python3 ../tools/train.py "/home/dhgbao/Machine_Learning/SpineXR/detection_task/mmdetection/configs/vfnet/vfnet_config.py"