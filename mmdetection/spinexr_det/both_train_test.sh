python3 train.py "config.py"
python3 test.py spine_configs/test_config.py spinexr_exps/latest.pth --work-dir spinexr_exps/ --eval 'bbox' --show-dir spinexr_exps/result_images/
