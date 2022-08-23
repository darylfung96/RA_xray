# RA_xray

## Getting Started
Run joint detection by using yolov5/detect.py with these arguments:
```angular2html
python detect.py --source "data/RA_Jun2/*hand*" --weights joint_all_multi.pt --save-txt --conf-thres 0.35 --iou-thres 0.1
```

feel free to change:
--source image directory to detect joint

keep other settings as it is

---
Once detection is done, it should appear in yolov5/runs/detect/exp*

There should also be a labels folder. The labels folder in there are required for joint extraction.

---

To do joint extraction, run:
```angular2html
python joint_multi_extraction.py
```
feel free to change the variables:

- **image_dir**
- **joints_feature_output_dir**
- **labels_dir**

image_dir is the image directory that you want to extract the joints from. This should be kept the same as --source when running detect.py

joints_feature_output_dir is the output folder for the extracted joint

labels_dir is the directory where the labels directory is created after running detect.py
