import cv2
import os


def extract_joint_features_from_yolo(labels_dir, imgs_dir):
    label_filenames = os.listdir(labels_dir)

    images_boxes = []
    for label_filename in label_filenames:
        with open(os.path.join(labels_dir, label_filename), 'r') as f:
            content = f.read()
            boxes = content.split('\n')[:-1]
            boxes = [box[2:].split() for box in boxes]

            img_filename = label_filename.split('.')[0] + '.jpg'
            img = cv2.imread(os.path.join(imgs_dir, img_filename))

            height, width, channel = img.shape
            img_boxes = []
            for box in boxes:
                ratio_x, ratio_y, ratio_width, ratio_height = box
                actual_y = float(ratio_y) * height
                actual_x = float(ratio_x) * width
                actual_width = float(ratio_width) * width
                actual_height = float(ratio_height) * height

                current_img_box = img[int(actual_y - actual_height//2):int(actual_y + actual_height//2):,
                                  int(actual_x - actual_width//2):int(actual_x + actual_width//2)]
                img_boxes.append(current_img_box)
            images_boxes.append(img_boxes)
    return images_boxes


labels_dir = 'yolov5/runs/detect/exp/labels/'
imgs_dir = 'all_images/cropped_xray_images/'
images_boxes = extract_joint_features_from_yolo(labels_dir, imgs_dir)

