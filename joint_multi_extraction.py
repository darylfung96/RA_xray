import cv2
import functools
from glob import glob
import os
import yaml

from joint_postprocessing import generate_cmp

image_dir = 'yolov5/data/RA_Jun2/'
joints_feature_output_dir = 'yolov5/data/extracted_joint_RA_Jun2'

labels_dir = 'yolov5/runs/detect/joint_all_multi/labels/'
label_info = 'yolov5/data/joint_all_multi.yaml'

os.makedirs(joints_feature_output_dir, exist_ok=True)


def get_actual_coord(x, y, width, height, x_size, y_size):
	"""
	:return:    The initial x,y and the ending x,y
	"""

	x, y, width, height = float(x), float(y), float(width), float(height)

	initial_x = (x - width / 2) * x_size
	initial_y = (y - height / 2) * y_size

	end_x = initial_x + width * x_size
	end_y = initial_y + height * y_size

	return int(initial_x), int(initial_y), int(end_x), int(end_y)


def save_extraction(selected_predictions, current_label, image_name, is_right_hand=False):
	hand_direction = 'right' if is_right_hand else 'left'

	for idx_prediction, selected_prediction in enumerate(selected_predictions):
		_, x, y, width, height = selected_prediction
		initial_x, initial_y, end_x, end_y = get_actual_coord(x, y, width, height, x_ratio, y_ratio)
		cropped = image_array[initial_y:end_y, initial_x:end_x]

		new_image_name = image_name.replace(os.path.dirname(image_name), joints_feature_output_dir)
		new_filename = new_image_name.replace('.tif', f'_{hand_direction}_{current_label}_{idx_prediction + 1}.tif')
		cv2.imwrite(new_filename, cropped)


with open(label_info, 'r') as f:
	labels = yaml.safe_load(f)['names']

all_images = glob(image_dir + '*.tif')

for image in all_images:
	current_label_filename = os.path.join(labels_dir, os.path.basename(image)).replace('.tif', '.txt')
	image_array = cv2.imread(image)
	x_ratio = image_array.shape[1]
	y_ratio = image_array.shape[0]

	with open(current_label_filename, 'r') as f:
		preds = f.read().split('\n')

	# arrange hands
	preds = [pred.split() for pred in preds][:-1]
	preds.sort()  # sort by labels

	# deal with each prediction (PIP individually, MCP individually, ...)
	for label_idx, current_label in enumerate(labels):

		selected_predictions = [pred for pred in preds if pred[0] == str(label_idx)]
		# sort on x
		x_index = 1
		cmp_function = generate_cmp(x_index)
		selected_predictions.sort(key=functools.cmp_to_key(cmp_function))

		if 'hands' in image.lower():
			# reverse because the numbering will be from 1,2,3,4,5 for right hand and is inverse for left hand
			left_hand = reversed(selected_predictions[:len(selected_predictions)//2])
			right_hand = selected_predictions[len(selected_predictions)//2:]
			save_extraction(left_hand, current_label, image, is_right_hand=False)
			save_extraction(right_hand, current_label, image, is_right_hand=True)
		elif 'hand' in image.lower() and 'left' in image.lower():
			left_hand = reversed(selected_predictions)
			save_extraction(left_hand, current_label, image, is_right_hand=False)
		elif 'hand' in image.lower() and 'right' in image.lower():
			right_hand = selected_predictions
			save_extraction(right_hand, current_label, image, is_right_hand=True)

