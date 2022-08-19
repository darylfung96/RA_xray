import cv2
from glob import glob
import os

from joint_postprocessing import arrange_prediction, label_hand, label_hands

image_dir = 'yolov5/data/RA_Jun2/'
labels_dir = 'yolov5/runs/detect/yolov5l6_joint_tif/labels/'

all_images = glob(image_dir + '*.tif')

# These are indexes
fingers = ['5', '4', '3', '2', '1']
joints = ['PIP', 'MCP']

for image in all_images:
	current_label_filename = os.path.join(labels_dir, os.path.basename(image)).replace('.tif', '.txt')

	with open(current_label_filename, 'r') as f:
		preds = f.read().split('\n')

	# arrange hands
	preds = [pred.split() for pred in preds]
	if 'hands' in current_label_filename:
		preds = preds[:28]
	elif 'hand' in current_label_filename and 'left' in current_label_filename:
		preds = preds[:14]
	elif 'hand' in current_label_filename and 'right' in current_label_filename:
		preds = preds[:14]
	preds = arrange_prediction(preds)
	if 'hands' in current_label_filename:
		preds = label_hands(preds)
	elif 'hand' in current_label_filename and 'left' in current_label_filename:
		preds = [label_hand(preds)]
	elif 'hand' in current_label_filename and 'right' in current_label_filename:
		preds = [label_hand(preds, True)]

	# preds will contain
	# [
	#   [pinky PIP, pinky MCP],
	#   [fourth PIP, fourth MCP],
	#   [middle PIP, middle MCP],
	#   [index PIP, index MCP],
	#   [thumb PIP, thumb MCP],
	# ]

	image_array = cv2.imread(image)
	x_ratio = image_array.shape[1]
	y_ratio = image_array.shape[0]

	def get_actual_coord(x, y, width, height, x_size, y_size):
		"""
		:return:    The initial x,y and the ending x,y
		"""
		initial_x = (x-width/2) * x_size
		initial_y = (y-height/2) * y_size

		end_x = initial_x + width * x_size
		end_y = initial_y + height * y_size

		return int(initial_x), int(initial_y), int(end_x), int(end_y)


	for hand in preds:
		for finger_idx, finger in enumerate(hand):
			for joint_idx, joint in enumerate(finger):
				joint = [float(number) for number in joint]
				_, x, y, width, height = joint
				initial_x, initial_y, end_x, end_y = get_actual_coord(x, y, width, height, x_ratio, y_ratio)
				cropped = image_array[initial_y:end_y, initial_x:end_x]

				new_filename = image.replace('.tif', f'{fingers[finger_idx]}_{joints[joint_idx]}.tif')
				cv2.imwrite(new_filename, cropped)

