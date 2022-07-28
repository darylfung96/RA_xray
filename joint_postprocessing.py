import os
from glob import glob
import numpy as np
import functools
import cv2
import sys

sys.path.append('yolov5')
from yolov5.utils.plots import Annotator, colors, save_one_box


directory = os.path.join('yolov5', 'runs', 'detect', 'yolov5l6_joint_tif', 'labels')
labels = glob(f'{directory}/*.txt')


def cmp_y(item1, item2):
	y1 = float(item1[2])
	y2 = float(item2[2])
	return y1 - y2


def cmp_x(item1, item2):
	x1 = float(item1[1])
	x2 = float(item2[1])
	return x1 - x2


def divide_chunks(current_list, chunk_size):
	for i in range(0, len(current_list), chunk_size):
		yield current_list[i:i+chunk_size]


def divide_hands(current_list):
	return list(divide_chunks(current_list, 14))


def divide_fingers(current_list):
	# expect the hand receive here to be a LEFT HAND

	four_fingers = current_list[:-5]

	index_finger_thumb = current_list[-5:]
	index_finger_thumb.sort(key=functools.cmp_to_key(cmp_y))
	thumb = index_finger_thumb[-2:]
	index_finger = index_finger_thumb[:-2]

	fingers = list(divide_chunks(four_fingers, 3))

	for i in range(len(fingers)):
		fingers[i].sort(key=functools.cmp_to_key(cmp_y))

	fingers.append(index_finger)
	fingers.append(thumb)

	return fingers


def label_hands(preds):
	hands = divide_hands(preds)
	# for each hand
	for i in range(len(hands)):
		hands[i] = label_hand(hands[i], right_hand=i == 1)
	return hands


def label_hand(preds, right_hand=False):
	if right_hand:
		preds = list(reversed(preds))

	preds = divide_fingers(preds)

	# for each finger
	for finger_index in range(len(preds)):
		# remove DIP
		if len(preds[finger_index]) > 2:
			del preds[finger_index][0]
	return preds


for label in labels:
	with open(label, 'r') as file:
		content = file.read()
	preds = content.split("\n")[:-1]
	preds = [pred.split() for pred in preds]
	preds.sort(key=functools.cmp_to_key(cmp_y))
	preds.sort(key=functools.cmp_to_key(cmp_x))

	if 'hands' in label:
		preds = label_hands(preds)
	elif 'hand' in label and 'left' in label:
		preds = [label_hand(preds)]
	elif 'hand' in label and 'right' in label:
		preds = [label_hand(preds, True)]

	img = cv2.imread(label.replace('labels/', '').replace('.txt', '.tif'))

	annotator = Annotator(img, line_width=0, font_size=10)
	for hand in preds:
		for finger in hand:
			pip, mcp = finger

			# pip
			_, x, y, width, height = pip
			initial_x = float(x) * img.shape[1] - (float(width) * img.shape[1]) / 2
			initial_y = float(y) * img.shape[0] - (float(height) * img.shape[1]) / 2
			end_x = initial_x + (float(width) * img.shape[1])
			end_y = initial_y + (float(height) * img.shape[0])
			xyxy = [initial_x, initial_y, end_x, end_y]
			annotator.box_label(xyxy, 'pip', color=colors(0, True))

			# mcp
			_, x, y, width, height = mcp
			initial_x = float(x) * img.shape[1] - (float(width) * img.shape[1]) / 2
			initial_y = float(y) * img.shape[0] - (float(height) * img.shape[1]) / 2
			end_x = initial_x + (float(width) * img.shape[1])
			end_y = initial_y + (float(height) * img.shape[0])
			xyxy = [initial_x, initial_y, end_x, end_y]
			annotator.box_label(xyxy, 'mcp', color=colors(1, True))

	generated_img = annotator.result()
	cv2.imshow('generated_img', generated_img)
	cv2.waitKey()







