import pytorch_lightning as pl
from collections import OrderedDict
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset, DataLoader


class RADataset(Dataset):
	def __init__(self, tif_folder, xlsx_filename):
		total_scores_column = ['TOTAL SCORES LF E  ',
		                       'TOTAL SCORES LF J  ',
		                       'TOTAL SCORES LH E  ',
		                       'TOTAL SCORES LH J  ',
		                       'TOTAL SCORES RF E  ',
		                       'TOTAL SCORES RF J  ',
		                       'TOTAL SCORES RH E  ',
		                       'TOTAL SCORES RH J  ']

		patient_information = pd.read_excel(xlsx_filename)
		self.patient_total_scores = patient_information[['Patient ID', 'Timepoint'] + total_scores_column]
		self.patient_filename = patient_information[['filename_RH', 'filename_LH', 'filename_RF', 'filename_LF']]
		self.tif_folder = tif_folder

		# remove rows that does not contain a single valid filename
		self.patient_total_scores = self.patient_total_scores[~self.patient_filename.isnull().all(1)]
		self.patient_filename = self.patient_filename[~self.patient_filename.isnull().all(1)]

		self.images = []
		self.labels = []
		self.__initialize_images_labels()

	def __initialize_images_labels(self):
		images = []
		labels = []
		for idx in range(len(self.patient_total_scores)):
			filename_RH, filename_LH, filename_RF, filename_LF = self.patient_filename.iloc[idx]

			# hands ########################
			if type(filename_RH) == str:  # sometimes filename is NaN
				# if it is only right
				if 'right' in filename_RH.lower():
					score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES RH E  '])
					if score.lower() != 'missing' and score.lower() != 'nan':
						errosion_rh = int(score)
						images.append(os.path.join(self.tif_folder, filename_RH))
						labels.append(errosion_rh)

			if type(filename_LH) == str:  # sometimes filename is NaN
				if 'left' in filename_LH.lower():
					score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES LH E  '])
					# make sure score not missing
					if score.lower() != 'missing' and score.lower() != 'nan':
						errosion_lh = int(score)
						images.append(os.path.join(self.tif_folder, filename_LH))
						labels.append(errosion_lh)

			if type(filename_LH) == str and type(filename_RH) == str:
				if 'right' not in filename_RH.lower() or 'left' not in filename_LH.lower():
					right_score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES RH E  '])
					left_score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES LH E  '])

					if 'missing' not in right_score.lower() and 'missing' not in left_score.lower():
						errosion_h = int(right_score) + \
						             int(left_score)
						images.append(os.path.join(self.tif_folder, filename_RH))
						labels.append(errosion_h)

			# feet ########################
			if type(filename_RF) == str:  # sometimes filename is NaN
				# if it is only right
				if 'right' in filename_RF.lower():
					score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES RF E  '])
					if score.lower() != 'missing' and score.lower() != 'nan':
						errosion_rf = int(score)
						images.append(os.path.join(self.tif_folder, filename_RF))
						labels.append(errosion_rf)

			if type(filename_LF) == str:  # sometimes filename is NaN
				if 'left' in filename_LF.lower():
					score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES LF E  '])
					if score.lower() != 'missing' and score.lower() != 'nan':
						errosion_lf = int(score)
						images.append(os.path.join(self.tif_folder, filename_LF))
						labels.append(errosion_lf)

			if type(filename_RF) == str and type(filename_LF) == str:  # sometimes filename is NaN
				if 'right' not in filename_RF.lower() or 'left' not in filename_LF.lower():
					right_score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES RF E  '])
					left_score = str(self.patient_total_scores.iloc[idx]['TOTAL SCORES LF E  '])
					if 'missing' not in right_score.lower() and 'missing' not in left_score.lower():
						errosion_f = int(right_score) + \
						             int(left_score)
						images.append(os.path.join(self.tif_folder, filename_RF))
						labels.append(errosion_f)

		self.images = images
		self.labels = labels

	def __len__(self):
		return len(self.patient_filename)

	def __getitem__(self, idx):
		image = cv2.imread(self.images[idx])
		image = cv2.resize(image, (512, 512))
		label = self.labels[idx]
		return image, label


data_loader = DataLoader(RADataset('all_RA_Jun2', 'all_CATCH_with_filename.xlsx'), batch_size=6)
for images, labels in data_loader:
	print(labels)