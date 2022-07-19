import pandas as pd
from PIL import Image
import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import wandb


class ObtainRADataset:
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

		self.train_images = []
		self.test_images = []
		self.train_labels = []
		self.test_labels = []
		self.train_label_scale = MinMaxScaler()
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
		self.labels = np.expand_dims(labels, -1)

		self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.images,
		                                                                                            self.labels,
		                                                                                            test_size=0.2)
		self.train_labels = self.train_label_scale.fit_transform(self.train_labels)
		self.test_labels = self.train_label_scale.transform(self.test_labels)

class RADataset(Dataset):
	def __init__(self, images, labels):
		super(RADataset, self).__init__()
		self.images = images
		self.labels = labels

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = Image.open(self.images[idx]).convert("RGB")
		preprocess = torchvision.transforms.Compose([
			torchvision.transforms.Resize((256, 256)),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		image = preprocess(image)

		label = self.labels[idx]
		return image, label


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.model = torchvision.models.resnet18(pretrained=True)
		self.fcn = nn.Sequential(
			nn.Linear(1000, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 1)
		)

	def forward(self, inputs):
		return self.fcn(self.model(inputs))

device = 'cuda'
wandb.init(project='RA severity classification', name='resnet')
obtain_ra_dataset = ObtainRADataset('all_RA_Jun2', 'all_CATCH_with_filename.xlsx')

train_ra_dataset = RADataset(obtain_ra_dataset.train_images, obtain_ra_dataset.train_labels)
train_data_loader = DataLoader(train_ra_dataset, batch_size=32, shuffle=True)

test_ra_dataset = RADataset(obtain_ra_dataset.test_images, obtain_ra_dataset.test_labels)
test_data_loader = DataLoader(test_ra_dataset, batch_size=32, shuffle=False)


network = Net().to(device)
wandb.watch(network, log='all')
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

for current_epoch in range(200):
	data_tqdm = tqdm.tqdm(train_data_loader)
	for images, labels in data_tqdm:
		images = images.to(device).float()
		labels = labels.to(device).float()

		outputs = torch.sigmoid(network(images))

		optimizer.zero_grad()
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		wandb.log({
			'train_loss': loss.item()
		})
		wandb_images = wandb.Image(images)
		wandb.log({
			'images': wandb_images
		})

	test_data_tqdm = tqdm.tqdm(test_data_loader)
	for test_images, test_labels in test_data_tqdm:
		test_images = test_images.to(device).float()
		test_labels = test_labels.to(device).float()

		outputs = torch.sigmoid(network(test_images))
		test_loss = criterion(outputs, test_labels)

		wandb.log({
			'test_loss': test_loss.item()
		})
		wandb_images = wandb.Image(test_images)
		wandb.log({
			'images': wandb_images
		})
		data_tqdm.set_description(f"current epoch: {current_epoch} test loss: {test_loss}")


	table = wandb.Table(columns=['Image', 'Predicted Severity Score Errosion (normalized)',
	                             'Predicted Severity Score Errosion',
	                             'Ground Truth Score Errosion (normalized)', 'Ground Truth Score Errosion'])
	scaled_labels = obtain_ra_dataset.train_label_scale.inverse_transform(test_labels.cpu().detach().numpy())
	scaled_outputs = obtain_ra_dataset.train_label_scale.inverse_transform(outputs.cpu().detach().numpy())

	for img, output, scaled_output, scaled_label, label in zip(test_images, outputs, scaled_outputs, scaled_labels, labels):
		wandb_img = wandb.Image(img)
		table.add_data(wandb_img, output[0], scaled_output[0], label[0], scaled_label[0])

	wandb.log({f"Test Data Table {current_epoch}": table})
