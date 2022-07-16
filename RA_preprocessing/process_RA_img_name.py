import numpy as np
import os
import glob
import pandas as pd
import shutil

CATCH_folders = os.listdir('RA_Jun2/')
destination_folder = 'all_RA_Jun2'

os.makedirs(destination_folder, exist_ok=True)

timepoint_alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
timepoint_range = [0, 3, 6, 9, 12, 18, 24, 36]


def convert_image_date_to_timepoint(current_patient_id, image_names, dates, timepoints):
	new_image_names = []

	# deal with different length in excel and number of images timepoints
	if len(dates) != len(timepoints):
		dates_dict = {}
		first_date = dates[0]
		dates_dict[first_date] = 'A'
		initial_year, initial_month, _ = first_date.split('-')

		for date in dates[1:]:
			current_year, current_month, _ = date.split('-')
			month_difference = int(current_month) - int(initial_month)
			year_difference = int(current_year) - int(initial_year)

			total_month_difference = year_difference * 12 + month_difference
			closest_month = min(timepoint_range, key = lambda x: abs(x-total_month_difference))
			alphabet = timepoint_alphabets[timepoint_range.index(closest_month)]
			dates_dict[date] = alphabet
		for image_name in image_names:
			list_image_name = image_name.split("_")
			current_date = list_image_name[-2]
			list_image_name[0] = current_patient_id
			list_image_name[-2] = dates_dict[current_date]
			new_image_name = '_'.join(list_image_name)
			new_image_names.append(new_image_name)

			if new_image_name == 'A_FootLeft.tif':
				print(current_patient_id)
				print(image_name)
				print(list_image_name)
				print(dates_dict[current_date])

	# deal with same length in excel and number of images timepoints
	else:
		dates_dict = {date: idx for idx, date in enumerate(dates)}
		for image_name in image_names:
			# get date
			list_image_name = image_name.split("_")
			date = list_image_name[-2]
			date_index = dates_dict[date]

			timepoint = non_missing_df.iloc[date_index]['Timepoint']
			list_image_name[0] = current_patient_id
			list_image_name[-2] = timepoint
			new_image_name = '_'.join(list_image_name)
			new_image_names.append(new_image_name)

			if new_image_name == 'A_FootLeft.tif':
				print(current_patient_id)
				print(image_name)
				print(list_image_name)

	return new_image_names


for CATCH_folder in CATCH_folders:
	xlsx_file = glob.glob(os.path.join('RA_Jun2/', CATCH_folder, '*.xlsx'))
	df = pd.read_excel(xlsx_file[0])

	unique_patient_id = np.unique(df['Patient ID'])
	for current_patient_id in unique_patient_id:
		# get excel information
		patient_df = df.loc[df['Patient ID'] == current_patient_id]
		non_missing_df = patient_df[~patient_df['TOTAL SCORES E + J  '].astype(str).str.contains('Missing', na=False)]

		try:
			# get images
			image_names = os.listdir(os.path.join('RA_Jun2', CATCH_folder, current_patient_id))
			image_names.sort(key=lambda x: x.split("_")[-2])

			# separate hand and feet because some of them are taken at different dates but treated as same timepoint
			hand_image_names = [image_name for image_name in image_names if 'hand' in image_name.lower()]
			foot_image_names = [image_name for image_name in image_names if 'foot' in image_name.lower() or 'feet' in image_name.lower()]
			hand_foot_image_names = [hand_image_names, foot_image_names]

			for image_names in hand_foot_image_names:
				# get unique dates
				dates = np.unique([image_name.split("_")[-2] for image_name in image_names])
				dates.sort()

				# get the category from the date of image name
				new_image_names = convert_image_date_to_timepoint(current_patient_id, image_names, dates, non_missing_df['Timepoint'])

				for i in range(len(new_image_names)):
					shutil.copy(os.path.join('RA_Jun2', CATCH_folder, current_patient_id, image_names[i]),
					            os.path.join(destination_folder, new_image_names[i]))


		except FileNotFoundError:
			continue

