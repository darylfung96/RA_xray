import numpy as np
import os
from collections import OrderedDict
import pandas as pd

summary_folder = 'xlsx_summary_files'
CATCH_files = [file for file in os.listdir(summary_folder) if 'CATCH' in file]

os.makedirs(os.path.join(summary_folder, 'date'), exist_ok=True)
os.makedirs(os.path.join(summary_folder, 'part'), exist_ok=True)

for file in CATCH_files:
	xlsx_file = pd.read_excel(os.path.join(summary_folder, file))

	dates = [col.split('__')[0] for col in xlsx_file.columns if '__' in col]
	unique_dates = np.unique(dates).tolist()
	unique_dates.sort()
	parts = [col.split('__')[1] for col in xlsx_file.columns if '__' in col]
	unique_parts = np.unique(parts).tolist()
	unique_parts.sort()

	all_patients = {}

	dates_dict = OrderedDict()
	for date in unique_dates:
		dates_dict[date] = 0
	parts_dict = OrderedDict()
	for part in unique_parts:
		parts_dict[part] = 0

	for patient_id in xlsx_file['patient_id']:
		all_patients[patient_id] = {}

		current_patient = xlsx_file.loc[xlsx_file['patient_id'] == patient_id]

		patient_dates_dict = OrderedDict()
		patient_dates_dict['patient_id'] = patient_id
		patient_dates_dict.update(dates_dict.copy())

		patient_parts_dict = OrderedDict()
		patient_parts_dict['patient_id'] = patient_id
		patient_parts_dict.update(parts_dict.copy())

		for date in dates_dict.keys():
			selected_date_df = current_patient.filter(regex=date)
			total_values_selected_date = selected_date_df.values.sum()
			patient_dates_dict[date] = total_values_selected_date

		for part in parts_dict.keys():
			selected_part_df = current_patient.filter(regex=part)
			total_values_selected_part = selected_part_df.values.sum()
			patient_parts_dict[part] = total_values_selected_part

		all_patients[patient_id]['date'] = patient_dates_dict
		all_patients[patient_id]['part'] = patient_parts_dict

	date_list = []
	part_list = []
	for patient in all_patients.keys():
		date_list.append(all_patients[patient]['date'])
		part_list.append(all_patients[patient]['part'])

	pd.DataFrame(date_list).to_excel(os.path.join(summary_folder, 'date', file))
	pd.DataFrame(part_list).to_excel(os.path.join(summary_folder, 'part', file))
