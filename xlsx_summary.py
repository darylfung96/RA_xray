import pandas as pd
import numpy as np
import os
import glob

processed_xlsx_folder = 'processed_xlsx_RA_files'

patients_info = []
total_patients = 0
summary_key = {}  # {'item': 'count' , ...}
for xlsx_file in glob.glob(f'{processed_xlsx_folder}/*.xlsx'):
	df = pd.read_excel(xlsx_file)

	num_patients = df.shape[0]
	total_patients += num_patients

	num_timepoints = len([col for col in df.columns if df.columns[2][:-3] in col])

	# get all columns not containing TOTAL SCORES
	only_joints_df = df[[col for col in df.columns[2:] if 'TOTAL SCORES' not in col]]
	summary = np.unique(only_joints_df.values.astype(np.str), return_counts=True)
	for idx, key in enumerate(summary[0]):
		if key == 'C':
			print('a')
		if summary_key.get(key, None) is None:
			summary_key[key] = summary[1][idx]
		else:
			summary_key[key] += summary[1][idx]

	for patient_id in df['Patient ID']:
		current_patient_info = {}
		current_patient_info['Patient ID'] = patient_id
		current_patient_info['num_timepoints'] = num_timepoints
		patients_info.append(current_patient_info)

pd.DataFrame(list(summary_key.items())).to_excel('joint_summary.xlsx')
pd.DataFrame(patients_info).to_excel('patient_info.xlsx')


